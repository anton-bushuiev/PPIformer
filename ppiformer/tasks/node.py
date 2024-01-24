from typing import Optional, Union
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import pandas as pd

from ppiformer.model.ppiformer import PPIformer
from ppiformer.utils.typing import DDG_INFERENCE_TYPE
from ppiformer.utils.torch import ScaledTanh


class DDGPPIformer(PPIformer):
    def __init__(
        self,
        kind: DDG_INFERENCE_TYPE = 'wt_marginals',
        correction: bool = False,
        test_csv_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.kind = kind
        if self.kind == 'embedding_difference':
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim // 2, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hparams.embedding_dim // 2, 1, bias=False),
                # ScaledTanh(low=-12.0, high=12.0)
            )
        elif self.kind == 'embedding_concatenation':
            self.head = torch.nn.Sequential(
                torch.nn.Linear(2 * self.hparams.embedding_dim, (2 * self.hparams.embedding_dim) // 2),
                torch.nn.ReLU(),
                torch.nn.Linear((2 * self.hparams.embedding_dim) // 2, 1),
                # ScaledTanh(low=-12.0, high=12.0)
            )
        else:
            self.head = None

        self.correction = correction
        if self.correction:
            # self.corrector = torch.nn.Linear(1, 1, bias=False)
            self.corrector = torch.nn.Linear(1, 1)
            self.corrector.weight.data.fill_(1.0)
            self.corrector.bias.data.fill_(0.0)

        self.test_csv_path = test_csv_path
        if self.test_csv_path is not None:
            self.test_csv_path = Path(self.test_csv_path)
            self.test_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.df_test = []

    def forward(
        self,
        data: Data,
        kind: Optional[DDG_INFERENCE_TYPE] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        if kind is None:
            kind = self.kind

        # Forward
        retval = self.predict_ddg(data, kind=kind, return_attn=return_attn)
        if return_attn:
            ddg_pred, attns = retval
        else:
            ddg_pred = retval
            
        # Correction
        if self.correction:
            ddg_pred = self.corrector(ddg_pred.unsqueeze(-1)).squeeze(-1)

        # Return
        if return_attn:
            return ddg_pred, attns
        else:
            return ddg_pred

    def step(
        self,
        data: Data,
        kind: Optional[DDG_INFERENCE_TYPE] = None,
    ) -> dict:
        # Predict
        ddg_pred = self(data, kind=kind)

        # Get labels
        ddg_true = data.mut_ddg

        # Calculate loss
        # NOTE: The loss is averaged over all nodes, not avergaing over graphs first.
        # This may bias training towards graphs with more mutations unless there is 
        # a sampling of equal number of mutation per graph in a data lodaer (as done by default).
        loss = F.mse_loss(ddg_pred, ddg_true)

        return dict(
            ddg_true=ddg_true,
            ddg_pred=ddg_pred,
            loss=loss,
        )
    
    def training_step(self, data, batch_idx):
        # Make step
        out = self.step(data, kind=self.kind)
        ddg_true = out['ddg_true']
        ddg_pred = out['ddg_pred']
        loss = out['loss']

        # Log loss
        self.log(
            'train_loss_step',
            loss,
            batch_size=data.n_muts.sum(),
            sync_dist=True
        )

        # Evaluate ddG prediction
        self.evaluate_ddg_step(
            ddg_pred,
            ddg_true,
            batch=data.mut_ddg_batch,
            metric_pref='train_',
            visualize_example=batch_idx == self.visualize_batch_idx
        )

        return loss
    
    def validation_step(self, data, batch_idx, dataloader_idx=0):
        metric_pref = 'val_'
        metric_suff = self.val_metric_suffs[dataloader_idx]

        # Make step
        out = self.step(data, kind=self.kind)
        ddg_true = out['ddg_true']
        ddg_pred = out['ddg_pred']
        loss = out['loss']

        # Log loss
        self.log(
            f'{metric_pref}loss_step{metric_suff}',
            loss,
            batch_size=data.n_muts.sum(),
            sync_dist=True,
            add_dataloader_idx=False
        )

        # Evaluate ddG prediction
        if len(ddg_pred):
            self.evaluate_ddg_step(
                ddg_pred,
                ddg_true,
                batch=data.mut_ddg_batch,
                metric_pref=metric_pref,
                metric_suff=metric_suff,
                visualize_example=(
                    batch_idx == self.visualize_batch_idx and 
                    dataloader_idx == self.visualize_dataloader_idx
                )
            )

    def test_step(self, data, batch_idx, dataloader_idx=0):
        # Make step
        out = self.step(data, kind=self.kind)
        ddg_true = out['ddg_true'].cpu().numpy()
        ddg_pred = out['ddg_pred'].cpu().numpy()

        # Convert to dataframe
        skempi_pdb = sum(data.skempi_pdb, start=[])
        skempi_mut = sum(data.skempi_mut, start=[])

        # Append rows to test dataframe
        assert len(skempi_pdb) == len(skempi_mut) == len(ddg_true) == len(ddg_pred)
        for row in zip(skempi_pdb, skempi_mut, ddg_true, ddg_pred):
            self.df_test.append(row)

    def on_test_epoch_end(self):
        # Save to disk
        self.df_test = pd.DataFrame(self.df_test, columns=['#Pdb', 'Mutation(s)_cleaned', 'ddG', 'ddG_pred'])
        self.df_test.to_csv(self.test_csv_path, index=False)

    def get_checkpoint_monitors(self) -> list[tuple[str, str]]:
        monitors = super().get_checkpoint_monitors()
        for suff in self.val_metric_suffs: 
            monitors.append((f'val_spearman_per_ppi{suff}', 'max'))
            monitors.append((f'val_precision_thr=0.0{suff}', 'max'))
            monitors.append((f'val_precision_thr=-0.5{suff}', 'max'))
            monitors.append((f'val_auroc{suff}', 'max'))
        return monitors
