from typing import Iterable, Literal, Optional
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_mean_pool, global_add_pool

from ppiformer.model.ppiformer import PPIformer
from ppiformer.utils.typing import DDG_INFERENCE_TYPE
from ppiformer.utils.torch import contains_nan_or_inf
from ppiformer.data.transforms import MaskedModelingTransform


class MaskedModelingPPIformer(PPIformer):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_smoothing_prior: Literal['uniform'] = 'uniform',
        focal_loss_gamma: float = 0.0,
        class_weights: bool = False,
        log_accuracy_per_class: bool = False,
        val_ddg_kinds: Iterable[DDG_INFERENCE_TYPE] = ('wt_marginals', 'masked_marginals'),
        **kwargs
    ) -> None:
        """_summary_

        Args:
            log_accuracy_per_class (bool, optional): _description_. Defaults to False.
            val_ddg_kinds (Sequence[DDG_INFERENCE_TYPE], optional): _description_. 
                Defaults to ('wt_marginals', 'masked_marginals').
        """
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.label_smoothing_prior = label_smoothing_prior
        self.focal_loss_gamma = focal_loss_gamma
        if class_weights:
            self.class_weights = torch.tensor([
                0.02610711, 0.14068244, 0.0361165 , 0.03169744, 0.04441645,
                0.02823625, 0.07922916, 0.03454453, 0.03701476, 0.02115424,
                0.08276267, 0.04613354, 0.0444467 , 0.04844361, 0.03427209,
                0.03127829, 0.0341901 , 0.02913087, 0.12340829, 0.04673495
            ])
        else:
            self.class_weights = None
        self.log_accuracy_per_class = log_accuracy_per_class
        self.val_ddg_kinds = val_ddg_kinds

    def forward(self, data: Data) -> torch.Tensor:
        """
        Modifies super().forward() by:
            1. using masked features `data.f_masked` of the `~data.node_mask` nodes.
            2. returning only outputs for the `~data.node_mask` nodes
        """
        # Set masked node features and node mask
        f = data.f
        data.f = data.f_masked

        # Predict
        y_logits = super().forward(data, classifier=True)[~data.node_mask]

        # Set unmasked features back
        data.f = f

        return y_logits
    
    def step(self, data: Data) -> dict:
        # Predict
        y_logits = self(data)
        y_proba = y_logits.softmax(dim=-1)

        # Get labels
        y_true = data.y[~data.node_mask]

        # Calculate average loss per sample (graph) to remove bias for samples with more nodes
        if self.label_smoothing_prior != 'uniform':
            raise NotImplementedError()
        
        # Init class weights
        if self.class_weights is not None:
            weight = self.class_weights.to(self.device)
        else:
            weight = None

        # Calcualte without reduction
        loss = F.cross_entropy(
            y_logits,
            y_true,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        # Apply focal loss
        if self.focal_loss_gamma > 0:
            wt_proba = self.calc_wt_proba(y_proba, y_true)
            loss = loss * (1 - wt_proba) ** self.focal_loss_gamma

        assert not contains_nan_or_inf(loss), 'Loss contains NaN or Inf'

        # Calculate perplexity and average loss
        # NOTE: Perplexity is estimated per-residue, as in ProteinMPNN
        perplexity = global_mean_pool(loss.exp(), batch=data.batch[~data.node_mask]).mean()
        loss = global_mean_pool(loss, batch=data.batch[~data.node_mask]).mean()

        return dict(
            y_true=y_true,
            y_logits=y_logits,
            y_proba=y_proba,
            loss=loss,
            perplexity=perplexity
        )
    
    def training_step(self, data, batch_idx):
        # Make step
        out = self.step(data)
        y_true = out['y_true']
        y_proba = out['y_proba']
        loss = out['loss']
        perplexity = out['perplexity']

        # Log loss and perplexity
        self.log(
            'train_loss_step',
            loss,
            batch_size=data.num_graphs,
            sync_dist=True
        )
        self.log(
            'train_perplexity_step',
            perplexity,
            batch_size=data.num_graphs,
            sync_dist=True
        )

        # Evaluate classification performance
        self.evaluate_classification_step(y_proba, y_true, metric_pref='train_')
        
        return loss
    
    def on_training_epoch_end(self):
        self.evaluate_classification_epoch_end(metric_pref='train_')
    
    def validation_step(self, data, batch_idx, dataloader_idx=0):
        metric_pref = 'val_'
        metric_suff = self.val_metric_suffs[dataloader_idx]

        # Make step
        out = self.step(data)
        y_true = out['y_true']
        y_proba = out['y_proba']
        loss = out['loss']
        perplexity = out['perplexity']

        # Log loss and perplexity
        self.log(
            f'{metric_pref}loss_step{metric_suff}',
            loss,
            batch_size=data.num_graphs,
            sync_dist=True,
            add_dataloader_idx=False
        )
        self.log(
            f'{metric_pref}perplexity_step{metric_suff}',
            perplexity,
            batch_size=data.num_graphs,
            sync_dist=True,
            add_dataloader_idx=False
        )

        # Evaluate classification performance
        self.evaluate_classification_step(
            y_proba,
            y_true,
            metric_pref=metric_pref,
            metric_suff=metric_suff,
        )

        # Evaluate zero-shot ddG performance
        if hasattr(data, 'n_muts') and data.n_muts.sum() > 0:
            for kind in self.val_ddg_kinds:
                # Predict ddG
                ddg_pred = self.predict_ddg(data, kind=kind)

                # Evaluate
                ddg_true = data.mut_ddg
                self.evaluate_ddg_step(
                    ddg_pred,
                    ddg_true,
                    batch=data.mut_ddg_batch,
                    metric_pref=metric_pref,
                    metric_suff=f'_{kind}{metric_suff}',
                    visualize_example=(
                        batch_idx == self.visualize_batch_idx and 
                        dataloader_idx == self.visualize_dataloader_idx
                    )
                )

                # Evaluate single-point performance
                # Warning: padding value assumed to be -1
                mask_sp = torch.sum(data.mut_pos != -1, dim=-1) == 1
                if mask_sp.sum() > 0:
                    self.evaluate_ddg_step(
                        ddg_pred[mask_sp],
                        ddg_true[mask_sp],
                        batch=data.mut_ddg_batch[mask_sp],
                        metric_pref=metric_pref,
                        metric_suff=f'_{kind}_sp{metric_suff}'
                    )

    def on_validation_epoch_end(self):
        for suff in self.val_metric_suffs:
            self.evaluate_classification_epoch_end(metric_pref='val_', metric_suff=suff)

    def get_checkpoint_monitors(self) -> list[tuple[str, str]]:
        monitors = super().get_checkpoint_monitors()
        monitors.extend([(f'val_acc{suff}', 'max') for suff in self.val_metric_suffs])
        # TODO: Generalize. Now we assume the -1 dataloder to return ddG annotations
        ddg_suff = self.val_metric_suffs[-1]
        for ddg_kind in self.val_ddg_kinds:
            monitors.append((f'val_pearson_{ddg_kind}{ddg_suff}', 'max'))
            monitors.append((f'val_spearman_per_ppi_{ddg_kind}{ddg_suff}', 'max'))
            monitors.append((f'val_precision_thr=0.0_{ddg_kind}{ddg_suff}', 'max'))
        return monitors
        