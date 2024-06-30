import typing
from typing import Optional, Sequence, Any, Iterable

import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torchmetrics import Metric, Accuracy, SpearmanCorrCoef, PearsonCorrCoef, MeanMetric, ConfusionMatrix
from torchmetrics.aggregation import SumMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAUROC
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from equiformer_pytorch import Equiformer

from ppiformer.model.egnn_clean import EGNN
from ppiformer.data.transforms import DDGLabelPretransform, PreSE3TransformerTransform, MaskedModelingTransform, PreEquiformerTransform
from ppiformer.utils.bio import class_to_amino_acid
from ppiformer.utils.plotting import plot_confusion_matrix, plot_ddg_scatter
from ppiformer.utils.typing import DDG_INFERENCE_TYPE
from ppiformer.utils.torch import contains_nan_or_inf
from ppiformer.definitions import SE3TRANSFORMER_REQUIRED

if SE3TRANSFORMER_REQUIRED:
    from se3_transformer.model import SE3Transformer
    from se3_transformer.model.fiber import Fiber
    from se3_transformer.runtime.utils import using_tensor_cores


class PPIformer(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module = None,  # TODO EGNN default
        classifier: torch.nn.Module = None,  # TODO MLP default
        optimizer: Optional[torch.optim.Optimizer] = None,  # TODO Handle None
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        n_classes: int = 20,
        # Evaluation
        val_dataset_names: Sequence[str] = ('',),  # TODO Pass list from hydra
        visualize_batch_idx: int = 0,
        visualize_dataloader_idx: int = 0,
        # ddG evaluation
        min_n_muts_for_per_ppi_correlation: int = 10,
        ddg_classification_thresholds: Iterable[float] = (-0.5, 0.0),
        verbose: bool = False,
        pre_encoder_transform_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.save_hyperparameters(logger=False)  # NOTE: Also saves args passed to a child class
        self.val_dataset_names = val_dataset_names
        self.visualize_batch_idx = visualize_batch_idx
        self.visualize_dataloader_idx = visualize_dataloader_idx
        self.min_n_muts_for_per_ppi_correlation = min_n_muts_for_per_ppi_correlation
        self.ddg_classification_thresholds = ddg_classification_thresholds
        self.pre_encoder_transform_kwargs = pre_encoder_transform_kwargs or {}

        # Init validation dataset-specific args to get with `dataloader_idx`
        self.val_metric_suffs = [f'_{name}' for name in self.val_dataset_names]

        # Init pre-encoder transform
        if SE3TRANSFORMER_REQUIRED and isinstance(self.encoder, SE3Transformer):
            self.pre_encoder_transform = PreSE3TransformerTransform(**self.pre_encoder_transform_kwargs)
        elif isinstance(self.encoder, Equiformer):
            self.pre_encoder_transform = PreEquiformerTransform(**self.pre_encoder_transform_kwargs)

        if verbose:
            print(f'{self.__class__}.encoder:', encoder)
            print(f'{self.__class__}.classifier:', classifier)

    def forward(self, data: Data, classifier: bool = False) -> torch.Tensor:
        # Encode
        if isinstance(self.encoder, EGNN):
            h, _ = self.encoder(data.f, data.x, data.edge_index)
        elif SE3TRANSFORMER_REQUIRED and isinstance(self.encoder, SE3Transformer):
            h = self.encoder(*self.pre_encoder_transform(data))
            h = h['0'].squeeze(-1)
        elif isinstance(self.encoder, Equiformer):
            inputs = self.pre_encoder_transform(data)
            out = self.encoder(**inputs)
            assert not contains_nan_or_inf(out.type0), f'{out.type0} contains_nan_or_inf'
            assert not contains_nan_or_inf(out.type1), f'{out.type1} contains_nan_or_inf'
            h = out.type0[inputs['mask']]
        else:
            raise ValueError(f'Corrupted `self.encoder` of type {type(self.encoder)}.')
        
        # Classify
        if classifier:
            y_logits = self.classifier(h)
            return y_logits
        else:
            return h

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        print(optimizer)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss_step',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}

    def get_checkpoint_monitors(self) -> list[tuple[str, str]]:
        monitors = [(f'val_loss_step{suff}', 'min') for suff in self.val_metric_suffs]
        return monitors
    
    def _update_metric(
        self,
        name: str,
        metric_class: type[Metric],
        update_args: Any,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        metric_kwargs: Optional[dict] = None,
        log: bool = True,
        log_n_samples: bool = False
    ) -> None:
        # Log total number of samples for debugging
        if log_n_samples:
            self._update_metric(
                name=name + '_n_samples',
                metric_class=SumMetric,
                update_args=(len(update_args[0]),),
                batch_size=1
            )

        # Init metric if does not exits yet
        if hasattr(self, name):
            metric = getattr(self, name)
        else:
            if metric_kwargs is None:
                metric_kwargs = dict()
            metric = metric_class(**metric_kwargs).to(self.device)
            setattr(self, name, metric)

        # Update
        metric(*update_args)

        # Log
        if log:
            self.log(
                name,
                metric,
                prog_bar=prog_bar,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                metric_attribute=name  # Suggested by a torchmetrics error
            )
    
    def evaluate_classification_step(
        self,
        y_proba: torch.Tensor,
        y_true: torch.Tensor,
        metric_pref: str = '',
        metric_suff: str = '',
        n_classes: Optional[int] = None
    ) -> None:
        # Process args
        if n_classes is None:
            n_classes = self.hparams.n_classes

        # Calculate argmax class predictions
        y_pred = torch.argmax(y_proba, dim=-1)

        # Update and log accuracy
        self._update_metric(
            f'{metric_pref}acc{metric_suff}',
            Accuracy,
            (y_proba, y_true),
            y_proba.size(0),
            metric_kwargs=dict(task='multiclass', num_classes=n_classes),
            log_n_samples=True
        )
        
        # Update confusion matrix
        self._update_metric(
            f'{metric_pref}confmat{metric_suff}',
            ConfusionMatrix,
            (y_pred, y_true),
            metric_kwargs=dict(task='multiclass', num_classes=n_classes),
            log=False
        )
        
        # Update and log accuracy per class
        if self.log_accuracy_per_class:
            for i in range(n_classes):
                aa = class_to_amino_acid(i)
                y_class_mask = y_true == i
                if y_class_mask.sum() != 0:
                    self._update_metric(
                        f'{metric_pref}acc_{aa}{metric_suff}',
                        Accuracy,
                        (y_proba[y_class_mask, :], y_true[y_class_mask]),
                        y_class_mask.sum(),
                        metric_kwargs=dict(task='multiclass', num_classes=n_classes)
                    )

    def evaluate_classification_epoch_end(
        self,
        metric_pref: str = '',
        metric_suff: str = '',
        n_classes: Optional[int] = None
    ) -> None:
        # Process args
        if n_classes is None:
            n_classes = self.hparams.n_classes

        # Log confusion matrix
        name = f'{metric_pref}confmat{metric_suff}'
        # TODO Probably the if is redundant
        if not hasattr(self, name):
            setattr(self, name, ConfusionMatrix(task='multiclass', num_classes=n_classes))
        confmat = getattr(self, name)

        fig = plot_confusion_matrix(confmat)
        if self.global_rank == 0:
            self.trainer.logger.experiment.log({f'{metric_pref}confmat{metric_suff}': fig})

        fig = plot_confusion_matrix(confmat, log_scale=True)
        if self.global_rank == 0:
            self.trainer.logger.experiment.log({f'{metric_pref}confmat_log{metric_suff}': fig})

        confmat.reset()

    @staticmethod
    def calc_wt_proba(
        y_proba: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        wt_proba = torch.gather(y_proba, 1, y_true.unsqueeze(-1)).flatten()
        return wt_proba

    @staticmethod
    def calc_log_odds(
        y_proba: torch.Tensor,
        y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wt_proba = PPIformer.calc_wt_proba(y_proba, y_true)
        log_odds = wt_proba.log().unsqueeze(dim=-1) - y_proba.log()
        return log_odds

    def predict_log_odds(
        self,
        data: Data,
    ) -> torch.Tensor:
        y_logits = PPIformer.forward(self, data, classifier=True)  # Always use base forward
        y_proba = y_logits.softmax(dim=-1)
        log_odds = self.calc_log_odds(y_proba, data.y)
        return log_odds

    def predict_ddg(
        self,
        data: Data,
        kind: DDG_INFERENCE_TYPE = 'wt_marginals',
        return_attn: bool = False
    ) -> torch.Tensor:
        if not hasattr(data, 'mut_pos') or not hasattr(data, 'mut_sub'):
            raise ValueError('`data` does not contain mutation information.')
        
        if kind not in typing.get_args(DDG_INFERENCE_TYPE):
            raise ValueError(f'Wrong `kind` value {kind}.')
        
        if kind == 'embedding_difference' and not hasattr(self, 'head'):
            raise ValueError(f'`self.head` is not defined for `kind` {kind}.')
        
        if return_attn and kind != 'masked_marginals':
            raise ValueError(f'`return_attn` is not implemented for `kind` {kind}.')

        masker = MaskedModelingTransform()
        f = data.f  # Save original node features

        # Iterate over all mutations in data
        # TODO Optimize by making parallel inference on batches containing one mutation from
        # each sample at a time
        # TODO Optimize by reusing predictions for same p but different s
        ddg_pred = []
        attns = []
        pos, sub, wt = DDGLabelPretransform.uncollate(data)
        for p, s, w in zip(pos, sub, wt):
            if kind in ['masked_marginals', 'wt_marginals']:
                if kind == 'masked_marginals':
                    data = masker(data, masked_nodes=p)
                else:  # kind == 'wt_marginals':
                    feats_wt = torch.nn.functional.one_hot(w, num_classes=masker.vocab_size).float()
                    data = masker(data, masked_nodes=p, masked_features=feats_wt)
                data.f = data.f_masked
                log_odds = self.predict_log_odds(data)
                data.f = f
                ddg_pred.append(log_odds[p, s].sum())
                if return_attn:
                    attns.append(self.encoder.last_forward_attn)

            elif kind in ['embedding_difference', 'embedding_concatenation']:  # baselines
                feats_wt = torch.nn.functional.one_hot(w, num_classes=masker.vocab_size).float()
                data = masker(data, masked_nodes=p, masked_features=feats_wt)
                data.f = data.f_masked
                z_wt = PPIformer.forward(self, data)[~data.node_mask]

                feats_sub = torch.nn.functional.one_hot(s, num_classes=masker.vocab_size).float()
                data = masker(data, masked_nodes=p, masked_features=feats_sub)
                data.f = data.f_masked
                z_sub = PPIformer.forward(self, data)[~data.node_mask]

                data.f = f
                if kind == 'embedding_difference':
                    ddg_pred_mut = self.head(torch.sum(z_wt - z_sub, dim=0)).squeeze()
                else:  # kind == 'embedding_concatenation':
                    ddg_pred_mut =  self.head(torch.hstack([z_wt.mean(dim=0), z_sub.mean(dim=0)])).squeeze()

                ddg_pred.append(ddg_pred_mut)
            else:
                raise ValueError(f'Wrong `kind` value {kind}.')

        ddg_pred = torch.stack(ddg_pred)
        if return_attn:
            attns = torch.stack(attns)
            return ddg_pred, attns
        else:
            return ddg_pred

    # NOTE: The implementation below can handle ddG augmentations (mtuations wild types not corresponding to wild-type strucutre).
    # Nevertheless, it currently does not reproduce the results of the original implementation above. Should be debugged
    # def predict_ddg(
    #     self,
    #     data: Data,
    #     kind: DDG_INFERENCE_TYPE = 'wt_marginals'
    # ) -> torch.Tensor:
    #     if not hasattr(data, 'mut_pos') or not hasattr(data, 'mut_sub'):
    #         raise ValueError('`data` does not contain mutation information.')
        
    #     if kind not in typing.get_args(DDG_INFERENCE_TYPE):
    #         raise ValueError(f'Wrong `kind` value {kind}.')
        
    #     if kind == 'embedding_difference' and not hasattr(self, 'head'):
    #         raise ValueError(f'`self.head` is not defined for `kind` {kind}.')

    #     masker = MaskedModelingTransform()
    #     f = data.f  # Save original node features
    #     y = data.y  # Save original wild types

    #     # Iterate over all mutations in data
    #     # TODO Optimize by making parallel inference on batches containing one mutation from
    #     # each sample at a time
    #     # TODO Optimize by reusing predictions for same p but different s
    #     ddg_pred = []
    #     pos, sub, wt = DDGLabelPretransform.uncollate(data)
    #     for i, (p, s, w) in enumerate(zip(pos, sub, wt)):
    #         if kind in ['masked_marginals', 'wt_marginals']:
    #             if kind == 'masked_marginals':
    #                 data = masker(data, masked_nodes=p)
    #             else:  # kind == 'wt_marginals':
    #                 feats_wt = torch.nn.functional.one_hot(w, num_classes=masker.vocab_size).float()
    #                 data = masker(data, masked_nodes=p, masked_features=feats_wt)

    #             # Replace wild-type features and wild-type classes and make inference
    #             data.f = data.f_masked
    #             y_current = data.y.clone()
    #             y_current[p] = w
    #             data.y = y_current
    #             log_odds = self.predict_log_odds(data)
    #             data.f = f
    #             data.y = y

    #             # NOTE Sum rather than mean seams to be crucial compared to ESM-IF
    #             ddg_pred.append(log_odds[p, s].sum())

    #         elif kind in ['embedding_difference', 'embedding_concatenation']:
    #             feats_wt = torch.nn.functional.one_hot(w, num_classes=masker.vocab_size).float()
    #             data = masker(data, masked_nodes=p, masked_features=feats_wt)
    #             data.f = data.f_masked
    #             z_wt = PPIformer.forward(self, data)[~data.node_mask]

    #             feats_sub = torch.nn.functional.one_hot(s, num_classes=masker.vocab_size).float()
    #             data = masker(data, masked_nodes=p, masked_features=feats_sub)
    #             data.f = data.f_masked
    #             z_sub = PPIformer.forward(self, data)[~data.node_mask]

    #             data.f = f
    #             if kind == 'embedding_difference':
    #                 ddg_pred_mut = self.head(torch.sum(z_wt - z_sub, dim=0)).squeeze()
    #                 # ddg_pred_mut = (self.head(torch.mean(z_wt - z_sub, dim=0)).squeeze() - self.head(torch.mean(z_sub - z_wt, dim=0)).squeeze()) / 2
    #             else:  # kind == 'embedding_concatenation':
    #                 ddg_pred_mut =  self.head(torch.hstack([z_wt.mean(dim=0), z_sub.mean(dim=0)])).squeeze()

    #             ddg_pred.append(ddg_pred_mut)
    #         else:
    #             raise ValueError(f'Wrong `kind` value {kind}.')

    #     ddg_pred = torch.stack(ddg_pred)
    #     return ddg_pred
        
    def evaluate_ddg_step(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        batch: torch.Tensor,
        metric_pref: str = '',
        metric_suff: str = '',
        visualize_example: bool = False
    ) -> None:
        # Update and log Pearson correlation
        self._update_metric(
            f'{metric_pref}pearson{metric_suff}',
            PearsonCorrCoef,
            (ddg_pred, ddg_true),
            ddg_pred.size(0),
            log_n_samples=True
        )

        # Update and log Spearman correlation
        self._update_metric(
            f'{metric_pref}spearman{metric_suff}',
            SpearmanCorrCoef,
            (ddg_pred, ddg_true),
            ddg_pred.size(0)
        )

        # Update and log Pearson correlation per sample (PPI complex)
        masks = [batch == b for b in range(batch[-1] + 1)]
        masks = list(filter(lambda x: x.sum() >= self.min_n_muts_for_per_ppi_correlation, masks))
        corrs = [pearson_corrcoef(ddg_pred[m], ddg_true[m]) for m in masks]
        self._update_metric(
            f'{metric_pref}pearson_per_ppi{metric_suff}',
            MeanMetric,
            (corrs,),
            len(corrs)
        )

        # Update and log Spearman correlation per sample (PPI complex)
        corrs = [spearman_corrcoef(ddg_pred[m], ddg_true[m]) for m in masks]
        self._update_metric(
            f'{metric_pref}spearman_per_ppi{metric_suff}',
            MeanMetric,
            (corrs,),
            len(corrs)
        )

        # Update and log information retrieval metrics
        for thr in self.ddg_classification_thresholds:
            ddg_pred_class = (ddg_pred < thr).int()  # < because negative ddG is stabilizing
            ddg_true_class = (ddg_true < thr).int()

            self._update_metric(
                f'{metric_pref}precision_thr={thr:.1f}{metric_suff}',
                BinaryPrecision,
                (ddg_pred_class, ddg_true_class),
                ddg_pred_class.size(0)
            )

            self._update_metric(
                f'{metric_pref}recall_thr={thr:.1f}{metric_suff}',
                BinaryRecall,
                (ddg_pred_class, ddg_true_class),
                ddg_pred_class.size(0)
            )

        self._update_metric(
            f'{metric_pref}auroc{metric_suff}',
            BinaryAUROC,
            (-ddg_pred, ddg_true_class),
            ddg_pred.size(0),
        )

        self._update_metric(
            f'{metric_pref}auroc_destabilizing{metric_suff}',
            BinaryAUROC,
            (ddg_pred, (ddg_true > thr).int()),
            ddg_pred.size(0),
        )

        # Visualize scatter for batch example
        if visualize_example:
            if not isinstance(self.logger, pl.loggers.WandbLogger):
                raise ValueError('`visualize_example` requires `self.logger` to be a `WandbLogger`.')
            fig = plot_ddg_scatter(ddg_pred, ddg_true, batch)
            self.trainer.logger.experiment.log(
                {f'{metric_pref}{self.visualize_batch_idx}_batch{metric_suff}': fig}
            )
