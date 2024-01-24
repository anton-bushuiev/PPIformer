import warnings
import graphein
from pathlib import Path

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import torch
import torch_geometric
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import equiformer_pytorch

from ppiformer.utils.torch import get_n_params
from ppiformer.utils.hydra import prepare_cfg

graphein.verbose(enabled=False)
warnings.simplefilter(action='ignore', category=FutureWarning)
# NOTE: You are using a CUDA device ('AMD Instinct MI250X') that has Tensor Cores. 
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.
# NOTE: the command has no effect if called from main for some reason
torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='../configs', config_name='run.yaml')
def main(cfg : DictConfig) -> None:
    # Generate job key (run id) if not present
    with open_dict(cfg):
        if not cfg.job_key:
            cfg.job_key = wandb.util.generate_id()
        if not cfg.run_name:
            cfg.run_name = cfg.job_key

    # Init logger
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = pl.loggers.WandbLogger(
        entity=cfg.team_name,
        project=cfg.project_name,
        name=cfg.run_name,
        config=config,
        tags=cfg.tags.split(',') if isinstance(cfg.tags, str) else cfg.tags,
        log_model=False,
        id=cfg.job_key,
        resume='allow'
    )

    # Prepare Hydra config
    prepare_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Pass full config to model to store in Lightning checkpoint (yaml to avoid instantiation)
    model_kwargs = dict()
    model_kwargs['run_cfg'] = OmegaConf.to_yaml(cfg)

    # Set seeds
    torch_geometric.seed_everything(cfg.seed)
    pl.seed_everything(cfg.seed)

    # Load data
    # TODO Extend to load multiple val datasets
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader)
    val_dataloaders = [hydra.utils.instantiate(cfg.val_dataloader)]

    # Prepare model arguments
    model_kwargs['val_dataset_names'] = [cfg.val_dataloader.dataset if not cfg.cv else 'cv']

    # Process tuples for Equiformer separately (hydra/omegaconf is broken for tuples)
    if cfg.model.checkpoint_path is None:
        if cfg.model.encoder._target_ == 'equiformer_pytorch.Equiformer':
            encoder_kwargs = dict()
            for k, v in cfg.model.encoder.items():
                if k == '_target_':
                    continue
                elif k in ['dim', 'dim_in', 'dim_head']:
                    encoder_kwargs[k] = tuple(v)
                else:
                    encoder_kwargs[k] = v
            encoder = equiformer_pytorch.Equiformer(**encoder_kwargs)
            del cfg.model.encoder
            model_kwargs['encoder'] = encoder

    # Init model either from checkpoint or from config. If loading from checkpoint, config params
    # for the model are ignored with the exception of optimizer. If the model is loaded from 
    # checkpoint due to resuming, completely everything from current config is ignored including
    # optimizer and data, etc. (see the beginning of the script).
    # UPDATE: Added model.test_csv_path and model.pre_encoder_transform_kwargs to be passed to
    # UPDATE: Added model.kind
    # UPDATE: Added model.correction
    # TODO Probably implement some method for subclasses to define what to pass and what to preserve
    if cfg.model.checkpoint_path is not None:
        if not cfg.resume_job_key:
            model_kwargs['optimizer'] = hydra.utils.instantiate(cfg.model.optimizer)
            model_kwargs['test_csv_path'] = cfg.model.test_csv_path
            model_kwargs['correction'] = cfg.model.correction
            model_kwargs['pre_encoder_transform_kwargs'] = cfg.model.pre_encoder_transform_kwargs
            if 'kind' in cfg.model:
                model_kwargs['kind'] = cfg.model.kind
        model_class = hydra._internal.utils._locate(cfg.model._target_)
        model = model_class.load_from_checkpoint(cfg.model.checkpoint_path, strict=False, **model_kwargs)
    else:
        model = hydra.utils.instantiate(cfg.model, **model_kwargs)

    # Init metric checkpoints and other callbacks. By default, the checpointing is done at the end 
    # of each training epoch
    callbacks = []
    for i, (monitor, mode) in enumerate(model.get_checkpoint_monitors()):
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            save_top_k=1,
            mode=mode,
            dirpath=Path(cfg.project_name) / cfg.run_name,
            filename=f'{{step:06d}}-{{{monitor}:03.03f}}',
            auto_insert_metric_name=True,
            save_last=(i == 0)
        )
        callbacks.append(checkpoint)
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Init trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Log state to wandb
    if trainer.global_rank == 0:
        for monitor, mode in model.get_checkpoint_monitors():
            wandb.define_metric(monitor, summary=mode)

    # Watch model on main process
    if trainer.global_rank == 0:
        # logger.watch(model)
        logger.experiment.config.update({'n_params': get_n_params(model)})

    # Just test if specified
    if cfg.test:
        if cfg.trainer.devices != 1:
            raise ValueError('Testing is only supported with single device.')
        if model.test_csv_path is None:
            raise ValueError('For testing, `model.test_csv_path` must be specified.')
        trainer.test(
            model,
            dataloaders=val_dataloaders
        )
        return 0

    # Validate untrained model
    trainer.validate(
        model,
        dataloaders=val_dataloaders
    )

    # Train
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloaders,
    )


if __name__ == '__main__':
    main()
