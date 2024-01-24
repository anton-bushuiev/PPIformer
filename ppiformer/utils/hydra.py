from omegaconf import DictConfig, OmegaConf


def prepare_cfg(cfg: DictConfig) -> None:
    # https://github.com/facebookresearch/hydra/issues/1939
    OmegaConf.resolve(cfg)
    del cfg.train_dataloader._pretransform
    del cfg.train_dataloader._prefilter
    del cfg.train_dataloader._transform
    del cfg.val_dataloader._pretransform
    del cfg.val_dataloader._prefilter
    del cfg.val_dataloader._transform
    