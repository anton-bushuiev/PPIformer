from typing import Sequence, Union
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from equiformer_pytorch.equiformer_pytorch import MLPAttention, L2DistAttention

from ppiformer.tasks.node import DDGPPIformer


def predict_ddg(
    models: Sequence[DDGPPIformer],
    ppi: Union[Path, str],
    muts: Sequence[str],
    return_attn: bool = False
) -> torch.Tensor:
    ppi = Path(ppi)
    assert ppi.is_file()

    # Read validation dataloader config used at model training
    cfg = models[0].hparams['run_cfg']
    cfg = OmegaConf.load(StringIO(cfg))
    cfg_dataloader = dict(cfg.val_dataloader)

    # Replace PPI
    cfg_dataloader['dataset'] = [str(ppi)]

    # Replace mutations (The pretransforms are instantiated separately beacuse OmegaConf does not support passing a DataFrame inside)
    df = pd.DataFrame([{'#Pdb': ppi.stem, 'Mutation(s)': m, 'ddG': np.nan} for m in muts])
    pretransform = []
    for t in cfg_dataloader['pretransform']:
        if t['_target_'] == 'ppiformer.data.transforms.DDGLabelPretransform':
            if 'df_path' in t:  # compatibility with older versions
                del t.df_path
            t = hydra.utils.instantiate(t, df=df, mut_col='Mutation(s)')
        else:
            t = hydra.utils.instantiate(t)
        pretransform.append(t)

    # Instantiate datalaoder from config
    dataloader = hydra.utils.instantiate(
        cfg_dataloader, pretransform=pretransform,
        skip_data_on_processing_errors=False, dataset_max_workers=1
    )

    # Compatibility with older Equiformer versions
    for model in models:
        model.encoder.store_last_forward_attn = True
        for layer in model.encoder.layers.blocks:
            layer = layer[0]  # 0: attention, 1: feedforward
            assert isinstance(layer, L2DistAttention) or isinstance(layer, MLPAttention)
            if not hasattr(layer, 'attn_head_gates'):
                layer.attn_head_gates = None
            if not hasattr(layer, 'last_forward_attn'):
                layer.store_last_forward_attn = True

    # Predict average ddG from ensembled models and get attention coefficients
    with torch.inference_mode():
        for batch in dataloader:  # always a single batchm since only one PPI
            ddg_preds = []
            attns = []
            for model in models:
                ddg_pred, attn = model(batch, return_attn=True)
                ddg_preds.append(ddg_pred)
                attns.append(attn)
            ddg_pred = torch.stack(ddg_preds).mean(dim=0)
            attns = torch.stack(attns)

    # Return attention as a tensor [model, muts, batch, layer, degree, head, node, node]
    if return_attn:
        return ddg_pred, attns
    return ddg_pred
    