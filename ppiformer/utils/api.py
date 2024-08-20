import warnings
import requests
import zipfile
import os
from tqdm import tqdm
from typing import Sequence, Union
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from equiformer_pytorch.equiformer_pytorch import MLPAttention, L2DistAttention

from mutils.mutations import Mutation
from ppiformer.tasks.node import DDGPPIformer
from ppiformer.definitions import PPIFORMER_ROOT_DIR


def download_from_zenodo(
    file: str,
    project_url: str = 'https://zenodo.org/records/12789167/files/',
    destination_folder: Union[Path, str] = None
) -> None:
    """
    Download a file from Zenodo and extract its contents.

    Args:
        file (str): Name of the file to download and unpack. For example, ``'weights.zip'`` to
            download all the folder with the weights.
        project_url (str, optional): URL of the Zenodo project.
        destination_folder (Union[Path, str], optional): Path to the destination folder. If None, 
            the folder will be created in the ``ppiref.definitions.PPIFORMER_ROOT_DIR`` directory.
    """
    # Create full file url
    url = project_url + file
    stem = Path(url).stem

    if destination_folder is None:
        destination_folder = PPIFORMER_ROOT_DIR / Path(file).stem

    # Check if the folder already exists
    if (destination_folder).is_dir():
        warnings.warn(f'{destination_folder} already exists. Skipping download.')
        return
    
    # Create the folder
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Download the file with progress bar
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, desc=f'Downloading to {destination_folder}', unit='iB', unit_scale=True)
    file_path = os.path.join(destination_folder, f'{stem}.zip')
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    # Check if the download was successful
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise RuntimeError("Download failed. Please try to download the file manually.")

    # Extract the contents of the zip file with progress bar
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # List of archive contents
        list_of_files = zip_ref.infolist()
        total_files = len(list_of_files)
        # Progress bar for extraction
        with tqdm(total=total_files, desc="Extracting", unit='files') as pbar:
            for file in list_of_files:
                zip_ref.extract(member=file, path=destination_folder)
                pbar.update(1)

    # Delete the zip file after extraction
    os.remove(file_path)


def predict_ddg(
    models: Sequence[DDGPPIformer],
    ppi: Union[Path, str],
    muts: Sequence[str],
    return_attn: bool = False,
    impute: bool = False,
    impute_val: float = 0.691834179286864  # average from the SKEMPI v2.0 training set
) -> torch.Tensor:
    if return_attn and impute:
        raise NotImplementedError('TODO Implement imputation for attention coefficients with zero tensors.')

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
        skip_data_on_processing_errors=False, dataset_max_workers=1, fresh=True
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

    # Impute values for the mutations out of the interaction interface
    if impute:
        to_impute = ~torch.tensor([Mutation.from_str(m).wt_in_pdb(ppi) for m in muts])
        ddg_pred_imputed = torch.full((len(to_impute),), impute_val)
        ddg_pred_imputed[~to_impute] = ddg_pred
        ddg_pred = ddg_pred_imputed

    # Return attention as a tensor [model, muts, batch, layer, degree, head, node, node]
    if return_attn:
        return ddg_pred, attns
    return ddg_pred
    