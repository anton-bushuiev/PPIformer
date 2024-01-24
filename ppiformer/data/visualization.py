import copy
from pathlib import Path
from typing import Union 

import torch
import plotly
import pandas as pd
from torch_geometric.data import Data, Batch
from graphein.ml.visualisation import plot_pyg_data

from ppiref.utils.ppipath import path_to_ppi_id


def plot_data(data: Union[Data, Batch], edges: bool = False) -> plotly.graph_objs._figure.Figure:
    # Adapt to graphein plotting
    data = copy.deepcopy(data)
    if hasattr(data, 'path'):
        if isinstance(data.path, list):
            data.name = ', '.join([Path(p).stem for p in data.path])
        else:
            data.name = Path(data.path).stem
    else:
        data.name = ''
    data.coords = data.x
    data.dist_mat = []

    # Set edges
    if not edges:
        data.edge_index = torch.tensor([[], []])

    # Color by chain
    if isinstance(data.node_id[0], list):
        data.node_id = sum(data.node_id, start=[])
    chain_id = list(map(lambda x: x.split(':')[0], data.node_id))
    if isinstance(data, Batch):  # make unique for each batch sample
        chain_id = [c + str(data.batch[i]) for i, c in enumerate(chain_id)]
    chain_id = torch.tensor(pd.Series(chain_id).astype('category').cat.codes)
    node_colour_tensor = chain_id

    # Color by mask if present
    if hasattr(data, 'node_mask'):
        node_colour_tensor[~data.node_mask] = node_colour_tensor.max() + 100

    # Add PDB ids to node ids if batch
    if isinstance(data, Batch):
        for i in range(data.num_nodes):
            data.node_id[i] = f'{path_to_ppi_id(data.path[data.batch[i]])}:{data.node_id[i]}'

    # Plot with plotly via graphein
    fig = plot_pyg_data(
        data,
        plot_title=data.name,
        node_size_multiplier=0.5,
        node_colour_tensor=node_colour_tensor
    )
    return fig
