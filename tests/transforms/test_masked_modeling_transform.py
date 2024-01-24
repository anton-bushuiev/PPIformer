import pytest
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from biopandas.pdb import PandasPdb

from mutils.pdb import get_sequences
from ppiformer.data.transforms import PDBToPyGPretransform, MaskedModelingTransform
from ppiformer.utils.bio import amino_acid_to_class
from ppiformer.definitions import PPIFORMER_TEST_DATA_DIR


@pytest.mark.parametrize('mask_ratio', [0.15, 0.2])
def test_masked_modeling_transform_mask_ratio(mask_ratio):
    path = PPIFORMER_TEST_DATA_DIR / '1bui_A_C.pdb'
    transform = Compose([
        PDBToPyGPretransform(),
        MaskedModelingTransform(mask_ratio=mask_ratio, same_chain=False)
    ]) 
    data = transform(path)
    
    assert torch.sum(~data.node_mask) == round(mask_ratio * data.num_nodes)

    node_mask = torch.any(data.f_masked != 0, dim=1)
    assert node_mask.sum() == round((1 - mask_ratio) * data.num_nodes)


@pytest.mark.parametrize('mask_sum', [1, 5])
def test_masked_modeling_transform_mask_sum(mask_sum):
    path = PPIFORMER_TEST_DATA_DIR / '1bui_A_C.pdb'
    transform = Compose([
        PDBToPyGPretransform(),
        MaskedModelingTransform(mask_sum=mask_sum)
    ])
    data = transform(path)
    
    assert torch.sum(~data.node_mask) == mask_sum

    node_mask = torch.any(data.f_masked != 0, dim=1)
    assert node_mask.sum() == data.num_nodes - mask_sum


def test_masked_modeling_transform_mask_deterministic():
    path = PPIFORMER_TEST_DATA_DIR / '1bui_A_C.pdb'
    transform = Compose([
        PDBToPyGPretransform(),
        MaskedModelingTransform(mask_ratio=0.15, deterministic=True)
    ]) 
    data = transform(path)
    node_mask_0 = data.node_mask

    data = transform(path)
    node_mask_1 = data.node_mask

    assert torch.equal(node_mask_0, node_mask_1)


def test_masked_modeling_transform_same_chain():
    path = PPIFORMER_TEST_DATA_DIR / '4GXU_A_B_C_D_E_F_M_N.pdb'
    transform = Compose([
        PDBToPyGPretransform(),
        MaskedModelingTransform(mask_ratio=0.15, same_chain=True)
    ]) 
    data = transform(path)

    masked_nodes = np.array(data.node_id)[~data.node_mask]
    masked_chains = list(set(list(map(lambda x: x.split(':', 1)[0], masked_nodes))))
    assert len(masked_chains) == 1

    masked_chain = masked_chains[0]
    masked_chain_length = len(get_sequences(path)[masked_chain])
    assert round(0.15 * masked_chain_length) == torch.sum(~data.node_mask)
