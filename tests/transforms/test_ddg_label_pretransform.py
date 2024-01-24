import pytest
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from biopandas.pdb import PandasPdb

from mutils.pdb import get_sequences
from ppiformer.data.transforms import PDBToPyGPretransform, DDGLabelPretransform
from ppiformer.utils.bio import amino_acid_to_class
from ppiformer.definitions import PPIFORMER_TEST_DATA_DIR


def dG(aff, t):
        return (8.314/4184) * t * np.log(aff)

def ddG(aff_wt, aff_mut, t=273.15 + 25.0):
    return dG(aff_mut, t) - dG(aff_wt, t)


def test_ddg_label_pretransform():
    # https://life.bsc.es/pid/skempi2/database/browse/mutations?keywords=mutations.pdb+contains+%224gxu%22+or+mutations.protein_1+contains+%224gxu%22+or+mutations.protein_2+contains+%224gxu%22&protein_search=4gxu
    path = PPIFORMER_TEST_DATA_DIR / '4GXU_A_B_C_D_E_F_M_N.pdb'
    transform = Compose([
        PDBToPyGPretransform(),
        DDGLabelPretransform(max_n_substs=2)
    ]) 
    data = transform(path)

    assert torch.equal(
        data.mut_ddg,
        torch.tensor([
            ddG(6.2e-9, 2.4e-7),
            ddG(6.2e-9, 1.7e-7),
            ddG(6.2e-9, 1e-6)
        ]).float()
    )
    assert torch.equal(
        data.mut_sub,
        torch.tensor([
            [amino_acid_to_class('E'), DDGLabelPretransform.pad_val],
            [amino_acid_to_class('G'), DDGLabelPretransform.pad_val],
            [amino_acid_to_class('E'),  amino_acid_to_class('G')]
        ])
    )
    assert data.mut_pos.shape == data.mut_sub.shape
    # NOTE: The following test may not hold for other complexes as node integer pos are scattered
    # across chains
    assert data.node_id[data.mut_pos[0][0]].split(':', 1)[-1] == 'ASP:189'
    assert data.node_id[data.mut_pos[1][0]].split(':', 1)[-1] == 'ASP:224'
    assert data.node_id[data.mut_pos[2][0]].split(':', 1)[-1] == 'ASP:189'
    assert data.node_id[data.mut_pos[2][1]].split(':', 1)[-1] == 'ASP:224'


def test_ddg_label_pretransform_no_labels():
    path = PPIFORMER_TEST_DATA_DIR / '1bui_A_C.pdb'
    transform = Compose([
        PDBToPyGPretransform(),
        DDGLabelPretransform(max_n_substs=2)
    ]) 
    data = transform(path)
    
    assert data.n_muts == 0


def test_ddg_label_pretransform_uncollate_synthetic():
    data = Batch(
        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1]),  # 5 nodes in first graph, 2 in second
        ptr=torch.tensor([0, 5, 7]),
        mut_ddg_batch=torch.tensor([0, 0, 1]),  # 2 mutation in first graph, 1 in second
        mut_pos=torch.tensor([
            [3, -1],
            [4, -1],
            [0, 1]
        ]),
        mut_sub=torch.tensor([
            [0, -1],
            [0, -1],
            [18, 19]
        ]),
        mut_wt=torch.tensor([
            [0, -1],
            [0, -1],
            [0, 0]
        ])
    )
    mut_pos_real, mut_sub_real, mut_wt_real = DDGLabelPretransform.uncollate(data)

    mut_pos_expected = [
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([0 + 5, 1 + 5]),
    ]
    mut_sub_expected = [
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([18, 19])
    ]
    mut_wt_expected = [
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0, 0])
    ]  

    for i in range(len(mut_pos_expected)):
        assert torch.equal(mut_pos_real[i], mut_pos_expected[i])
        assert torch.equal(mut_sub_real[i], mut_sub_expected[i])
        assert torch.equal(mut_wt_real[i], mut_wt_expected[i])


def test_ddg_label_pretransform_uncollate_real():
    # Test uncollate increments nodes correctly on real data.
    path = PPIFORMER_TEST_DATA_DIR / '4GXU_A_B_C_D_E_F_M_N.pdb'
    batch_paths = 2 * [path]
    transform = Compose([
        PDBToPyGPretransform(),
        DDGLabelPretransform(max_n_substs=2)
    ]) 
    data = Batch.from_data_list(
        [transform(path) for path in batch_paths],
        follow_batch=DDGLabelPretransform.follow_batch_attrs()
    )
    mut_pos, _, _ = DDGLabelPretransform.uncollate(data)

    n_amino_acids = sum(map(len, get_sequences(path).values()))

    n_muts = data.n_muts[0]
    node_id = np.array(sum(data.node_id, start=[]) if isinstance(data.node_id, list) else data.node_id)
    for i in range(n_muts):
        assert torch.equal(mut_pos[i], mut_pos[i + n_muts] - n_amino_acids)
        assert np.array_equal(node_id[mut_pos[i]], node_id[mut_pos[i + n_muts]])
