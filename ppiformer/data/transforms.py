import copy
import random
from typing import Literal, Any, Iterable, Optional, Callable, Union
from pathlib import Path
from functools import partial
from collections import Counter

import torch
import pandas as pd
import torch_geometric.transforms as T
import einops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import index_to_mask, to_dense_batch
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.ml import GraphFormatConvertor
from graphein.protein.edges.distance import add_k_nn_edges
from graphein.protein.features.nodes.geometry import add_virtual_beta_carbon_vector, add_sequence_neighbour_vector, VECTOR_FEATURE_NAMES
from Bio.PDB.Polypeptide import protein_letters_3to1

from mutils.data import load_SKEMPI2
from ppiref.utils.ppipath import path_to_pdb_id, path_to_partners
from ppiformer.model.egnn_clean import get_edges
from ppiformer.utils.bio import amino_acid_to_class
from ppiformer.utils.torch import unpad_sequence, pad_fixed_length, contains_nan_or_inf
from ppiformer.definitions import SE3TRANSFORMER_REQUIRED
import warnings
if SE3TRANSFORMER_REQUIRED:
    import dgl
    from se3_transformer.data_loading.qm9 import _get_relative_pos
    from se3_transformer.model.basis import get_basis
    from se3_transformer.runtime.utils import using_tensor_cores


protein_letters_1to3 = {v: k for k, v in protein_letters_3to1.items()}


class StochasticTransform(T.BaseTransform):
    def __init__(
        self,
        deterministic: bool = False,
        seed: Optional[int] = None
    ):
        """_summary_

        Args:
            deterministic (bool, optional): _description_. Set to True to make deterministic w.r.t.
            to an input sample. Usefult for validating on same data despite shuffling. Defaults to 
            True.
            seed (Optional[int], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.deterministic = deterministic
        self.rng = random.Random(seed)

    def __call__(self, data: Data) -> Data:
        if self.deterministic:
            seed = int(data.x[0][0])
            self._reset_rng(seed)
        return data

    def _reset_rng(self, seed) -> None:
        self.rng = random.Random(seed)


class PDBToPyGPretransform(T.BaseTransform):
    def __init__(
        self,
        k: Optional[int] = 10,
        undirected: bool = True,
        type1_features: Iterable[str] = (
            'virtual_c_beta_vector',
            'sequence_neighbour_vector_n_to_c',
            'sequence_neighbour_vector_c_to_n'
        ),
        divide_coords_by: float = 1.0,
    ):
        self.config = ProteinGraphConfig(
            edge_construction_functions=[] if k is None else [
                partial(add_k_nn_edges, k=k, long_interaction_threshold=0)
            ],
            node_metadata_functions=[amino_acid_one_hot]
        )
        self.convertor = GraphFormatConvertor(
            src_format='nx',
            dst_format='pyg',
            columns=[
                'coords',
                'node_id',
                'amino_acid_one_hot',
                'edge_index',
                *VECTOR_FEATURE_NAMES
            ]
        )

        self.undirected = undirected
        if not undirected:
            raise NotImplementedError()
        self.type1_features = type1_features
        self.divide_coords_by = divide_coords_by

    def __call__(self, path: Path):
        # Construct NetworkX graph
        g = construct_graph(config=self.config, path=str(path), verbose=False)
        if self.type1_features:
            if 'virtual_c_beta_vector' in self.type1_features:
                add_virtual_beta_carbon_vector(g)
            if 'sequence_neighbour_vector_n_to_c' in self.type1_features:
                add_sequence_neighbour_vector(g)
            if 'sequence_neighbour_vector_c_to_n' in self.type1_features:
                add_sequence_neighbour_vector(g, n_to_c=False)

        # Convert NetworkX graph to PyG data
        data = self.convertor(g)

        # TODO Compression
        # Remove half of the edges if directed is specified and the graph is undirected (for 
        # memory-efficiency) 
        # if not self.undirected and is_undirected(data):

        # Construct node features
        data.f = data.amino_acid_one_hot.float()

        # Rename and retype attributes
        data.x = data.coords.float()
        data.x /= self.divide_coords_by

        # Create amino acid classification labels
        data.y = torch.argmax(data.amino_acid_one_hot, dim=1)

        # Remove unnecessary fields for efficiency
        del data.coords
        del data.amino_acid_one_hot

        # Remember path
        data.path = path

        # Validate data
        for attr in ['x', 'f', 'y', 'virtual_c_beta_vector']:
            if hasattr(data, attr):
                if contains_nan_or_inf(getattr(data, attr)):
                    raise ValueError(f'`data.{attr}` from {path} contains NaN of Inf.')

        return data

    @staticmethod
    def validate_node(node: str):
        # Check amino acid identifier is correct (e.g. not UNK) in 
        # graphein amino acid in id in
        aa = node.split(':')[1]
        try:
            protein_letters_3to1[aa]
            return True
        except:
            return False


class MaskedModelingTransform(StochasticTransform):
    """Masks random nodes

    Adds:
        - `node_mask` [*]: boolean tensor with True for nodes that are not masked
        - `f_masked` [*]: type-0 features zero-masked at `~node_mask`
    """
    def __init__(
        self,
        mask_ratio: Optional[float] = None,  # At 0.15 corresponds to BERT, ESM-1, ESM-2
        mask_sum: Optional[int] = None,
        bert: bool = False,  # https://github.com/google-research/bert/blob/master/create_pretraining_data.py
        same_chain: bool = True,  # All masked nodes are from the same chain
        vocab_size: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mask_ratio is None and mask_sum is None:
            mask_ratio = 0.15
        if mask_ratio is not None and mask_sum is not None:
            raise ValueError('Overspecified masking.')

        self.mask_ratio = mask_ratio
        self.mask_sum = mask_sum
        self.bert = bert
        self.same_chain = same_chain
        self.vocab_size = vocab_size

    def __call__(
        self,
        data: Data,
        masked_nodes: Optional[torch.Tensor] = None,
        masked_features: Optional[torch.Tensor] = None
    ) -> Data:
        super().__call__(data)

        # Sample nodes
        if masked_nodes is None:

            # Define nodes to sample from
            if self.same_chain and self.mask_sum != 1:
                node_id_chains = list(map(lambda x: x.split(':', 1)[0], data.node_id))
                chain = self.rng.sample(list(set(node_id_chains)), 1)[0]
                population = [i for i, c in enumerate(node_id_chains) if c == chain]
            else:
                population = list(range(data.num_nodes))

            # Define number of nodes to sample
            if self.mask_ratio is not None:
                k=round(self.mask_ratio * len(population))
                if k == 0:
                    k = 1
            elif self.mask_sum is not None:
                k=self.mask_sum

            # Sample nodes
            masked_nodes = self.rng.sample(population=population, k=k)
            
            # Convert to torch
            masked_nodes = torch.tensor(masked_nodes).to(data.f.device)
        
        # Construct node mask
        node_mask = ~index_to_mask(masked_nodes, size=data.num_nodes)
        
        # Mask node features
        if masked_features is None:
            masked_features = torch.zeros((torch.sum(~node_mask), self.vocab_size))
        masked_features = masked_features.to(data.f.device)
        f_masked = data.f.clone()
        f_masked[~node_mask, :] = masked_features
        if self.bert:
            for node in masked_nodes:
                if self.rng.random() < 0.2:
                    if self.rng.random() < 0.5:
                        f_masked[node] = torch.nn.functional.one_hot(
                            torch.tensor(self.rng.randint(0, self.vocab_size - 1)),
                            self.vocab_size
                        ).float().to(f_masked.device)
                    else:
                        f_masked[node] = data.f[node].clone()

        data.node_mask, data.f_masked = node_mask, f_masked
        return data


# TODO Test for insertion codes handling
# (e.g. 3BDY_HL_V DL27aN,PL27cA,RL27dK,SL28T,YL53F,TL93S,TL94S) in raw 'Mutation(s)_PDB'
class DDGLabelPretransform(T.BaseTransform):
    """Adds ddG annotations from SKEMPI2

    Adds:
        - `mut_ddg` [*]: ddG annotations from SKEMPI2 correpsonding to all multi-point mutations
            on the `data`
        - `mut_pos` [max_n_substs, *]: padded positions (integer node ids) of substitutions of
            multi-point mutations
        - `mut_sub` [max_n_substs, *]: padded classes (0-19) of mutated amino acids of
            multi-point mutations
    """

    pad_val: Any = -1

    def __init__(
        self,
        df: Optional[Union[str, Path, pd.DataFrame]] = None,
        mut_col: Union[Literal['Mutation(s)_PDB', 'Mutation(s)_cleaned'], str] = 'Mutation(s)_cleaned',
        max_n_substs: int = 26,  # 26 is max number of substitions per mutation in SKEMPI2
        strict_wt: bool = True  # True to make sure wild types in a mutation match the structure
    ):
        super().__init__()
        self.mut_col = mut_col
        self.max_n_substs = max_n_substs
        self.strict_wt = strict_wt

        # Load dataframe with mutations in SKEMPI2 format
        if df is not None:
            if isinstance(df, pd.DataFrame):
                df_skempi = df
            else:  # Path or str
                df_skempi = pd.read_csv(df)
        else:
            df_skempi, _ = load_SKEMPI2()

        # Preprocess PPI id
        df_skempi['PDB Id'] = df_skempi['#Pdb'].apply(
            lambda x: x.split('_', 1)[0].upper()
        )
        df_skempi['Partners'] = df_skempi['#Pdb'].apply(
            lambda x: set(x.split('_', 1)[1].replace('_', ''))
        )

        self.df_skempi = df_skempi

    def __call__(self, data: Data) -> Data:
        # Get PDB id and interacting partners from the PPI .pdb file
        pdb = path_to_pdb_id(data.path).upper()
        partners = set(path_to_partners(data.path))

        # Subselect the SKEMPI dataframe
        df = self.df_skempi[
            (self.df_skempi['PDB Id'] == pdb) &
            (self.df_skempi['Partners'] == partners)
        ]

        # Get node ids from data
        if self.strict_wt:
            node_id = data.node_id
        else:
            node_id = list(map(self.hide_graphein_wt, data.node_id))

        # From each row get node ids for `mut_pos`, mutant classes for `mut_sub` and ddG
        # annotations for `mut_ddg`. Additionaly, store `mut_wt`.
        mut_pos, mut_sub, mut_ddg, mut_wt = [], [], [], []
        # Store auxiliary mutation info not needed for training
        skempi_pdb, skempi_mut = [], []
        processed_muts = []  # To hande duplicated rows
        for _, row in df.iterrows():
            skip_mut = False
            row_mut_ddg = row['ddG']
            row_mut_pos = []
            row_mut_sub = []
            row_mut_wt = []

            # Interate over substituion points
            processed_point_muts = []  # to handle RA32E,KA34E,RB32E,KA34E in 2C5D_AB_CD
            for mut in row[self.mut_col].split(','):
                if mut in processed_point_muts:
                    continue
                processed_point_muts.append(mut)
                node = self.skempi_mut_to_graphein_node(mut)
                # Get integer id of a mutated node
                try:
                    row_mut_pos.append(node_id.index(node))
                except ValueError:  # One of the pos of a subst is not in the `data`
                    skip_mut = True
                    continue
                # Get substitution and wild-type class
                row_mut_sub.append(amino_acid_to_class(mut[-1]))
                row_mut_wt.append(amino_acid_to_class(mut[0]))

            # Skip if row is duplicated or some points of a mutation are not in the `data`
            if skip_mut:
                continue
            if (row_mut_pos, row_mut_sub, row_mut_ddg) in processed_muts:
                continue
            else:
                processed_muts.append((row_mut_pos, row_mut_sub, row_mut_ddg))

            # Append
            mut_ddg.append(torch.tensor(row_mut_ddg).float())
            mut_pos.append(torch.tensor(row_mut_pos).long())
            mut_sub.append(torch.tensor(row_mut_sub).long())
            mut_wt.append(torch.tensor(row_mut_wt).long())
            skempi_pdb.append(row['#Pdb'])
            skempi_mut.append(row[self.mut_col])

        # Pad
        mut_ddg = torch.tensor(mut_ddg)
        mut_pos = pad_fixed_length(mut_pos, self.max_n_substs, self.pad_val)
        mut_sub = pad_fixed_length(mut_sub, self.max_n_substs, self.pad_val)
        mut_wt = pad_fixed_length(mut_wt, self.max_n_substs, self.pad_val)

        # Store at `data`
        data.mut_ddg, data.mut_pos, data.mut_sub, data.mut_wt = mut_ddg, mut_pos, mut_sub, mut_wt
        data.skempi_pdb, data.skempi_mut = skempi_pdb, skempi_mut
        data.n_muts = len(data.mut_ddg)
        return data

    def skempi_mut_to_graphein_node(self, mut: str) -> str:
        # Parse SKEMPI mutation (up to last character representing substitution)
        # NOTE: Insertion codes in SKEMPI2 are lower case but we assume them to always be
        # upper case in PDB files. Therefore, we always convert them to upper case
        wt, chain, pos = mut[0], mut[1], mut[2:-1]
        wt = protein_letters_1to3[wt]
        ins = pos[-1] if not pos.isdigit() else ''
        ins = ins.upper()
        if ins:
            pos = pos[:-1]

        # Construct graphein node id
        if ins:
            node = f'{chain}:{wt}:{pos}:{ins}'
        else:
            node = f'{chain}:{wt}:{pos}'

        # Mask wild type
        if not self.strict_wt:
            node = self.hide_graphein_wt(node)

        return node
    
    def hide_graphein_wt(self, node_id: str) -> str:
        parts = node_id.split(':')
        node_id = f"{parts[0]}:???:{':'.join(parts[2:])}"
        return node_id

    @staticmethod
    def follow_batch_attrs() -> list[str]:
        return ['mut_ddg']
    
    @staticmethod
    def uncollate(data: Batch, increment: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Uncollates `mut_pos` and `mut_sub` into lists and increments `mut_pos` for batched nodes.
        """
        # Unpad
        lengths = data.mut_pos.ne(-1).sum(dim=1)
        mut_pos = unpad_sequence(data.mut_pos, lengths, batch_first=True)
        mut_sub = unpad_sequence(data.mut_sub, lengths, batch_first=True)
        mut_wt = unpad_sequence(data.mut_wt, lengths, batch_first=True)

        # Increment `mut_pos` for batched nodes
        if increment:
            mut_pos = [p + data.ptr[data.mut_ddg_batch[i]] for i, p in enumerate(mut_pos)]

        return mut_pos, mut_sub, mut_wt
    

class DDGLabelSamplerTransform(StochasticTransform):
    def __init__(
        self,
        n_samples: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def __call__(self, data: Data) -> Data:
        super().__call__(data)

        # Sample `n_samples` mutations
        if data.n_muts == 0:
            raise ValueError('`DDGLabelSamplerTransform` applied but no mutations found.')
        idx = torch.tensor(self.rng.sample(range(data.n_muts), self.n_samples))

        # Store at `data`
        data.mut_ddg = data.mut_ddg[idx]
        data.mut_pos = data.mut_pos[idx]
        data.mut_sub = data.mut_sub[idx]
        data.mut_wt = data.mut_wt[idx]
        data.skempi_pdb = data.skempi_pdb[idx]
        data.skempi_mut = data.skempi_mut[idx]
        data.n_muts = self.n_samples

        return data


class CleanPretransform(T.BaseTransform):
    def __init__(
        self,
        attrs: Iterable[str] = ('node_id', 'path')
    ):
        super().__init__()
        self.attrs = attrs

    def __call__(self, data: Data) -> Data:
        for attr in self.attrs:
            if hasattr(data, attr):
                delattr(data, attr)
        return data


class PrecomputeBasesPretransform(T.BaseTransform):
    def __init__(self, **bases_kwargs):
        super().__init__()
        if 'amp' in bases_kwargs:
            bases_kwargs['use_pad_trick'] = using_tensor_cores(bases_kwargs['amp'])
        self.bases_kwargs = bases_kwargs

    def __call__(self, data: Data) -> Data:
        src, dst = data.edge_index[0], data.edge_index[1]
        rel_pos = data.x[dst] - data.x[src]
        # TODO if not torch.cuda.is_available():
        data.bases = {k: v.cpu() for k, v in get_basis(rel_pos.cuda(), **self.bases_kwargs).items()}
        return data


class ComposeFilters():
    def __init__(self, filters: Iterable[Callable]):
        self.filters = filters

    def __call__(self, data: Data) -> Data:
        for filter in self.filters:
            if isinstance(data, (list, tuple)):
                retval = all([filter(d) for d in data])
            else:
                retval = filter(data)
            if retval is False:
                return False
        return True

    def __repr__(self) -> str:
        args = [f'  {filter}' for filter in self.filters]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))


class PPISizeFilter:
    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes

    def __call__(self, data: Data) -> bool:
        return data.num_nodes <= self.max_nodes


class DDGLabelFilter:
    def __call__(self, data: Data) -> bool:
        # Has at least one ddG label
        if not hasattr(data, 'n_muts'):
            raise ValueError('DDGFilter applied but not DDGLabelPretransform')
        return data.n_muts > 0


class DeepCopyTransform(T.BaseTransform):
    def __init__(self):
        super().__init__()
        warnings.warn('`DeepCopyTransform` may slow data processing of big data significantly as it currently uses `copy.deepcopy`.')

    def __call__(self, data: Data):
        return copy.deepcopy(data)


class DockingScorePretransform(T.BaseTransform):
    def __init__(self, csv_file_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_file_path)
    def __call__(self, data: Data) -> Data:
        name = data.path.split('/')[-1].split('.')[0].rsplit('_', 2)[0] + '.pdb'
        #find the FNAT in the dataframe with the same decoy name
        try:
            data.fnat = torch.tensor(self.df[self.df['structure'] == name]['fnat'].values[0], dtype=torch.float)
        except IndexError:
            # raise ValueError(f'No FNAT score found for {name}')
            warnings.warn(f'No FNAT score found for {name}')
            data.fnat = torch.nan
        return data


class DockingScoreFilter(T.BaseTransform):
    def __init__(self: str):
        super().__init__()
        
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'fnat'):
            return torch.isnan(torch.tensor(data.fnat))
        else:
            raise ValueError(f'DockingScoreFilter applied but no fnat score found for {data.path}')


# NOTE: Pre<encoder> transforms are not proper transforms (do not return Data)
        
class PreSE3TransformerTransform(T.BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Union[Data, Batch]) -> tuple:
        # Convert graph to DGL
        # data.x = data.x.half()  # TODO amp
        data_dgl = dgl.graph((data.edge_index[0], data.edge_index[1]))
        data_dgl.ndata['pos'] = data.x
        data_dgl.ndata['attr'] = data.f
        data_dgl.edata['rel_pos'] = _get_relative_pos(data_dgl)

        # Set node features
        node_feats = {'0': data_dgl.ndata['attr'][:, :, None]}
        if hasattr(data, 'virtual_c_beta_vector'):
            node_feats['1'] = data.virtual_c_beta_vector[:, None].float()
        
        # Set edge features
        edge_feats = None

        # Set pre-computed bases
        all_bases = data.bases if hasattr(data, 'bases') else None

        return data_dgl, node_feats, edge_feats, all_bases


class PreEquiformerTransform(T.BaseTransform):
    def __init__(
        self,
        coord_fill_value: float = 0.0,
        type1_features: Union[Iterable[str], str] = (  # subset of VECTOR_FEATURE_NAMES
            'virtual_c_beta_vector',
        ),
        divide_coords_by: float = 4.0,
        intra_inter_edge_features: bool = True
    ):
        super().__init__()
        self.coord_fill_value = coord_fill_value

        # Some reasonable combinations for easier passing through hydra
        if isinstance(type1_features, str):
            if type1_features in VECTOR_FEATURE_NAMES:
                self.type1_features = [type1_features]
            elif type1_features == 'virtual_c_beta_vector_and_neigh_res':
                self.type1_features = [
                    'virtual_c_beta_vector',
                    'sequence_neighbour_vector_n_to_c',
                    'sequence_neighbour_vector_c_to_n'
                ]
            elif type1_features == 'virtual_c_beta_vector_and_neigh_atoms':
                self.type1_features = [
                    'virtual_c_beta_vector',
                    'ca_to_n_vector',
                    'ca_to_c_vector'
                ]
            else:
                raise ValueError(f'Wrong `type1_features` value {type1_features}.')
        else:
            self.type1_features = type1_features

        self.divide_coords_by = divide_coords_by
        self.intra_inter_edge_features = intra_inter_edge_features

    def __call__(self, data: Union[Data, Batch]) -> tuple:
        if not isinstance(data, Batch):
            raise NotImplementedError(f'Not implemented for {type(data)}.')

        # Init type-0 features
        feats_0 = to_dense_batch(data.f, data.batch)[0]
        feats_0 = einops.rearrange(feats_0, 'b n d -> b n d 1')

        # Init type-1 features
        feats_1 = []    
        for feat_name in self.type1_features:
            # if hasattr(data, feat_name):
            feat = getattr(data, feat_name)
            feat = to_dense_batch(feat, data.batch, fill_value=self.coord_fill_value)[0]
            feats_1.append(feat)
        feats_1 = torch.stack(feats_1, dim=-2)

        # Init input fiber
        feats = {0: feats_0, 1: feats_1}

        # Init coords and sequence padding mask
        coors, mask = to_dense_batch(data.x, data.batch, fill_value=self.coord_fill_value)

        # Rescale
        coors /= self.divide_coords_by

        # Convert types
        feats = {t: f.float() for t, f in feats.items()}
        coors = coors.float()

        # Init inter/intra binary edge features
        if self.intra_inter_edge_features:
            max_nodes = max([len(sample_node_id) for sample_node_id in data.node_id])
            edges = []
            for sample_node_id in data.node_id:
                sample_chain_id = (list(map(lambda x: x.split(':', 1)[0], sample_node_id)))
                chain_sizes = Counter(sample_chain_id).values()
                blocks = [torch.ones(chain_size, chain_size, dtype=torch.long) for chain_size in chain_sizes]
                padding_size = max_nodes - sum(chain_sizes)
                blocks.append(torch.zeros(padding_size, padding_size, dtype=torch.long))
                sample_edges = torch.block_diag(*blocks)
                edges.append(sample_edges)
            edges = torch.stack(edges)
            edges = edges.to(coors.device)
        else:
            edges = None

        # Validate
        for deg, feat in feats.items():
            assert not contains_nan_or_inf(feat), f'feats[{deg}] contains NaN or Inf in {data.path}'
        assert not contains_nan_or_inf(coors), f'coords contains NaN or Inf in {data.path}'
        assert not contains_nan_or_inf(mask), f'mask contains NaN or Inf in {data.path}'

        return dict(inputs=feats, coors=coors, mask=mask, edges=edges)


class CompleteGraphTransform(T.BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data) -> Data:
        data.edge_index = torch.tensor(get_edges(data.num_nodes))
        return data
