import os
import warnings
from typing import Union
from pathlib import Path

import graphein
from tqdm import tqdm
import torch
import torch.multiprocessing
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, separate, collate

from ppiref.split import read_fold
from ppiformer.data.transforms import *
from ppiformer.definitions import PPIFORMER_PYG_DATA_CACHE_DIR
from  concurrent.futures import ProcessPoolExecutor


graphein.verbose(enabled=False)
warnings.simplefilter(action='ignore', category=FutureWarning)


class PPIInMemoryDataset(InMemoryDataset):

    n_classes = 20  # Number of amino acid types

    def __init__(
        self,
        split: Optional[str] = 'ppiref_filtered_clustered_03',  # PPIRef50K
        fold: Optional[str] = '8',  # 8 random samples
        raw_data_file_names: Optional[list[Union[str, Path]]] = None,
        verbose: bool = True,
        fresh: bool = False,
        pre_transform=T.Compose([PDBToPyGPretransform()]),
        pre_filter=ComposeFilters([]),
        transform=T.Compose([]),
        max_workers: Optional[int] = None,
        skip_data_on_processing_errors: bool = True
    ):
        """_summary_

        Args:
            split (str): PPIRef split.
            fresh (bool, optional): PPIRef fold.
            pre_transform (_type_, optional): _description_. Defaults to T.Compose([PDBToPyGPretransform()]).
            pre_filter (_type_, optional): _description_. Defaults to ComposeFilters([]).
            transform (_type_, optional): _description_. Defaults to T.Compose([]).
            max_workers (Optional[int], optional): _description_. Defaults to None.
            skip_data_on_processing_errors (bool, optional): _description_. Defaults to True.
        """
        # Process input args
        assert (split is not None and fold is not None) or raw_data_file_names is not None
        if raw_data_file_names is None:
            raw_data_file_names = read_fold(split, fold)
            dataset_id = f'{split}_{fold}'
        else:
            split, fold = None, None
            dataset_id = f'from_files'
        self.raw_data_file_names = raw_data_file_names
        self.split = split
        self.fold = fold
        self.dataset_id = dataset_id

        self.root = PPIFORMER_PYG_DATA_CACHE_DIR
        self.verbose = verbose
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() - 2
        self.skip_data_on_processing_errors = skip_data_on_processing_errors

        # Clean cache
        self.clean_cache(transforms_only=True)

        # Init
        super().__init__(self.root, transform, pre_transform, pre_filter, verbose)

        # Load data
        if len(self.processed_paths) > 1:
            data_list = []
            for p in self.processed_paths:
                data_collated, slices = torch.load(p)
                for idx in range(slices['x'].shape[0] - 1):  # estimate n. graphs in dataset by coordinate slices
                    data = separate.separate(cls=data_collated.__class__, batch=data_collated, slice_dict=slices, idx=idx, decrement=False)
                    data_list.append(data)
            print(f'Loaded {len(data_list)} graphs from {len(self.processed_paths)} files')
            print('Collating...')
            self.data, self.slices = self.collate(data_list=data_list)
        else:
            if fresh:
                self.process()
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])
        
        # Init attrbiutes
        self.n_features = self._data.f.shape[1]
        self.n_coords = self._data.x.shape[1]

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.raw_data_file_names

    @property
    def processed_file_names(self):
        if self.fold is not None and '+' in self.fold:
            fold_names = self.fold.split('+')
            files = [f'ppi_inmemory_dataset_{self.split}_{fold_name}.pt' for fold_name in fold_names]
            if all([os.path.isfile(self.root / "processed" / f) for f in files]):
                return files
            else:
                warnings.warn(f'Could not find all {len(files)} preprocessed data files.')
                return [f'ppi_inmemory_dataset_{self.split}_{self.fold}.pt']
        else:
            return [f'ppi_inmemory_dataset_{self.dataset_id}.pt']

    def pre_transform_chunk(self, chunk):
        data_list = []
        for path in tqdm(chunk, desc=f'Process {os.getpid()} preparing data'):
            try:
                data = self.pre_transform(path)
            except Exception as e:
                if not self.skip_data_on_processing_errors:
                    raise e
                else:
                    print(f'Process {os.getpid()} failed on {path}\n{e}')  # TODO Trace
            else:
                data_list.append(data)
        return data_list

    def process(self):
        # Read and process data points into the list of `Data`.
        # NOTE: the order of pre-filter and pre-transform is opposite to PyG docs
        if self.pre_transform is not None:
            if self.max_workers > 1:
                torch.multiprocessing.set_sharing_strategy('file_system')
                n_chunks = (self.max_workers - 2)
                chunksize = max(1, len(self.raw_paths) // n_chunks)
                chunks = [self.raw_paths[i:i + chunksize] for i in range(0, len(self.raw_paths), chunksize)]
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    data_list = list(executor.map(self.pre_transform_chunk, chunks))
                data_list = sum(data_list, start=[])
            else:
                data_list = self.pre_transform_chunk(self.raw_paths)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if len(self.raw_data_file_names) != len(data_list):
            warnings.warn(f'Only {len(data_list)} our of {len(self.raw_data_file_names)} files were read and processed.')

        assert len(data_list) > 0, 'No data were read and succesfully processed or all data were filtered out.'

        # Ensure safe collate (at least partially) by dropping data points with inconsistent attribute types
        bad_idx = []
        for attr in data_list[0].stores[0].keys():
            expected_type = type(getattr(data_list[0], attr))
            for i, data in enumerate(data_list):
                real_type = type(getattr(data, attr))
                if real_type != expected_type:
                    bad_idx.append(i)
                    msg = f'Inconsistent attribute type for {attr} in {data.path}. Real: {real_type}, expected: {expected_type}.'
                    if self.skip_data_on_processing_errors:
                        print(msg)
                    else:
                        raise ValueError(msg)
        bad_idx = set(bad_idx)
        data_list = [data for i, data in enumerate(data_list) if i not in bad_idx]

        # Collate and save
        data, slices = self.collate(data_list)
        self.data, self.slices = data, slices
        torch.save((data, slices), self.processed_paths[0])

    def clean_cache(self, transforms_only: bool = False) -> None:
        processed_dir = self.root / 'processed'
        cache_files = [processed_dir / 'pre_transform.pt', processed_dir / 'pre_filter.pt']
        if not transforms_only:
            cache_files += [processed_dir / name for name in self.processed_file_names]

        for path in cache_files:
            path.unlink(missing_ok=True)
        
    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        retval = f'{self.__class__.__name__}({arg_repr})'
        if hasattr(self._data, 'n_muts'):
            n_muts = self._data.n_muts
            n_muts = n_muts.sum().item() if isinstance(n_muts, torch.Tensor) else n_muts
            retval = retval[:-1] + f', n_muts={n_muts})'
        return retval
