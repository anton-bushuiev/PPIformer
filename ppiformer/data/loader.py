from typing import Optional, Union, Iterable

from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader

from ppiformer.data.transforms import *
from ppiformer.data.dataset import PPIInMemoryDataset


# TODO class PPIDataLoader(DataLoader)
# - put `get_dataloader` logic into constructor
# - extend `Collater` to properly collate `node_id` of Graphein strings into a single list


def get_dataloader(
    dataset: Union[str, Iterable[Union[str, Path]]],
    pretransform: Iterable[BaseTransform] = tuple(),
    prefilter: Iterable[BaseTransform] = tuple(),
    transform: Iterable[BaseTransform] = tuple(),
    dataset_max_workers: Optional[int] = None,
    shuffle: bool = True,
    batch_size: int = 8,
    num_workers: int = 0,
    fresh: bool = False,
    deterministic: bool = False,
    verbose: bool = True,
    skip_data_on_processing_errors: bool = True,
    **kwargs
) -> DataLoader:
    """_summary_

    Args:
        dataset (Union[str, list[Union[str, Path]]]): PPIRef "<dataset>,<split>" or list of paths to .pdb files
        prefilter (Iterable[BaseTransform], optional): _description_. Defaults to tuple().
        transform (Iterable[BaseTransform], optional): _description_. Defaults to tuple().
        dataset_max_workers (Optional[int], optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to True.
        batch_size (int, optional): _description_. Defaults to 8.
        num_workers (int, optional): _description_. Defaults to 0.
        fresh (bool, optional): _description_. Defaults to False.
        deterministic (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    # Make all transforms determenisitic if debugging
    if deterministic:
        for t in pretransform + transform:
            if isinstance(t, StochasticTransform):
                t.deterministic = True

    # Preprocess dataset argument
    if isinstance(dataset, str):
        split, fold = parse_dataset_argument(dataset)
        raw_data_file_names = None
    elif isinstance(dataset, Iterable):
        split, fold = None, None
        raw_data_file_names = dataset
    else:
        raise ValueError(f'Invalid dataset argument of type {type(dataset)}.')
    dataset_name = dataset if isinstance(dataset, str) else f'{len(list(dataset))} PPIs'

    # Load dataset
    dataset = PPIInMemoryDataset(
        split=split,
        fold=fold,
        raw_data_file_names=raw_data_file_names,
        pre_transform=T.Compose(pretransform),
        pre_filter=ComposeFilters(prefilter),
        transform=T.Compose(transform),
        fresh=fresh,
        max_workers=dataset_max_workers,
        skip_data_on_processing_errors=skip_data_on_processing_errors,
    )
    if verbose:
        print(f'{dataset_name} loaded: {dataset}')

    # Get data attributes produced by pre-processing that have their own batching (different
    # from node batching)
    follow_batch = []
    for t in pretransform + transform:
        if hasattr(t, 'follow_batch_attrs'):
            follow_batch.extend(t.follow_batch_attrs())

    # Convert to dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=shuffle,
        follow_batch=follow_batch,
        **kwargs
    )
    return dataloader


def parse_dataset_argument(arg: str) -> tuple[str, str]:
    split, fold = arg.strip().replace(' ', '').split(',')
    return split, fold
