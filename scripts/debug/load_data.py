from tqdm import tqdm
from ppiformer.data.loader import get_dataloader
from ppiformer.data.transforms import *


def main():
    datasets = [
        'ppiref_filtered_sample_500,whole',
        'ppiref_filtered_clustered_03,whole',
        'ppiref_filtered_sample_500,whole',
    ]

    n_iters = 8
    # n_iters = 300
    # n_iters = 100_000

    dataloaders = []
    for dataset in datasets:
        dataloader = get_dataloader(
            dataset=dataset,
            pretransform=[],
            transform=[DeepCopyTransform(), CompleteGraphTransform()],
            transform=[MaskedModelingTransform(mask_sum=1)],
            num_workers=0
        )
        dataloaders.append(dataloader)
        i = 0
        for data in tqdm(iter(dataloader), desc=dataset, total=n_iters):
            i += 1
            if i == n_iters:
                break
        # data.to('cuda')


if __name__ == '__main__':
    main()
