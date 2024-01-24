from tqdm import tqdm
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from ppiformer.data.transforms import *


def main():
    transform = T.Compose([DeepCopyTransform(), CompleteGraphTransform()])
    datasets = [QM9(root='/tmp/QM9', transform=transform)]
    datasets.append(datasets[0][:len(datasets[0]) // 8])
    datasets = datasets[::-1]

    n_iters = 30

    for dataset in datasets:
        i = 0
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        for batch in tqdm(loader, total=len(loader)):
            if i == n_iters:
                break
            i += 1


if __name__ == '__main__':
    main()