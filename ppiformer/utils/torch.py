from typing import Any

import torch


def get_n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_fixed_length(
    seqs: list,
    length: int,
    padding_value: Any
) -> torch.Tensor:
    if len(seqs) == 0:
        return torch.tensor([])

    # Pad first sequence to fixed length
    constant_padder = torch.nn.ConstantPad1d(
        (0, length - len(seqs[0])),
        padding_value
    )
    seqs[0] = constant_padder(seqs[0])

    # Pad the rest
    return torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=padding_value
    )


# NOTE: Now merged to torch master
def unpad_sequence(
    padded_sequences,
    lengths,
    batch_first: bool = False,
):
    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length, device=lengths.device)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def contains_nan_or_inf(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


class ScaledTanh(torch.nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.low + (self.high-self.low)*(self.tanh(x)+1)/2


def fill_diagonal(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Fill the diagonal in last two dimension of a tensor with a value.
    """
    assert tensor.ndim >= 2
    assert tensor.shape[-1] == tensor.shape[-2]
    tensor[..., torch.arange(tensor.shape[-1]), torch.arange(tensor.shape[-1])] = value
    return tensor
