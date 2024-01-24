import math

import torch

from ppiformer.model.ppiformer import PPIformer


def test_calc_log_odds():
    y_proba = torch.tensor([
        [0.5, 0.5],  # 20, not 2 classes in practice
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    y_true = torch.tensor([0, 1, 0])

    log_odds_expected = torch.tensor([
        [0.0, 0.0],
        [math.log(0.8) - math.log(0.2), 0.0],
        [0.0, math.log(0.1) - math.log(0.9)],
    ])

    assert torch.equal(PPIformer.calc_log_odds(y_proba, y_true), log_odds_expected)
