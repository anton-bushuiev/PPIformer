import typing
from typing import Literal, Iterable, Any


TASK_TYPE = Literal[
    'masked_modeling',
    'ddg_regression_wt_marginals',
    'ddg_regression_masked_marginals',
    'docking_scoring'
]
DDG_INFERENCE_TYPE = Literal[
    'masked_marginals',
    'wt_marginals',  # basline
    'embedding_difference',  # baseline
    'embedding_concatenation'  # baseline
]


def task_to_ddg_inference_type(task: TASK_TYPE) -> DDG_INFERENCE_TYPE:
    if not task.startswith('ddg_regression'):
        raise ValueError(f'Wrong task {task}.')
    ddg_inference = task.split('_', 2)[-1]
    if ddg_inference not in typing.get_args(DDG_INFERENCE_TYPE):
        raise ValueError(f'Wrong task suffix for ddG inference {ddg_inference}.')
    return ddg_inference


def none_or_str(value: Any) -> Any:
    if value == 'None':
        return None
    return value


def is_iterable_and_not_str(obj: Any) -> bool:
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def wrap_into_list(obj: Any) -> list:
    if is_iterable_and_not_str(obj):
        return obj
    return [obj]
