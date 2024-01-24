import sys
import inspect
import pathlib


# Paths
PPIFORMER_ROOT_DIR = pathlib.Path(__file__).parent.absolute().parent
PPIFORMER_TEST_DATA_DIR = PPIFORMER_ROOT_DIR / 'tests/data'
PPIFORMER_WEIGHTS_DIR = PPIFORMER_ROOT_DIR / 'weights'
PPIFORMER_PYG_DATA_CACHE_DIR = PPIFORMER_ROOT_DIR / '.pyg_dataset_cache'
TRAINING_SUBMISSION_DIR = PPIFORMER_ROOT_DIR / 'scripts/jobs'

# Dependecies
try:  # Determine if SE(3)-Transformer is required by checking whether it is installed or not
    import se3_transformer
    SE3TRANSFORMER_REQUIRED = True
except ImportError:
    SE3TRANSFORMER_REQUIRED = False
