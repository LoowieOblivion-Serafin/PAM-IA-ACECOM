# Fix for PyTorch Lightning 2.x compatibility
# In newer versions, utilities.distributed was reorganized

import sys
import importlib

# Try to import from the new location first, then fall back to old
try:
    from pytorch_lightning.utilities import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.utilities.distributed import rank_zero_only
    except ImportError:
        # If both fail, define a dummy function
        def rank_zero_only(fn):
            return fn

# Inject it into the proper module location for backward compatibility
if 'pytorch_lightning.utilities.distributed' not in sys.modules:
    distributed_module = type(sys)('pytorch_lightning.utilities.distributed')
    distributed_module.rank_zero_only = rank_zero_only
    sys.modules['pytorch_lightning.utilities.distributed'] = distributed_module
