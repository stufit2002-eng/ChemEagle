"""CUDA_MODEL_LOCK — re-exported from _shared_models for backward compatibility.

All agent modules that previously imported from this file continue to work
unchanged.  The lock itself now lives in _shared_models so it is a single
object shared across the entire process.
"""

from _shared_models import CUDA_MODEL_LOCK  # noqa: F401
