"""
CUDA_MODEL_LOCK — now a no-op context manager.

Thread-safety is handled at the root cause:
  rxnim/transformer/decoder.py  and
  molnextr/transformer/decoder.py

both use threading.local() for their decoder state dict, so concurrent
inference calls on the same model instance never share the cache.

CUDA_MODEL_LOCK is kept here as a nullcontext so existing call sites
(with CUDA_MODEL_LOCK: ...) compile and run without any locking overhead.
"""

import contextlib

CUDA_MODEL_LOCK = contextlib.nullcontext()
