"""
Shared re-entrant lock for CUDA model inference.

All local ML models (RxnIM / model1, ChemIEToolkit / model,
ChemNER / cner) are module-level singletons whose internal state
(decoder cache, pre_kv tensors, attention buffers) is NOT thread-safe.

Import CUDA_MODEL_LOCK in every sub-agent file and wrap each
model.predict / model.extract call with:

    with CUDA_MODEL_LOCK:
        result = model.some_method(...)

Azure/OS API calls (network I/O) do NOT need the lock — they release
the GIL while waiting, so multiple threads can overlap their API waits
even when only one holds the lock for local inference.
"""

import threading

CUDA_MODEL_LOCK = threading.RLock()
