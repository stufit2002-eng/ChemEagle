"""_shared_models.py – Singleton models, caches, and API clients for ChemEagle.

All agent modules import from here so that:
  • ChemIEToolkit and RxnIM are loaded exactly once (previously loaded 4×).
  • Base64 image encoding is cached per (file-path, mtime) — no re-reads.
  • Prompt files are cached in memory — no disk I/O per call.
  • AzureOpenAI / OpenAI clients are reused across calls.
  • retry_api_call is defined once (previously duplicated across 5 files).
  • CUDA_MODEL_LOCK is re-exported for backward compatibility.

Thread safety
-------------
  • Model singletons use double-checked locking so the first caller loads
    the model while subsequent callers wait rather than loading again.
  • get_raw_prediction() and get_coref_results() use per-path locks so that
    two workers processing the same image path wait for one computation
    rather than duplicating it.
"""

from __future__ import annotations

import base64
import functools
import os
import threading
import time
from typing import Optional

import torch
from openai import APIError, AzureOpenAI, InternalServerError, OpenAI, RateLimitError

# ── Device ────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required but not available. "
        "Ensure CUDA drivers and a CUDA-capable GPU are present."
    )

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda")

# ── CUDA inference lock (re-exported for backward compat with _model_lock.py) ─
# Serialises all local model inference (easyocr, MolNexTR, etc.) so that
# concurrent worker threads do not corrupt shared CUDA state.
CUDA_MODEL_LOCK = threading.Lock()

# ── Lazy-loaded model singletons ──────────────────────────────────────────────
_chemiie_toolkit = None
_rxnim = None
_chemiie_lock = threading.Lock()
_rxnim_lock = threading.Lock()


def get_chemiie_toolkit():
    """Return the shared ChemIEToolkit instance, loading it on first call."""
    global _chemiie_toolkit
    if _chemiie_toolkit is None:
        with _chemiie_lock:
            if _chemiie_toolkit is None:
                from chemietoolkit import ChemIEToolkit  # noqa: PLC0415
                print("  [shared] Loading ChemIEToolkit …", flush=True)
                _chemiie_toolkit = ChemIEToolkit(device=DEVICE)
                print("  [shared] ChemIEToolkit ready.", flush=True)
    return _chemiie_toolkit


def get_rxnim():
    """Return the shared RxnIM instance, loading it on first call."""
    global _rxnim
    if _rxnim is None:
        with _rxnim_lock:
            if _rxnim is None:
                from rxnim import RxnIM  # noqa: PLC0415
                print("  [shared] Loading RxnIM …", flush=True)
                _rxnim = RxnIM("./rxn.ckpt", device=DEVICE)
                print("  [shared] RxnIM ready.", flush=True)
    return _rxnim


# ── Image encoding (cached per path + mtime) ──────────────────────────────────
@functools.lru_cache(maxsize=256)
def _encode_image_impl(image_path: str, mtime: float) -> str:
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def encode_image(image_path: str) -> str:
    """Return a base64 PNG string for *image_path*, invalidated by mtime."""
    mtime = os.path.getmtime(image_path)
    return _encode_image_impl(image_path, mtime)


# ── Prompt file caching ───────────────────────────────────────────────────────
@functools.lru_cache(maxsize=64)
def read_prompt(path: str) -> str:
    """Read a prompt file and keep it in memory for subsequent calls."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ── Azure OpenAI client singleton ─────────────────────────────────────────────
_azure_client: Optional[AzureOpenAI] = None
_azure_client_lock = threading.Lock()


def get_azure_client() -> AzureOpenAI:
    """Return the shared AzureOpenAI client, creating it on first call."""
    global _azure_client
    if _azure_client is None:
        with _azure_client_lock:
            if _azure_client is None:
                api_key = os.getenv("API_KEY")
                endpoint = os.getenv("AZURE_ENDPOINT")
                version = os.getenv("API_VERSION")
                if not api_key or not endpoint:
                    raise ValueError(
                        "API_KEY and AZURE_ENDPOINT environment variables must be set."
                    )
                _azure_client = AzureOpenAI(
                    api_key=api_key,
                    api_version=version,
                    azure_endpoint=endpoint,
                )
    return _azure_client


# ── Open-source (vLLM / Ollama) client singletons ────────────────────────────
_os_clients: dict[tuple[str, str], OpenAI] = {}
_os_clients_lock = threading.Lock()


def get_os_client(base_url: str, api_key: str) -> OpenAI:
    """Return (or create) a cached OpenAI-compatible client."""
    key = (base_url, api_key)
    if key not in _os_clients:
        with _os_clients_lock:
            if key not in _os_clients:
                _os_clients[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _os_clients[key]


def resolve_os_client(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAI:
    """Resolve env-var defaults and return a cached OS client."""
    url = base_url or os.getenv(
        "VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1")
    )
    key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))
    return get_os_client(url, key)


# ── Raw-prediction cache (RxnIM.predict_image_file) ──────────────────────────
# Shared across all agent modules.  Keyed by image_path string.
# Per-path locks prevent duplicate GPU computation when two workers race on
# the same crop path.  The CUDA_MODEL_LOCK is held only during actual inference.
_predict_cache: dict[str, list] = {}
_predict_path_locks: dict[str, threading.Lock] = {}
_predict_meta_lock = threading.Lock()


def get_raw_prediction(image_path: str) -> list:
    """Call RxnIM.predict_image_file once per image path and cache the result.

    Concurrent callers with the same path block until the first computation
    completes rather than launching duplicate GPU work.
    """
    if image_path in _predict_cache:
        return _predict_cache[image_path]

    with _predict_meta_lock:
        if image_path not in _predict_path_locks:
            _predict_path_locks[image_path] = threading.Lock()
        img_lock = _predict_path_locks[image_path]

    with img_lock:
        if image_path not in _predict_cache:
            with CUDA_MODEL_LOCK:
                result = get_rxnim().predict_image_file(
                    image_path, molnextr=True, ocr=True
                )
            _predict_cache[image_path] = result
        return _predict_cache[image_path]


# ── Coref-results cache (ChemIEToolkit.extract_molecule_corefs_from_figures) ──
# Same pattern as _predict_cache above.
_coref_cache: dict[str, list] = {}
_coref_path_locks: dict[str, threading.Lock] = {}
_coref_meta_lock = threading.Lock()


def get_coref_results(image_path: str) -> list:
    """Call extract_molecule_corefs_from_figures once per image path (cached).

    Concurrent callers with the same path block until the first computation
    completes rather than launching duplicate GPU work.
    """
    if image_path in _coref_cache:
        return _coref_cache[image_path]

    with _coref_meta_lock:
        if image_path not in _coref_path_locks:
            _coref_path_locks[image_path] = threading.Lock()
        img_lock = _coref_path_locks[image_path]

    with img_lock:
        if image_path not in _coref_cache:
            from PIL import Image  # noqa: PLC0415
            image = Image.open(image_path).convert("RGB")
            with CUDA_MODEL_LOCK:
                result = get_chemiie_toolkit().extract_molecule_corefs_from_figures(
                    [image]
                )
            _coref_cache[image_path] = result
        return _coref_cache[image_path]


# ── Retry helper (previously duplicated across 5 agent files) ─────────────────
def retry_api_call(
    func,
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs,
):
    """Call *func* with *args*/*kwargs*, retrying on 503 / overloaded errors.

    Uses exponential back-off: delay = base_delay * backoff_factor^attempt.
    All other exceptions propagate immediately without retry.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (InternalServerError, RateLimitError, APIError) as exc:
            last_exc = exc
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            msg = str(exc)
            if code == 503 or "overloaded" in msg.lower() or "503" in msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (backoff_factor**attempt)
                    print(
                        f"⚠️ API call failed (503/overloaded), "
                        f"retry {attempt + 1}/{max_retries} in {delay:.1f}s …"
                    )
                    time.sleep(delay)
                    continue
            raise
        except Exception:
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("API call failed (unknown error)")
