"""
CUDA_MODEL_LOCK — a module-level threading.Lock that serialises all
local model inference (RxnIM, ChemIEToolkit / MolNexTR, easyocr).

Why a real lock is required
----------------------------
When process_pdf.py runs with --workers N, N threads share the SAME
module-level model instances (model, model1 in each agent module).
Several components inside those instances are NOT concurrency-safe:

  • easyocr.Reader  – internal CUDA buffers are overwritten by
                      concurrent readtext() calls, producing corrupt or
                      empty OCR results.
  • MolNexTR.convert_graph_to_output – may return [] when invoked
                      simultaneously, causing downstream
                      "list index out of range" at output[0]['smiles'].
  • Any other stateful sub-model called from predict_image_file or
    extract_molecule_corefs_from_figures.

The threading.local() decoder-state fix in
  rxnim/transformer/decoder.py
  molnextr/transformer/decoder.py
addresses only the autoregressive decoder cache.  It does NOT cover
easyocr or the rest of the inference pipeline.

Serialising model inference with this lock is safe because:
  • GPU inference is fast (typically < 2 s per crop).
  • The slow operations — LLM API calls — are NOT wrapped in this lock,
    so they still execute concurrently across all worker threads.
  • Net effect: LLM latency is fully overlapped; only the short model
    inference window is serialised.
"""

import threading

CUDA_MODEL_LOCK = threading.Lock()
