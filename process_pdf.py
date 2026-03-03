#!/usr/bin/env python3
"""
ChemEagle PDF Pipeline
======================
Given a PDF, this script:

  1. Converts every page to a PIL image                    (pdf2image)
  2. Detects figures and tables on each page               (VisualHeist / Florence-2)
  3. Crops and saves each detection                        (PNG)
  4. Runs ChemEagle on every cropped image                 (Azure or OS backend)
  5. Writes all results to a structured output directory

Output layout:
    <output_root>/<pdf_stem>_<YYYYMMDD_HHMMSS>/
        summary.json               ← top-level metadata + per-crop index
        page_01/
            crop_01_figure.png     ← cropped image
            crop_01_figure_result.json   ← ChemEagle output
            crop_01_figure_meta.json     ← bbox, label, timing, success flag
            crop_02_table.png
            crop_02_table_result.json
            ...
        page_02/
            ...

Usage:
    python process_pdf.py paper.pdf
    python process_pdf.py paper.pdf --output-dir ./my_results --model base
    python process_pdf.py paper.pdf --workers 8
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

# ── CUDA guard ────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required but not available. "
        "Ensure CUDA drivers and a CUDA-capable GPU are present."
    )

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True   # auto-tune conv kernels

# Thread-safe print lock
_print_lock = threading.Lock()

def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


# ── VisualHeist (Florence-2) detection — CUDA-aware wrapper ──────────────────

def _load_visualheist(model_size: str):
    """Load the VisualHeist model onto CUDA and return (model, processor)."""
    from pdfmodel.methods import _create_model, LARGE_MODEL_ID, BASE_MODEL_ID

    use_large = model_size.lower() == "large"
    model_id  = LARGE_MODEL_ID if use_large else BASE_MODEL_ID
    tag       = "large" if use_large else "base"

    print(f"  Loading VisualHeist-{tag} …", flush=True)
    model, processor = _create_model(model_id, tag)
    model = model.to(DEVICE).eval()
    print("  VisualHeist loaded on CUDA.")
    return model, processor


def _detect_figures(page_image: Image.Image, model, processor) -> dict:
    """Run VisualHeist object detection on one page image (CUDA).

    Returns the annotation dict: {"bboxes": [...], "labels": [...]}
    """
    prompt = "<OD>"
    inputs = processor(text=prompt, images=page_image, return_tensors="pt")
    # Move all tensor inputs to CUDA
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    annotation = processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(page_image.width, page_image.height),
    )
    return annotation["<OD>"]


# ── Per-crop ChemEagle worker ─────────────────────────────────────────────────

def _run_crop(task: dict, chemeagle_fn) -> dict:
    """
    Execute ChemEagle on one cropped image and write result + meta JSON files.

    Parameters
    ----------
    task : dict with keys:
        page_num, crop_num, label, bbox,
        crop_img_path, crop_result_path, crop_meta_path,
        skipped (bool), skip_reason (str|None)
    chemeagle_fn : callable — ChemEagle or ChemEagle_OS

    Returns
    -------
    dict with keys: page_num, crop_num, success, error, elapsed, n_rxn, result
    """
    page_num        = task["page_num"]
    crop_num        = task["crop_num"]
    crop_stem       = task["crop_stem"]
    label           = task["label"]
    bbox            = task["bbox"]
    crop_img_path   = task["crop_img_path"]
    crop_result_path = task["crop_result_path"]
    crop_meta_path  = task["crop_meta_path"]

    # Skipped crops (e.g. tables)
    if task.get("skipped"):
        result  = {"skipped": True, "reason": task.get("skip_reason", "")}
        success = None
        error   = None
        elapsed = 0.0
        _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  [skipped — {task.get('skip_reason', 'table')}]")
    else:
        t0      = time.perf_counter()
        result  = None
        error   = None
        success = False
        try:
            result  = chemeagle_fn(str(crop_img_path))
            elapsed = time.perf_counter() - t0
            success = True
            n_rxn   = len((result or {}).get("reactions", []))
            _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  ok  ({elapsed:.1f}s, {n_rxn} reaction(s))")
        except Exception as ce:
            elapsed = time.perf_counter() - t0
            error   = str(ce)
            result  = {"error": error, "parsed": False}
            _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  FAILED  ({error[:80]})")

    # Write result JSON
    crop_result_path.write_text(
        json.dumps(result or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Write per-crop metadata
    meta = {
        "page":         page_num,
        "crop":         crop_num,
        "label":        label,
        "bbox":         [float(v) for v in bbox],
        "image_file":   crop_img_path.name,
        "result_file":  crop_result_path.name,
        "processing_s": round(elapsed, 2),
        "success":      success,
        "error":        error,
    }
    crop_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "page_num":  page_num,
        "crop_num":  crop_num,
        "success":   success,
        "error":     error,
        "elapsed":   elapsed,
        "n_rxn":     len((result or {}).get("reactions", [])) if success else 0,
        "result":    result,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    output_root: str = "./pdf_results",
    model_size: str = "large",
    backend: str = "azure",
    skip_tables: bool = False,
    workers: int = 4,
) -> Path:
    """
    Run the full PDF → crop → ChemEagle pipeline.

    Parameters
    ----------
    pdf_path    : Path to the input PDF.
    output_root : Root directory for all output runs.
    model_size  : VisualHeist model to use — "base" or "large".
    backend     : "azure"  → use ChemEagle (AzureOpenAI)
                  "os"     → use ChemEagle_OS (vLLM / Ollama)
    skip_tables : If True, skip crops labelled as "table" (only process figures).
    workers     : Number of parallel ChemEagle workers (default 4).
                  Set to 1 to disable parallelism.

    Returns
    -------
    Path to the run output directory.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Output directory ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{pdf_path.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory : {run_dir}")

    # ── Step 1: PDF → per-page PIL images ────────────────────────────────────
    print("\n[1/3] Converting PDF pages to images …")
    from pdfmodel.methods import _pdf_to_image
    page_images = _pdf_to_image(str(pdf_path))
    print(f"      {len(page_images)} page(s) loaded.")

    # ── Step 2: Load models ───────────────────────────────────────────────────
    print("\n[2/3] Loading models …")
    vh_model, vh_processor = _load_visualheist(model_size)

    # Lazy-import ChemEagle here so VisualHeist loading prints first
    if backend == "azure":
        from main import ChemEagle as _run_chemeagle
        chemeagle_fn = _run_chemeagle
    else:
        from main import ChemEagle_OS as _run_chemeagle_os
        chemeagle_fn = _run_chemeagle_os

    print(f"      ChemEagle backend : {backend}")
    print(f"      Parallel workers  : {workers}")

    # ── Step 3a: Detection — sequential (GPU) ─────────────────────────────────
    print("\n[3/3] Detecting figures …\n")

    summary: dict = {
        "pdf":           str(pdf_path),
        "timestamp":     datetime.now().isoformat(),
        "model_size":    model_size,
        "backend":       backend,
        "workers":       workers,
        "n_pages":       len(page_images),
        "skip_tables":   skip_tables,
        "pages":         [],
    }

    # Build a flat list of crop tasks (detection runs first, sequentially)
    all_tasks: list[dict] = []
    page_entries: list[dict] = []

    for page_idx, page_img in enumerate(page_images):
        page_num = page_idx + 1
        print(f"  ── Page {page_num}/{len(page_images)} ", end="", flush=True)

        try:
            annotation = _detect_figures(page_img, vh_model, vh_processor)
        except Exception as det_err:
            print(f"[detection failed: {det_err}]")
            page_entries.append({"page": page_num, "n_detected": 0, "crops": []})
            continue

        bboxes = annotation.get("bboxes", [])
        labels = annotation.get("labels", [])
        print(f"→ {len(bboxes)} object(s) detected")

        if not bboxes:
            page_entries.append({"page": page_num, "n_detected": 0, "crops": []})
            continue

        page_dir = run_dir / f"page_{page_num:02d}"
        page_dir.mkdir(exist_ok=True)

        page_entry: dict = {
            "page":       page_num,
            "n_detected": len(bboxes),
            "crops":      [],
        }

        for crop_idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            crop_num   = crop_idx + 1
            label_slug = label.lower().replace(" ", "_")
            crop_stem  = f"crop_{crop_num:02d}_{label_slug}"

            crop_img_path    = page_dir / f"{crop_stem}.png"
            crop_result_path = page_dir / f"{crop_stem}_result.json"
            crop_meta_path   = page_dir / f"{crop_stem}_meta.json"

            # Crop and save now (sequential, avoids CUDA contention later)
            x1, y1, x2, y2 = bbox
            page_img.crop((x1, y1, x2, y2)).save(str(crop_img_path))

            is_table  = skip_tables and "table" in label.lower()
            all_tasks.append({
                "page_num":        page_num,
                "crop_num":        crop_num,
                "crop_stem":       crop_stem,
                "label":           label,
                "bbox":            bbox,
                "crop_img_path":   crop_img_path,
                "crop_result_path": crop_result_path,
                "crop_meta_path":  crop_meta_path,
                "skipped":         is_table,
                "skip_reason":     "table" if is_table else None,
                "run_dir":         run_dir,
            })

            # Reserve slot in page_entry (filled in after parallel run)
            page_entry["crops"].append({
                "crop":        crop_num,
                "label":       label,
                "bbox":        [float(v) for v in bbox],
                "image_file":  str(crop_img_path.relative_to(run_dir)),
                "result_file": str(crop_result_path.relative_to(run_dir)),
                "success":     None,   # filled after parallel phase
            })

        page_entries.append(page_entry)

    total_crops = len(all_tasks)
    print(f"\n  {total_crops} crop(s) queued across {len(page_images)} page(s).")

    # ── Step 3b: ChemEagle — parallel ────────────────────────────────────────
    print(f"\n  Running ChemEagle with {workers} worker(s) …\n")

    total_success = 0
    total_skipped = 0
    total_failed  = 0

    # Index results by (page_num, crop_num) for merging back into page_entries
    results_index: dict[tuple[int, int], dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(_run_crop, task, chemeagle_fn): task
            for task in all_tasks
        }
        for future in as_completed(future_to_task):
            try:
                info = future.result()
            except Exception as exc:
                task = future_to_task[future]
                info = {
                    "page_num": task["page_num"],
                    "crop_num": task["crop_num"],
                    "success":  False,
                    "error":    str(exc),
                    "elapsed":  0.0,
                    "n_rxn":    0,
                }

            key = (info["page_num"], info["crop_num"])
            results_index[key] = info

            if info["success"] is True:
                total_success += 1
            elif info["success"] is None:
                total_skipped += 1
            else:
                total_failed += 1

    # ── Step 3c: Merge results back into page_entries ─────────────────────────
    for page_entry in page_entries:
        for crop_slot in page_entry["crops"]:
            key = (page_entry["page"], crop_slot["crop"])
            info = results_index.get(key)
            if info is not None:
                crop_slot["success"] = info["success"]

    # ── Write top-level summary ───────────────────────────────────────────────
    summary["pages"] = page_entries
    summary.update({
        "n_crops_total": total_crops,
        "n_success":     total_success,
        "n_skipped":     total_skipped,
        "n_failed":      total_failed,
        "completed_at":  datetime.now().isoformat(),
    })
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"  PDF pipeline complete")
    print(f"  PDF      : {pdf_path.name}")
    print(f"  Pages    : {len(page_images)}")
    print(f"  Crops    : {total_crops}  "
          f"(success={total_success}, failed={total_failed}, skipped={total_skipped})")
    print(f"  Workers  : {workers}")
    print(f"  Output   : {run_dir}")
    print(f"{'='*60}\n")

    return run_dir


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract figures from a PDF and run ChemEagle on each one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output-dir", default="./pdf_results",
        help="Root directory for output runs.",
    )
    parser.add_argument(
        "--model", choices=["base", "large"], default="large",
        help="VisualHeist model size for figure/table detection.",
    )
    parser.add_argument(
        "--backend", choices=["azure", "os"], default="azure",
        help="ChemEagle backend: 'azure' (AzureOpenAI) or 'os' (vLLM/Ollama).",
    )
    parser.add_argument(
        "--skip-tables", action="store_true",
        help="Skip crops whose label contains 'table' (only process figures).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel ChemEagle workers. Set to 1 to disable parallelism.",
    )
    args = parser.parse_args()

    process_pdf(
        pdf_path=args.pdf,
        output_root=args.output_dir,
        model_size=args.model,
        backend=args.backend,
        skip_tables=args.skip_tables,
        workers=args.workers,
    )
