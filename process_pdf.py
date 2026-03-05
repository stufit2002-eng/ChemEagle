#!/usr/bin/env python3
"""
ChemEagle PDF Pipeline
======================
Given a PDF, this script:

  1. Converts every page to a PIL image                    (pdf2image)
  2. Detects figures and tables on each page               (VisualHeist / Florence-2)
  3. Crops and saves each detection                        (PNG)
  4. Runs ChemEagle on every cropped image                 (Azure or OS backend)
     — Phase 3 uses a ThreadPoolExecutor so multiple crops are processed
       concurrently (LLM API calls overlap; local GPU models are made
       thread-safe via per-thread decoder state).
  5. Writes all results to a structured output directory

Quality checks + retry
----------------------
Each crop is retried up to MAX_RETRIES (3) times when any of the following
are detected in the output:

  1. Processing error  — ChemEagle raised an exception.
  2. R-group not substituted — 2 or more components (reactants + products) in
     non-template reactions still contain * or [R…] SMILES placeholders.
  3. Invalid SMILES — any reactant/product SMILES string is chemically invalid
     (validated with RDKit when available).

After MAX_RETRIES attempts, any crop that still fails any check is flagged
``human_review_required: true`` in both its result JSON and metadata JSON,
and tallied separately in summary.json.

Output layout:
    <output_root>/<pdf_stem>_<YYYYMMDD_HHMMSS>/
        summary.json               ← top-level metadata + per-crop index
        hitl_crops.json            ← list of crops requiring human review
        page_01/
            crop_01_figure.png
            crop_01_figure_result.json
            crop_01_figure_meta.json
            ...
        page_02/
            ...

Usage:
    python process_pdf.py paper.pdf
    python process_pdf.py paper.pdf --output-dir ./my_results --model base
    python process_pdf.py paper.pdf --workers 6
"""

import argparse
import json
import re
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

MAX_RETRIES = 3   # maximum retry attempts per crop (not counting the initial attempt)

# Thread-safe print
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


# ── Quality-check helpers ─────────────────────────────────────────────────────

def _has_rgroup(smiles: str) -> bool:
    """Return True if *smiles* contains R-group or wildcard placeholders.

    Detects:
      *          — SMILES wildcard / attachment point
      [R], [R1], [R2], …  — explicit R-group atoms in bracket notation
    """
    if not smiles or not isinstance(smiles, str):
        return False
    if "*" in smiles:
        return True
    if re.search(r"\[R\d*\]", smiles):
        return True
    return False


def _is_template_reaction(rxn: dict) -> bool:
    """Heuristic: return True if *rxn* looks like an R-group template.

    A reaction is considered a template when either:
      • it carries an explicit marker  (``is_template``, ``type == "template"``), or
      • more than half of its reactant + product SMILES contain R-group placeholders
        (meaning the reaction is showing the *general scheme*, not a concrete instance).
    """
    if rxn.get("is_template") or rxn.get("type") in ("template", "general_scheme"):
        return True

    all_smiles = []
    for role in ("reactants", "products"):
        for item in rxn.get(role, []):
            s = item.get("smiles") or item.get("SMILES") or ""
            if s:
                all_smiles.append(s)

    if not all_smiles:
        return False

    rgroup_count = sum(1 for s in all_smiles if _has_rgroup(s))
    return rgroup_count > len(all_smiles) * 0.5


def _validate_smiles(smiles: str) -> bool:
    """Return True if *smiles* is chemically valid.

    Uses RDKit when available; falls back to True (permissive) when RDKit is
    absent to avoid false positives blocking the pipeline.
    """
    if not smiles or not isinstance(smiles, str):
        return True
    if smiles.strip().lower() in ("", "n/a", "none", "unknown", "null"):
        return True
    try:
        from rdkit import Chem  # noqa: PLC0415
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return True   # RDKit unavailable — treat as valid


def _check_result_issues(result: dict) -> list:
    """Inspect a ChemEagle result dict for the three quality criteria.

    Returns a list of short issue class strings; empty list means all checks
    passed.  The class strings map directly to the three HITL categories:

    ``"invalid_smiles"``        — any reactant or product SMILES is chemically
        invalid (RDKit check; permissive fallback when RDKit is unavailable).
    ``"rgroup_not_substituted"``— 2 or more components (reactants + products)
        in non-template reactions still carry * / [R…] placeholders.
    """
    if not result or not isinstance(result, dict):
        return ["invalid_smiles"]   # empty / unparseable result → suspect SMILES

    reactions = result.get("reactions", [])
    if not reactions:
        return []

    rgroup_count         = 0    # reactants + products with R-group placeholders
    invalid_smiles_found = False

    for rxn in reactions:
        if _is_template_reaction(rxn):
            continue            # skip general-scheme templates

        for role in ("reactants", "products"):
            for item in rxn.get(role, []):
                s = item.get("smiles") or item.get("SMILES") or ""
                if not s:
                    continue
                if _has_rgroup(s):
                    rgroup_count += 1
                elif not _validate_smiles(s):
                    invalid_smiles_found = True

    issues = []
    if invalid_smiles_found:
        issues.append("invalid_smiles")
    if rgroup_count >= 2:           # ≥ 2 components (reactants + products)
        issues.append("rgroup_not_substituted")
    return issues


# ── Per-crop ChemEagle worker (with retry + HITL) ────────────────────────────

def _run_crop(task: dict, chemeagle_fn) -> dict:
    """
    Execute ChemEagle on one cropped image with up to MAX_RETRIES retries.

    Retry is triggered when any of the following are detected:
      1. ChemEagle raised an exception.
      2. More than 2 reactants in non-template reactions still carry R groups.
      3. One or more invalid SMILES strings in the output.

    After MAX_RETRIES attempts any still-failing crop is flagged
    ``human_review_required: true`` and returned with ``hitl=True``.

    Parameters
    ----------
    task         : dict — page_num, crop_num, crop_stem, label, bbox,
                   crop_img_path, crop_result_path, crop_meta_path,
                   skipped, skip_reason, run_dir
    chemeagle_fn : callable — ChemEagle or ChemEagle_OS

    Returns
    -------
    dict with: page_num, crop_num, success, error, elapsed, hitl,
               hitl_reasons, n_rxn, result
    """
    page_num         = task["page_num"]
    crop_num         = task["crop_num"]
    crop_stem        = task["crop_stem"]
    label            = task["label"]
    bbox             = task["bbox"]
    crop_img_path    = Path(task["crop_img_path"])
    crop_result_path = Path(task["crop_result_path"])
    crop_meta_path   = Path(task["crop_meta_path"])

    # ── Skipped crops (e.g. tables) ──────────────────────────────────────────
    if task.get("skipped"):
        result       = {"skipped": True, "reason": task.get("skip_reason", "")}
        success      = None
        error        = None
        elapsed      = 0.0
        hitl         = False
        hitl_reasons = []
        _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  "
                f"[skipped — {task.get('skip_reason', 'table')}]")

    # ── Active crops ─────────────────────────────────────────────────────────
    else:
        result       = None
        error        = None
        success      = False
        hitl         = False
        hitl_reasons = []
        elapsed      = 0.0

        for attempt in range(MAX_RETRIES + 1):   # 0 = initial, 1-3 = retries
            attempt_tag = "initial" if attempt == 0 else f"retry {attempt}/{MAX_RETRIES}"
            t_start = time.perf_counter()

            try:
                result   = chemeagle_fn(str(crop_img_path))
                elapsed += time.perf_counter() - t_start

                issues = _check_result_issues(result)

                if not issues:
                    # ── All quality checks passed ─────────────────────────
                    success = True
                    n_rxn   = len((result or {}).get("reactions", []))
                    retry_note = f", after {attempt} retry/retries" if attempt > 0 else ""
                    _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  "
                            f"ok  ({elapsed:.1f}s, {n_rxn} rxn(s){retry_note})")
                    break

                else:
                    # ── Quality issue detected ────────────────────────────
                    if attempt < MAX_RETRIES:
                        _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  "
                                f"quality issue [{', '.join(issues)}] "
                                f"— {attempt_tag}, retrying …")
                    else:
                        # Max retries exhausted — flag for human review
                        success      = True   # pipeline ran; result exists but suspect
                        hitl         = True
                        hitl_reasons = issues
                        result.setdefault("human_review_required", True)
                        result["human_review_reasons"] = hitl_reasons
                        n_rxn = len((result or {}).get("reactions", []))
                        _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  "
                                f"HITL  ({elapsed:.1f}s, {n_rxn} rxn(s), "
                                f"issues: {', '.join(issues)})")
                        break

            except Exception as exc:
                elapsed += time.perf_counter() - t_start
                error    = str(exc)

                if attempt < MAX_RETRIES:
                    _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  "
                            f"error ({error[:70]}) — {attempt_tag}, retrying …")
                    error = None   # reset before next attempt
                else:
                    # Max retries exhausted — flag for human review
                    hitl         = True
                    hitl_reasons = ["processing_error"]
                    result       = {
                        "error":                 error,
                        "parsed":                False,
                        "human_review_required": True,
                        "human_review_reasons":  hitl_reasons,
                    }
                    _tprint(f"    [p{page_num}/c{crop_num}] {crop_stem}  "
                            f"FAILED+HITL  ({error[:70]})")
                    break

    # ── Write result JSON ─────────────────────────────────────────────────────
    crop_result_path.write_text(
        json.dumps(result or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Write per-crop metadata ───────────────────────────────────────────────
    meta = {
        "page":                  page_num,
        "crop":                  crop_num,
        "label":                 label,
        "bbox":                  [float(v) for v in bbox],
        "image_file":            crop_img_path.name,
        "result_file":           crop_result_path.name,
        "processing_s":          round(elapsed, 2),
        "success":               success,
        "error":                 error,
        "human_review_required": hitl,
        "human_review_reasons":  hitl_reasons,
    }
    crop_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "page_num":    page_num,
        "crop_num":    crop_num,
        "success":     success,
        "error":       error,
        "elapsed":     elapsed,
        "hitl":        hitl,
        "hitl_reasons": hitl_reasons,
        "n_rxn":       len((result or {}).get("reactions", [])) if success else 0,
        "result":      result,
        # preserve paths for summary merging
        "crop_img_path":    str(crop_img_path),
        "crop_result_path": str(crop_result_path),
        "label":            label,
        "bbox":             bbox,
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
    skip_tables : If True, skip crops labelled as "table".
    workers     : Number of parallel ChemEagle worker threads (default 4).

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
    from pdfmodel.methods import _pdf_to_image  # noqa: PLC0415
    page_images = _pdf_to_image(str(pdf_path))
    print(f"      {len(page_images)} page(s) loaded.")

    # ── Step 2: Load models ───────────────────────────────────────────────────
    print("\n[2/3] Loading models …")
    vh_model, vh_processor = _load_visualheist(model_size)

    if backend == "azure":
        from main import ChemEagle as _ce        # noqa: PLC0415
        chemeagle_fn = _ce
    else:
        from main import ChemEagle_OS as _ce_os  # noqa: PLC0415
        chemeagle_fn = _ce_os

    print(f"      ChemEagle backend : {backend}")
    print(f"      Parallel workers  : {workers}")
    print(f"      Max retries/crop  : {MAX_RETRIES}")

    # ── Step 3a: Detection — sequential (GPU) ─────────────────────────────────
    print("\n[3/3] Detecting figures …\n")

    summary: dict = {
        "pdf":           str(pdf_path),
        "timestamp":     datetime.now().isoformat(),
        "model_size":    model_size,
        "backend":       backend,
        "workers":       workers,
        "max_retries":   MAX_RETRIES,
        "n_pages":       len(page_images),
        "skip_tables":   skip_tables,
        "pages":         [],
    }

    all_tasks:    list = []
    page_entries: list = []

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

        page_entry: dict = {"page": page_num, "n_detected": len(bboxes), "crops": []}

        for crop_idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            crop_num   = crop_idx + 1
            label_slug = label.lower().replace(" ", "_")
            crop_stem  = f"crop_{crop_num:02d}_{label_slug}"

            crop_img_path    = page_dir / f"{crop_stem}.png"
            crop_result_path = page_dir / f"{crop_stem}_result.json"
            crop_meta_path   = page_dir / f"{crop_stem}_meta.json"

            x1, y1, x2, y2 = bbox
            page_img.crop((x1, y1, x2, y2)).save(str(crop_img_path))

            is_table = skip_tables and "table" in label.lower()
            all_tasks.append({
                "page_num":         page_num,
                "crop_num":         crop_num,
                "crop_stem":        crop_stem,
                "label":            label,
                "bbox":             bbox,
                "crop_img_path":    str(crop_img_path),
                "crop_result_path": str(crop_result_path),
                "crop_meta_path":   str(crop_meta_path),
                "skipped":          is_table,
                "skip_reason":      "table" if is_table else None,
                "run_dir":          str(run_dir),
            })

            page_entry["crops"].append({
                "crop":        crop_num,
                "label":       label,
                "bbox":        [float(v) for v in bbox],
                "image_file":  str(crop_img_path.relative_to(run_dir)),
                "result_file": str(crop_result_path.relative_to(run_dir)),
                "success":            None,   # filled after parallel phase
                "human_review_required": False,
            })

        page_entries.append(page_entry)

    total_crops = len(all_tasks)
    print(f"\n  {total_crops} crop(s) queued across {len(page_images)} page(s).")

    # ── Step 3b: ChemEagle — parallel ─────────────────────────────────────────
    print(f"\n  Running ChemEagle with {workers} worker(s) "
          f"(up to {MAX_RETRIES} retries per crop) …\n")

    total_success = 0
    total_skipped = 0
    total_failed  = 0
    total_hitl    = 0
    hitl_crops: list = []   # filled with summary entries for human-review crops

    results_index: dict = {}   # (page_num, crop_num) → info dict

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
                    "page_num":    task["page_num"],
                    "crop_num":    task["crop_num"],
                    "success":     False,
                    "error":       str(exc),
                    "elapsed":     0.0,
                    "hitl":        True,
                    "hitl_reasons": ["unhandled_future_exception"],
                    "n_rxn":       0,
                    "result":      None,
                    "crop_img_path":    task["crop_img_path"],
                    "crop_result_path": task["crop_result_path"],
                    "label":            task["label"],
                    "bbox":             task["bbox"],
                }

            key = (info["page_num"], info["crop_num"])
            results_index[key] = info

            if info["success"] is None:
                total_skipped += 1
            elif info.get("hitl"):
                total_hitl += 1
                hitl_crops.append({
                    "page":                  info["page_num"],
                    "crop":                  info["crop_num"],
                    "label":                 info.get("label", ""),
                    "image_file":            str(Path(info["crop_img_path"]).name),
                    "result_file":           str(Path(info["crop_result_path"]).name),
                    "human_review_reasons":  info.get("hitl_reasons", []),
                })
            elif info["success"]:
                total_success += 1
            else:
                total_failed += 1

    # ── Step 3c: Merge results back into page_entries ─────────────────────────
    for page_entry in page_entries:
        for crop_slot in page_entry["crops"]:
            key  = (page_entry["page"], crop_slot["crop"])
            info = results_index.get(key)
            if info is not None:
                crop_slot["success"]               = info["success"]
                crop_slot["human_review_required"] = info.get("hitl", False)

    # ── Write HITL report ─────────────────────────────────────────────────────
    (run_dir / "hitl_crops.json").write_text(
        json.dumps(hitl_crops, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Write top-level summary ───────────────────────────────────────────────
    summary["pages"] = page_entries
    summary.update({
        "n_crops_total":          total_crops,
        "n_success":              total_success,
        "n_skipped":              total_skipped,
        "n_failed":               total_failed,
        "n_human_review":         total_hitl,
        "human_review_crop_list": hitl_crops,
        "completed_at":           datetime.now().isoformat(),
    })
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PDF pipeline complete")
    print(f"  PDF            : {pdf_path.name}")
    print(f"  Pages          : {len(page_images)}")
    print(f"  Crops total    : {total_crops}")
    print(f"    ✓ success    : {total_success}")
    print(f"    ✗ failed     : {total_failed}")
    print(f"    ⚠ HITL       : {total_hitl}  (human review required)")
    print(f"    — skipped    : {total_skipped}")
    if hitl_crops:
        print(f"\n  Crops flagged for human review:")
        for h in hitl_crops:
            print(f"    page {h['page']} / crop {h['crop']}  "
                  f"({h['label']})  reasons: {', '.join(h['human_review_reasons'])}")
    print(f"\n  Workers  : {workers}   Max retries : {MAX_RETRIES}")
    print(f"  Output   : {run_dir}")
    print(f"{'='*60}\n")

    return run_dir


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract figures from a PDF and run ChemEagle on each one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
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
        help="Skip crops whose label contains 'table'.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel ChemEagle worker threads.",
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
