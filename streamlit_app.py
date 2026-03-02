"""
ChemEagle Streamlit UI
======================
Two tabs:
  Tab 1 – Image Analysis : upload a single reaction image, run ChemEagle,
           browse history of past single-image runs.
  Tab 2 – PDF Results    : browse structured output produced by process_pdf.py,
           page-by-page and crop-by-crop, with the same visual renderers.

Run with:
    streamlit run streamlit_app.py
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChemEagle Analyzer",
    layout="wide",
    page_icon="🧪",
)

# ── Persistent storage roots ──────────────────────────────────────────────────
HISTORY_DIR     = Path("./chemeagle_history")
PDF_RESULTS_DIR = Path("./pdf_results")
HISTORY_DIR.mkdir(exist_ok=True)
PDF_RESULTS_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Single-image history helpers
# ═════════════════════════════════════════════════════════════════════════════

def _history_save(image_bytes: bytes, image_suffix: str,
                  original_name: str, result: dict) -> Path:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug    = Path(original_name).stem[:30].replace(" ", "_")
    run_dir = HISTORY_DIR / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"image{image_suffix}").write_bytes(image_bytes)
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "timestamp":         datetime.now().isoformat(),
        "original_filename": original_name,
        "n_reactions":       len(result.get("reactions", [])),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def _history_list() -> list[dict]:
    runs = []
    for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if not (d / "meta.json").exists() or not (d / "result.json").exists():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        ts_raw = meta.get("timestamp", "")
        try:
            ts_label = datetime.fromisoformat(ts_raw).strftime("%b %d, %Y  %H:%M:%S")
        except Exception:
            ts_label = ts_raw
        runs.append({
            "run_id":            d.name,
            "ts_label":          ts_label,
            "original_filename": meta.get("original_filename", "unknown"),
            "n_reactions":       meta.get("n_reactions", 0),
            "run_dir":           d,
        })
    return runs


def _history_load(run_dir: Path) -> tuple[bytes | None, dict]:
    image_bytes = None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        p = run_dir / f"image{ext}"
        if p.exists():
            image_bytes = p.read_bytes()
            break
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return image_bytes, result


def _history_delete(run_dir: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PDF run helpers
# ═════════════════════════════════════════════════════════════════════════════

def _pdf_runs_list() -> list[dict]:
    """Return all PDF pipeline runs from ./pdf_results/, newest-first."""
    runs = []
    for d in sorted(PDF_RESULTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        summary_file = d / "summary.json"
        if not summary_file.exists():
            continue
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts_raw = summary.get("timestamp", "")
        try:
            ts_label = datetime.fromisoformat(ts_raw).strftime("%b %d, %Y  %H:%M:%S")
        except Exception:
            ts_label = ts_raw
        pdf_name = Path(summary.get("pdf", d.name)).name
        n_crops  = summary.get("n_crops_total", 0)
        runs.append({
            "run_id":    d.name,
            "ts_label":  ts_label,
            "pdf_name":  pdf_name,
            "summary":   summary,
            "run_dir":   d,
            "label":     f"{ts_label}  ·  {pdf_name}  ({n_crops} crops)",
        })
    return runs


# ═════════════════════════════════════════════════════════════════════════════
#  Cached ML helpers
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ChemEagle models (first run only)…")
def _load_pipeline():
    from main import ChemEagle  # noqa: PLC0415
    return ChemEagle


@st.cache_data(show_spinner=False)
def _validate_smiles(smiles: str) -> tuple[bool, bytes | None]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(300, 220))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return True, buf.getvalue()
    except Exception:
        return False, None


# ═════════════════════════════════════════════════════════════════════════════
#  Shared molecule / reaction renderers
# ═════════════════════════════════════════════════════════════════════════════

def _render_molecule(smiles: str, label: str = "", show_smiles: bool = True) -> None:
    is_valid, png = _validate_smiles(smiles)
    if label:
        st.markdown(f"**{label}**")
    if is_valid and png:
        st.image(png, use_container_width=True)
        st.caption("🟢 Valid SMILES")
    else:
        st.caption("🔴 Invalid SMILES")
        st.caption("_(structure unavailable)_")
    if show_smiles:
        st.code(smiles, language=None)


def _render_reaction_with_flag(rxn: dict, show_smiles_code: bool) -> None:
    rid    = rxn.get("reaction_id", "?")
    note   = rxn.get("note", "")
    header = f"**Reaction {rid}**" + (f" — {note}" if note else "")
    st.markdown(header)

    reactants  = rxn.get("reactants", [])
    products   = rxn.get("products",  [])
    conditions = rxn.get("conditions", [])

    n_r = max(len(reactants), 1)
    n_p = max(len(products),  1)
    col_weights = [2] * n_r + [1] + [2] * n_p
    cols = st.columns(col_weights)

    for i, r in enumerate(reactants):
        with cols[i]:
            with st.container(border=True):
                _render_molecule(r.get("smiles", ""),
                                 r.get("label", f"Reactant {i + 1}"),
                                 show_smiles=show_smiles_code)
    if not reactants:
        with cols[0]:
            st.caption("_(no reactants)_")

    with cols[n_r]:
        st.markdown(
            "<div style='text-align:center;font-size:2rem;padding-top:60px;'>⟶</div>",
            unsafe_allow_html=True,
        )
        if conditions:
            st.markdown(
                "<div style='text-align:center;font-size:0.85rem;color:#555;'>"
                + "<br>".join(f"• {c}" for c in conditions)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center;font-size:0.8rem;color:#aaa;'>no conditions</div>",
                unsafe_allow_html=True,
            )

    for j, p in enumerate(products):
        with cols[n_r + 1 + j]:
            with st.container(border=True):
                _render_molecule(p.get("smiles", ""),
                                 p.get("label", f"Product {j + 1}"),
                                 show_smiles=show_smiles_code)
    if not products:
        with cols[n_r + 1]:
            st.caption("_(no products)_")


def _extra_molecules(data: dict, reaction_smiles: set[str]) -> list[dict]:
    out:  list[dict] = []
    seen: set[str]   = set()

    def _add(smiles: str, label: str, context: str) -> None:
        smiles = (smiles or "").strip()
        if smiles and smiles not in seen and smiles not in reaction_smiles:
            seen.add(smiles)
            out.append({"smiles": smiles, "label": label, "context": context})

    for smiles, info in data.get("original_molecule_list", {}).items():
        label = info[0] if isinstance(info, list) and info else ""
        role  = info[2] if isinstance(info, list) and len(info) > 2 else ""
        _add(smiles, label, f"molecule list · {role}" if role else "molecule list")

    for item in data.get("molecule_coref", []):
        texts = item.get("texts", [])
        _add(item.get("smiles", ""), texts[0] if texts else "", "molecule_coref")

    return out


def _display_chemeagle_result(result: dict, show_smiles_code: bool,
                               show_invalid: bool) -> None:
    """Render a ChemEagle result dict (reactions + extra molecules).

    Does NOT render the source image — callers handle that themselves.
    """
    if result.get("parsed") is False:
        st.warning("Pipeline could not parse structured JSON.")
        st.text(result.get("content", ""))
        return
    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
        return
    if result.get("skipped"):
        st.info(f"Skipped — {result.get('reason', 'no reason given')}")
        return

    with st.expander("📄 Raw JSON", expanded=False):
        st.json(result)

    reactions = result.get("reactions", [])
    reaction_smiles: set[str] = set()

    if reactions:
        st.markdown(f"**⚗️ {len(reactions)} reaction(s) found**")
        for rxn in reactions:
            for r in rxn.get("reactants", []):
                reaction_smiles.add((r.get("smiles") or "").strip())
            for p in rxn.get("products", []):
                reaction_smiles.add((p.get("smiles") or "").strip())
        reaction_smiles.discard("")
        for rxn in reactions:
            _render_reaction_with_flag(rxn, show_smiles_code)
            st.divider()
    else:
        st.info("No reactions found in ChemEagle output.")

    extras = _extra_molecules(result, reaction_smiles)
    if extras:
        with st.expander(f"🔬 Other extracted molecules ({len(extras)})",
                         expanded=not bool(reactions)):
            validity  = {m["smiles"]: _validate_smiles(m["smiles"])[0] for m in extras}
            valid_n   = sum(1 for v in validity.values() if v)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(extras))
            c2.metric("Valid ✓", valid_n)
            c3.metric("Invalid ✗", len(extras) - valid_n)

            display = extras if show_invalid else [
                m for m in extras if validity[m["smiles"]]
            ]
            if not display:
                st.info("All extra SMILES are invalid.")
            else:
                for i in range(0, len(display), 3):
                    batch = display[i : i + 3]
                    cols  = st.columns(3)
                    for col, mol in zip(cols, batch):
                        with col:
                            with st.container(border=True):
                                smiles   = mol["smiles"]
                                is_valid, png = _validate_smiles(smiles)
                                st.markdown("🟢 **Valid**" if is_valid else "🔴 **Invalid**")
                                if mol["label"]:
                                    st.markdown(f"**Label:** `{mol['label']}`")
                                if mol["context"]:
                                    st.markdown(
                                        f"<span style='font-size:0.8em;color:#888'>"
                                        f"{mol['context']}</span>",
                                        unsafe_allow_html=True,
                                    )
                                if show_smiles_code:
                                    st.code(smiles, language=None)
                                if is_valid and png:
                                    st.image(png, use_container_width=True)
                                else:
                                    st.caption("_(structure unavailable)_")


def _display_result(result: dict, image_bytes: bytes | None,
                    show_smiles_code: bool, show_invalid: bool) -> None:
    """Tab-1 wrapper: optionally shows the source image, then delegates."""
    if image_bytes:
        st.image(image_bytes, use_container_width=False, width=600)
    _display_chemeagle_result(result, show_smiles_code, show_invalid)


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 1 — single-image analysis
# ═════════════════════════════════════════════════════════════════════════════

def _tab_image(show_smiles_code: bool, show_invalid: bool) -> None:
    st.markdown(
        "Upload a reaction image to extract molecules, validate SMILES with "
        "**RDKit**, and view 2-D structure drawings. "
        "All results are saved locally and browsable from the sidebar."
    )

    uploaded = st.file_uploader(
        "Drop a reaction image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )
    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=False, width=600)

    if st.button("▶  Analyze Image", type="primary", disabled=not uploaded):
        suffix      = Path(uploaded.name).suffix or ".png"
        image_bytes = uploaded.getvalue()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = None
        try:
            pipeline = _load_pipeline()
            with st.spinner("Running ChemEagle extraction pipeline (CUDA)…"):
                result = pipeline(tmp_path)
        except Exception as exc:
            st.error(f"**Pipeline error:** {exc}")
            return
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if result is None:
            st.error("Pipeline returned no result.")
            return
        if result.get("parsed") is False:
            st.warning("Pipeline could not parse a structured JSON output.")
            st.text(result.get("content", ""))
            return

        run_dir = _history_save(image_bytes, suffix, uploaded.name, result)
        st.success(f"Result saved to `{run_dir}`")

        ts_label = datetime.now().strftime("%b %d, %Y  %H:%M:%S")
        st.session_state.active_result    = result
        st.session_state.active_image     = image_bytes
        st.session_state.active_run_label = f"{ts_label} — {uploaded.name}"
        st.rerun()

    if st.session_state.get("active_result") is not None:
        label = st.session_state.get("active_run_label", "")
        if label:
            st.caption(f"Showing: **{label}**")
        _display_result(
            st.session_state.active_result,
            st.session_state.active_image,
            show_smiles_code,
            show_invalid,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 2 — PDF pipeline results browser
# ═════════════════════════════════════════════════════════════════════════════

def _tab_pdf(show_smiles_code: bool, show_invalid: bool) -> None:
    st.markdown(
        "Browse structured output produced by **`process_pdf.py`**. "
        "Select a run, then drill down page-by-page and crop-by-crop."
    )

    runs = _pdf_runs_list()
    if not runs:
        st.info(
            "No PDF results found yet.  \n"
            "Run the pipeline first:  \n"
            "```\npython process_pdf.py your_paper.pdf\n```"
        )
        return

    # ── Run selector ─────────────────────────────────────────────────────────
    run_labels = [r["label"] for r in runs]
    selected_idx = st.selectbox(
        "PDF run",
        range(len(runs)),
        format_func=lambda i: run_labels[i],
        key="pdf_run_selector",
    )
    selected = runs[selected_idx]
    summary  = selected["summary"]
    run_dir  = selected["run_dir"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pages",    summary.get("n_pages",        "?"))
    c2.metric("Crops",    summary.get("n_crops_total",  "?"))
    c3.metric("✅ Success", summary.get("n_success",    "?"))
    c4.metric("❌ Failed",  summary.get("n_failed",     "?"))
    c5.metric("⏭ Skipped", summary.get("n_skipped",    "?"))
    completed = summary.get("completed_at", "")
    try:
        completed = datetime.fromisoformat(completed).strftime("%b %d, %Y  %H:%M:%S")
    except Exception:
        pass
    c6.metric("Completed", completed or "—")

    # ── Page-level navigation ─────────────────────────────────────────────────
    pages = summary.get("pages", [])
    if not pages:
        st.info("No pages recorded in summary.")
        return

    st.markdown("---")

    # Offer a page filter when there are many pages
    all_page_nums = [p["page"] for p in pages]
    if len(pages) > 5:
        selected_pages = st.multiselect(
            "Filter pages (leave empty to show all)",
            options=all_page_nums,
            default=[],
            key="pdf_page_filter",
        )
        pages_to_show = [p for p in pages if p["page"] in selected_pages] \
                        if selected_pages else pages
    else:
        pages_to_show = pages

    auto_expand = len(pages_to_show) <= 3

    for page_entry in pages_to_show:
        page_num   = page_entry["page"]
        crops      = page_entry.get("crops", [])
        n_detected = page_entry.get("n_detected", len(crops))
        n_ok       = sum(1 for c in crops if c.get("success") is True)
        n_fail     = sum(1 for c in crops if c.get("success") is False)
        n_skip     = sum(1 for c in crops if c.get("success") is None)

        expander_label = (
            f"📄 Page {page_num}  —  {n_detected} detected"
            + (f"  ·  ✅ {n_ok}" if n_ok else "")
            + (f"  ·  ❌ {n_fail}" if n_fail else "")
            + (f"  ·  ⏭ {n_skip}" if n_skip else "")
        )

        with st.expander(expander_label, expanded=auto_expand):
            if not crops:
                st.caption("Nothing detected on this page.")
                continue

            for crop_entry in crops:
                crop_num   = crop_entry.get("crop",   "?")
                label      = crop_entry.get("label",  "unknown")
                success    = crop_entry.get("success")
                image_rel  = crop_entry.get("image_file",  "")
                result_rel = crop_entry.get("result_file", "")
                bbox       = crop_entry.get("bbox",   [])

                # Derive the meta file path from the result path
                meta_rel = result_rel.replace("_result.json", "_meta.json")

                status_icon = "✅" if success is True else ("⏭" if success is None else "❌")
                crop_header = f"{status_icon} **Crop {crop_num} — {label}**"

                # Load meta for timing info
                processing_s = None
                meta_path = run_dir / meta_rel
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        processing_s = meta.get("processing_s")
                    except Exception:
                        pass

                if processing_s is not None:
                    crop_header += (
                        f"  <span style='font-size:0.8em;color:#888;'>"
                        f"({processing_s}s)</span>"
                    )
                if bbox:
                    bx = [round(v) for v in bbox]
                    crop_header += (
                        f"  <span style='font-size:0.75em;color:#aaa;'>"
                        f"bbox {bx}</span>"
                    )

                with st.container(border=True):
                    st.markdown(crop_header, unsafe_allow_html=True)

                    crop_img_path    = run_dir / image_rel
                    crop_result_path = run_dir / result_rel

                    # Two-column layout: narrow left (crop image), wide right (result)
                    col_img, col_result = st.columns([1, 3])

                    with col_img:
                        st.markdown("**Extracted crop**")
                        if crop_img_path.exists():
                            st.image(str(crop_img_path), use_container_width=True)
                        else:
                            st.caption("_(image file not found)_")

                    with col_result:
                        if not crop_result_path.exists():
                            st.caption("_(result file not found)_")
                        else:
                            try:
                                result = json.loads(
                                    crop_result_path.read_text(encoding="utf-8")
                                )
                            except Exception as e:
                                st.error(f"Could not read result JSON: {e}")
                                continue
                            _display_chemeagle_result(
                                result, show_smiles_code, show_invalid
                            )


# ═════════════════════════════════════════════════════════════════════════════
#  Main UI — sidebar + tabs
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Session state ─────────────────────────────────────────────────────────
    if "active_result"    not in st.session_state:
        st.session_state.active_result    = None
    if "active_image"     not in st.session_state:
        st.session_state.active_image     = None
    if "active_run_label" not in st.session_state:
        st.session_state.active_run_label = ""

    st.title("🧪 ChemEagle Analyzer")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        show_smiles_code = st.checkbox("Show SMILES strings",           value=True)
        show_invalid     = st.checkbox("Show invalid SMILES in extras", value=True)

        st.markdown("---")
        st.header("🕑 Image History")
        runs = _history_list()
        if not runs:
            st.caption("No previous single-image runs yet.")
        else:
            st.caption(f"{len(runs)} run(s) saved.")
            for run in runs:
                n_rxn  = run["n_reactions"]
                btn_label = (
                    f"**{run['ts_label']}**  \n"
                    f"`{run['original_filename']}`  \n"
                    f"{n_rxn} reaction{'s' if n_rxn != 1 else ''}"
                )
                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    if st.button(btn_label, key=f"load_{run['run_id']}",
                                 use_container_width=True):
                        img_bytes, result = _history_load(run["run_dir"])
                        st.session_state.active_result    = result
                        st.session_state.active_image     = img_bytes
                        st.session_state.active_run_label = (
                            f"{run['ts_label']} — {run['original_filename']}"
                        )
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{run['run_id']}",
                                 help="Delete this run"):
                        _history_delete(run["run_dir"])
                        if st.session_state.active_run_label.startswith(run["ts_label"]):
                            st.session_state.active_result    = None
                            st.session_state.active_image     = None
                            st.session_state.active_run_label = ""
                        st.rerun()

        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "ChemEagle extracts structured reaction data from chemical "
            "scheme images using a multi-agent vision pipeline."
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_img, tab_pdf = st.tabs(["🖼 Image Analysis", "📄 PDF Results"])

    with tab_img:
        _tab_image(show_smiles_code, show_invalid)

    with tab_pdf:
        _tab_pdf(show_smiles_code, show_invalid)


if __name__ == "__main__":
    main()
