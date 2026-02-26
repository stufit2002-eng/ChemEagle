"""
ChemEagle Streamlit UI
======================
Upload a chemical reaction image, run the extraction pipeline, then
display each reaction visually: 2-D RDKit structure drawings for every
reactant and product, with reaction conditions shown in the middle.

Results are persisted to ./chemeagle_history/ so you can reload any
previous run without re-running the pipeline.

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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChemEagle Analyzer",
    layout="wide",
    page_icon="🧪",
)

# ── History directory ─────────────────────────────────────────────────────────
HISTORY_DIR = Path("./chemeagle_history")
HISTORY_DIR.mkdir(exist_ok=True)


# ── History helpers ───────────────────────────────────────────────────────────

def _history_save(image_bytes: bytes, image_suffix: str,
                  original_name: str, result: dict) -> Path:
    """Persist one pipeline run to disk and return its folder path."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = Path(original_name).stem[:30].replace(" ", "_")
    run_dir = HISTORY_DIR / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    img_path = run_dir / f"image{image_suffix}"
    img_path.write_bytes(image_bytes)

    # Save pipeline output
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save metadata
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
    """Return all saved runs sorted newest-first.

    Each entry: {run_id, ts_label, original_filename, n_reactions, run_dir}
    """
    runs = []
    for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_file = d / "meta.json"
        result_file = d / "result.json"
        if not meta_file.exists() or not result_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
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
    """Load image bytes and result dict from a history folder."""
    # Find the image file
    image_bytes = None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        candidate = run_dir / f"image{ext}"
        if candidate.exists():
            image_bytes = candidate.read_bytes()
            break
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return image_bytes, result


def _history_delete(run_dir: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)


# ── Cached ML helpers ─────────────────────────────────────────────────────────

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


# ── Molecule / reaction renderers ─────────────────────────────────────────────

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
    out: list[dict] = []
    seen: set[str]  = set()

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


# ── Result display (shared by new runs and loaded history) ────────────────────

def _display_result(result: dict, image_bytes: bytes | None,
                    show_smiles_code: bool, show_invalid: bool) -> None:
    """Render pipeline output: reactions + extra molecules."""

    if image_bytes:
        st.image(image_bytes, use_container_width=False, width=600)

    with st.expander("📄 Raw pipeline output (JSON)", expanded=False):
        st.json(result)

    reactions = result.get("reactions", [])
    reaction_smiles: set[str] = set()

    if reactions:
        st.subheader(f"⚗️ Reactions — {len(reactions)} found")
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
        st.info("No reactions found in the pipeline output.")

    extras = _extra_molecules(result, reaction_smiles)
    if extras:
        with st.expander(
            f"🔬 Other extracted molecules ({len(extras)})",
            expanded=not bool(reactions),
        ):
            validity = {m["smiles"]: _validate_smiles(m["smiles"])[0] for m in extras}
            valid_n   = sum(1 for v in validity.values() if v)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(extras))
            c2.metric("Valid ✓", valid_n)
            c3.metric("Invalid ✗", len(extras) - valid_n)

            display = extras if show_invalid else [
                m for m in extras if validity[m["smiles"]]
            ]
            if not display:
                st.info("All extra SMILES are invalid "
                        "(enable 'Show invalid SMILES' in sidebar).")
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


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    # ── session state init ────────────────────────────────────────────────
    if "active_result"    not in st.session_state:
        st.session_state.active_result    = None   # dict  – current result
    if "active_image"     not in st.session_state:
        st.session_state.active_image     = None   # bytes – current image
    if "active_run_label" not in st.session_state:
        st.session_state.active_run_label = ""     # str   – label shown in header

    st.title("🧪 ChemEagle — Chemical Reaction Analyzer")
    st.markdown(
        "Upload a reaction image to extract molecules, "
        "validate SMILES with **RDKit**, and view 2-D structure drawings. "
        "All results are saved locally and browsable from the sidebar."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        show_smiles_code = st.checkbox("Show SMILES strings",              value=True)
        show_invalid     = st.checkbox("Show invalid SMILES in extras",    value=True)

        st.markdown("---")
        st.header("🕑 History")

        runs = _history_list()
        if not runs:
            st.caption("No previous runs yet.")
        else:
            st.caption(f"{len(runs)} run(s) saved locally.")
            for run in runs:
                n_rxn = run["n_reactions"]
                label = (
                    f"**{run['ts_label']}**  \n"
                    f"`{run['original_filename']}`  \n"
                    f"{n_rxn} reaction{'s' if n_rxn != 1 else ''}"
                )
                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    if st.button(label, key=f"load_{run['run_id']}",
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
                        # Clear active if this was it
                        if st.session_state.active_run_label.startswith(
                            run["ts_label"]
                        ):
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

    # ── Upload + run panel ────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop a reaction image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=False, width=600)

    analyze = st.button(
        "▶  Analyze Image",
        type="primary",
        disabled=not uploaded,
        use_container_width=False,
    )

    if uploaded and analyze:
        suffix       = Path(uploaded.name).suffix or ".png"
        image_bytes  = uploaded.getvalue()

        # Write temp file for pipeline
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result: dict | None = None
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
            st.warning("Pipeline could not parse a structured JSON output. "
                       "Raw response below:")
            st.text(result.get("content", ""))
            return

        # ── Save to history ───────────────────────────────────────────────
        run_dir = _history_save(image_bytes, suffix, uploaded.name, result)
        st.success(f"Result saved to `{run_dir}`")

        # ── Update session state ──────────────────────────────────────────
        ts_label = datetime.now().strftime("%b %d, %Y  %H:%M:%S")
        st.session_state.active_result    = result
        st.session_state.active_image     = image_bytes
        st.session_state.active_run_label = f"{ts_label} — {uploaded.name}"
        st.rerun()

    # ── Display active result (new run OR loaded from history) ────────────
    if st.session_state.active_result is not None:
        if st.session_state.active_run_label:
            st.caption(f"Showing: **{st.session_state.active_run_label}**")
        _display_result(
            st.session_state.active_result,
            st.session_state.active_image,
            show_smiles_code,
            show_invalid,
        )


if __name__ == "__main__":
    main()
