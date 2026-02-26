"""
ChemEagle Streamlit UI
======================
Upload a chemical reaction image, run the extraction pipeline, then
validate every returned SMILES with RDKit and display 2-D structure
drawings for valid ones.

Run with:
    streamlit run streamlit_app.py
"""

import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="ChemEagle Analyzer",
    layout="wide",
    page_icon="🧪",
)


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ChemEagle models (first run only)…")
def _load_pipeline():
    """Import and return the ChemEagle function.

    Using cache_resource keeps the heavy ML models in memory across
    Streamlit re-runs, so they are only loaded once.
    """
    from main import ChemEagle  # noqa: PLC0415
    return ChemEagle


@st.cache_data(show_spinner=False)
def _validate_smiles(smiles: str) -> tuple[bool, bytes | None]:
    """Validate a SMILES string with RDKit.

    Returns
    -------
    (is_valid, png_bytes)
        is_valid  – True if RDKit can parse the SMILES
        png_bytes – PNG bytes of the 2-D structure, or None if invalid
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None

        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(320, 240))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return True, buf.getvalue()
    except Exception:
        return False, None


# ── SMILES extraction ─────────────────────────────────────────────────────────

def _extract_smiles(data: dict) -> list[dict]:
    """Walk the pipeline output and collect every unique SMILES.

    Handles the three known output shapes:
    * ``original_molecule_list``  – keys are SMILES strings
    * ``molecule_coref``          – list of ``{smiles, texts, …}``
    * ``reactions``               – nested reactants / products
    Plus a generic recursive sweep for any other ``smiles`` key.

    Returns a list of dicts: ``{smiles, label, context}``.
    """
    seen: set[str] = set()
    out: list[dict] = []

    def _add(smiles: str, label: str = "", context: str = "") -> None:
        smiles = (smiles or "").strip()
        if smiles and smiles not in seen:
            seen.add(smiles)
            out.append({"smiles": smiles, "label": str(label), "context": context})

    # original_molecule_list  {smiles: [label, note, role, …]}
    for smiles, info in data.get("original_molecule_list", {}).items():
        label = info[0] if isinstance(info, list) and info else ""
        role  = info[2] if isinstance(info, list) and len(info) > 2 else ""
        _add(smiles, label, f"molecule list · {role}" if role else "molecule list")

    # molecule_coref  [{smiles, texts, bbox_id}, …]
    for item in data.get("molecule_coref", []):
        texts = item.get("texts", [])
        _add(item.get("smiles", ""), texts[0] if texts else "", "molecule_coref")

    # reactions  [{reaction_id, reactants:[{smiles,…}], products:[…]}, …]
    for rxn in data.get("reactions", []):
        rid = rxn.get("reaction_id", "")
        note = rxn.get("note", "")
        tag = f"rxn {rid}" + (f" · {note}" if note else "")
        for r in rxn.get("reactants", []):
            _add(r.get("smiles", ""), r.get("label", "reactant"), f"{tag} · reactant")
        for p in rxn.get("products", []):
            _add(p.get("smiles", ""), p.get("label", "product"), f"{tag} · product")

    # Generic recursive sweep catches any remaining {smiles: …} keys
    def _sweep(obj: object, ctx: str = "") -> None:
        if isinstance(obj, dict):
            s = obj.get("smiles") or obj.get("SMILES")
            if s:
                _add(str(s), str(obj.get("label", "")), ctx)
            for k, v in obj.items():
                if k not in ("smiles", "SMILES"):
                    _sweep(v, ctx or k)
        elif isinstance(obj, list):
            for item in obj:
                _sweep(item, ctx)

    _sweep(data)
    return out


# ── Render a single molecule card ─────────────────────────────────────────────

def _render_molecule_card(mol: dict) -> None:
    """Render one SMILES entry as a Streamlit column card."""
    smiles    = mol["smiles"]
    label     = mol["label"]
    context   = mol["context"]

    is_valid, png = _validate_smiles(smiles)

    badge = "🟢 **Valid SMILES**" if is_valid else "🔴 **Invalid SMILES**"
    st.markdown(badge)

    if label:
        st.markdown(f"**Label:** `{label}`")
    if context:
        st.markdown(
            f"<span style='font-size:0.8em;color:#888'>{context}</span>",
            unsafe_allow_html=True,
        )

    st.code(smiles, language=None)

    if is_valid and png:
        st.image(png, use_container_width=True)
    else:
        st.caption("_(structure unavailable)_")


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🧪 ChemEagle — Chemical Reaction Analyzer")
    st.markdown(
        "Upload a reaction image to extract molecules, "
        "validate SMILES with **RDKit**, and view 2-D structure drawings."
    )

    # ── Sidebar options ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        cols_per_row = st.slider("Molecules per row", 1, 5, 3)
        show_invalid = st.checkbox("Show invalid SMILES", value=True)
        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "ChemEagle extracts structured reaction data from chemical "
            "scheme images using a multi-agent vision pipeline."
        )

    # ── File uploader ─────────────────────────────────────────────────────
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

    if not (uploaded and analyze):
        return

    # ── Run pipeline ──────────────────────────────────────────────────────
    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    result: dict | None = None
    try:
        pipeline = _load_pipeline()
        with st.spinner("Running ChemEagle extraction pipeline…"):
            result = pipeline(tmp_path)
    except Exception as exc:
        st.error(f"**Pipeline error:** {exc}")
        return
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Handle fallback / error output ────────────────────────────────────
    if result is None:
        st.error("Pipeline returned no result.")
        return

    if result.get("parsed") is False:
        st.warning("Pipeline could not parse a structured JSON output. Raw response below:")
        st.text(result.get("content", ""))
        return

    # ── Raw JSON ──────────────────────────────────────────────────────────
    with st.expander("📄 Raw pipeline output (JSON)", expanded=False):
        st.json(result)

    # ── Reactions table (if present) ──────────────────────────────────────
    if result.get("reactions"):
        with st.expander(f"⚗️ Reactions ({len(result['reactions'])} found)", expanded=True):
            for rxn in result["reactions"]:
                rid  = rxn.get("reaction_id", "?")
                note = rxn.get("note", "")
                header = f"**Reaction {rid}**" + (f" — {note}" if note else "")
                st.markdown(header)

                r_smiles = [r.get("smiles", "") for r in rxn.get("reactants", [])]
                p_smiles = [p.get("smiles", "") for p in rxn.get("products", [])]
                conds    = rxn.get("conditions", [])

                c1, c2, c3 = st.columns(3)
                c1.markdown("**Reactants**")
                for s in r_smiles:
                    c1.code(s, language=None)
                c2.markdown("**Products**")
                for s in p_smiles:
                    label = next(
                        (p.get("label", "") for p in rxn.get("products", []) if p.get("smiles") == s),
                        "",
                    )
                    c2.code(f"{s}  # {label}" if label else s, language=None)
                c3.markdown("**Conditions**")
                for cond in conds:
                    c3.markdown(f"- {cond}")
                st.divider()

    # ── SMILES validation grid ─────────────────────────────────────────────
    molecules = _extract_smiles(result)

    if not molecules:
        st.warning("No SMILES found in the pipeline output.")
        return

    # Pre-validate all to build summary metrics before rendering
    validity = {m["smiles"]: _validate_smiles(m["smiles"])[0] for m in molecules}
    valid_n   = sum(1 for v in validity.values() if v)
    invalid_n = len(molecules) - valid_n

    # Summary metrics at the top
    m1, m2, m3 = st.columns(3)
    m1.metric("Total molecules", len(molecules))
    m2.metric("Valid ✓", valid_n)
    m3.metric("Invalid ✗", invalid_n)

    st.subheader("Molecule Validation")

    display = molecules if show_invalid else [m for m in molecules if validity[m["smiles"]]]

    if not display:
        st.info("All extracted SMILES are invalid (enable 'Show invalid SMILES' in sidebar to view them).")
        return

    for i in range(0, len(display), cols_per_row):
        batch = display[i : i + cols_per_row]
        cols  = st.columns(cols_per_row)
        for col, mol in zip(cols, batch):
            with col:
                with st.container(border=True):
                    _render_molecule_card(mol)


if __name__ == "__main__":
    main()
