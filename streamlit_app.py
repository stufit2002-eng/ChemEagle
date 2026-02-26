"""
ChemEagle Streamlit UI
======================
Upload a chemical reaction image, run the extraction pipeline, then
display each reaction visually: 2-D RDKit structure drawings for every
reactant and product, with reaction conditions shown in the middle.

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
        img = Draw.MolToImage(mol, size=(300, 220))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return True, buf.getvalue()
    except Exception:
        return False, None


# ── Render a single molecule (image or fallback text) ─────────────────────────

def _render_molecule(smiles: str, label: str = "", show_smiles: bool = True) -> None:
    """Render one molecule as a 2-D image with label and validity badge.

    If RDKit cannot parse the SMILES the raw string is shown instead.
    """
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


# ── Render one reaction as a visual row ───────────────────────────────────────

def _render_reaction(rxn: dict, mol_img_width: int = 200) -> None:
    """Display a single reaction: reactants → [conditions] → products.

    Each molecule is drawn as a 2-D RDKit image inside its own column.
    Conditions are shown in a centre column between reactants and products.
    """
    rid   = rxn.get("reaction_id", "?")
    note  = rxn.get("note", "")
    header = f"**Reaction {rid}**" + (f" — {note}" if note else "")
    st.markdown(header)

    reactants  = rxn.get("reactants", [])
    products   = rxn.get("products",  [])
    conditions = rxn.get("conditions", [])

    # ── Build column layout: [reactants…] [arrow+conditions] [products…] ──────
    n_r = max(len(reactants), 1)
    n_p = max(len(products),  1)

    # weights: each reactant/product gets weight 2, arrow column weight 1
    col_weights = [2] * n_r + [1] + [2] * n_p
    cols = st.columns(col_weights)

    # ── Reactants ────────────────────────────────────────────────────────────
    for i, r in enumerate(reactants):
        smiles = r.get("smiles", "")
        label  = r.get("label", f"Reactant {i + 1}")
        with cols[i]:
            with st.container(border=True):
                _render_molecule(smiles, label)

    # If no reactants, leave the first column(s) empty (already have 1 col)
    if not reactants:
        with cols[0]:
            st.caption("_(no reactants)_")

    # ── Arrow + conditions ───────────────────────────────────────────────────
    arrow_col = cols[n_r]
    with arrow_col:
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
                "<div style='text-align:center;font-size:0.8rem;color:#aaa;'>"
                "no conditions"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Products ─────────────────────────────────────────────────────────────
    for j, p in enumerate(products):
        smiles = p.get("smiles", "")
        label  = p.get("label", f"Product {j + 1}")
        with cols[n_r + 1 + j]:
            with st.container(border=True):
                _render_molecule(smiles, label)

    if not products:
        with cols[n_r + 1]:
            st.caption("_(no products)_")


# ── Collect extra molecules (not already in reactions) ────────────────────────

def _extra_molecules(data: dict, reaction_smiles: set[str]) -> list[dict]:
    """Return molecules from original_molecule_list / molecule_coref
    that were not already shown in any reaction."""
    out: list[dict] = []
    seen: set[str] = set()

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
        show_smiles_code = st.checkbox("Show SMILES strings", value=True)
        show_invalid     = st.checkbox("Show invalid SMILES in extra molecules", value=True)
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

    # ── Reactions (visual) ────────────────────────────────────────────────
    reactions = result.get("reactions", [])

    if reactions:
        st.subheader(f"⚗️ Reactions — {len(reactions)} found")

        # Collect all SMILES shown in reactions for de-duplication later
        reaction_smiles: set[str] = set()
        for rxn in reactions:
            for r in rxn.get("reactants", []):
                reaction_smiles.add((r.get("smiles") or "").strip())
            for p in rxn.get("products", []):
                reaction_smiles.add((p.get("smiles") or "").strip())
        reaction_smiles.discard("")

        for rxn in reactions:
            # Temporarily override show_smiles inside each molecule renderer
            # by monkey-patching the closure via a flag stored in session state
            # — simpler: just pass show_smiles_code via a wrapper
            _render_reaction_with_flag(rxn, show_smiles_code)
            st.divider()
    else:
        st.info("No reactions found in the pipeline output.")
        reaction_smiles = set()

    # ── Extra molecules (not in reactions) ────────────────────────────────
    extras = _extra_molecules(result, reaction_smiles)
    if extras:
        with st.expander(
            f"🔬 Other extracted molecules ({len(extras)})", expanded=bool(not reactions)
        ):
            # Validity summary
            validity = {m["smiles"]: _validate_smiles(m["smiles"])[0] for m in extras}
            valid_n   = sum(1 for v in validity.values() if v)
            invalid_n = len(extras) - valid_n
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(extras))
            c2.metric("Valid ✓", valid_n)
            c3.metric("Invalid ✗", invalid_n)

            display = extras if show_invalid else [m for m in extras if validity[m["smiles"]]]
            if not display:
                st.info("All extra SMILES are invalid (enable 'Show invalid SMILES' in sidebar).")
            else:
                cols_per_row = 3
                for i in range(0, len(display), cols_per_row):
                    batch = display[i : i + cols_per_row]
                    cols  = st.columns(cols_per_row)
                    for col, mol in zip(cols, batch):
                        with col:
                            with st.container(border=True):
                                smiles  = mol["smiles"]
                                label   = mol["label"]
                                context = mol["context"]
                                is_valid, png = _validate_smiles(smiles)
                                badge = "🟢 **Valid**" if is_valid else "🔴 **Invalid**"
                                st.markdown(badge)
                                if label:
                                    st.markdown(f"**Label:** `{label}`")
                                if context:
                                    st.markdown(
                                        f"<span style='font-size:0.8em;color:#888'>{context}</span>",
                                        unsafe_allow_html=True,
                                    )
                                if show_smiles_code:
                                    st.code(smiles, language=None)
                                if is_valid and png:
                                    st.image(png, use_container_width=True)
                                else:
                                    st.caption("_(structure unavailable)_")


# ── Wrapper so show_smiles_code reaches _render_molecule ─────────────────────

def _render_reaction_with_flag(rxn: dict, show_smiles_code: bool) -> None:
    """Like _render_reaction but respects the show_smiles_code toggle."""
    rid   = rxn.get("reaction_id", "?")
    note  = rxn.get("note", "")
    header = f"**Reaction {rid}**" + (f" — {note}" if note else "")
    st.markdown(header)

    reactants  = rxn.get("reactants", [])
    products   = rxn.get("products",  [])
    conditions = rxn.get("conditions", [])

    n_r = max(len(reactants), 1)
    n_p = max(len(products),  1)

    col_weights = [2] * n_r + [1] + [2] * n_p
    cols = st.columns(col_weights)

    # Reactants
    for i, r in enumerate(reactants):
        smiles = r.get("smiles", "")
        label  = r.get("label", f"Reactant {i + 1}")
        with cols[i]:
            with st.container(border=True):
                _render_molecule(smiles, label, show_smiles=show_smiles_code)

    if not reactants:
        with cols[0]:
            st.caption("_(no reactants)_")

    # Arrow + conditions
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
                "<div style='text-align:center;font-size:0.8rem;color:#aaa;'>"
                "no conditions"
                "</div>",
                unsafe_allow_html=True,
            )

    # Products
    for j, p in enumerate(products):
        smiles = p.get("smiles", "")
        label  = p.get("label", f"Product {j + 1}")
        with cols[n_r + 1 + j]:
            with st.container(border=True):
                _render_molecule(smiles, label, show_smiles=show_smiles_code)

    if not products:
        with cols[n_r + 1]:
            st.caption("_(no products)_")


if __name__ == "__main__":
    main()
