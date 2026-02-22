# ChemEAGLE — Pipeline Failure Analysis

> **Purpose:** Maps each known and discoverable failure mode to the exact pipeline stage, agent, model, or tool responsible, and explains *why* the failure occurs mechanistically.

---

## Full Pipeline with Failure Hotspots

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                   │
│  ┌────────────┐    ┌────────────────────────────────────────────────┐   │
│  │  PDF file  │───►│  PDF Extractor  (pdfmodel / VisualHeist)       │   │
│  └────────────┘    │  Florence-2 OD: detects ALL figures & tables   │   │
│                    │  ⚠ ISSUE 3 — no chemistry-content filter        │   │
│                    └────────────────────┬───────────────────────────┘   │
│                                         │ .png crops (ALL figures)      │
│  ┌─────────────┐                        │                               │
│  │ Image file  │────────────────────────┘                               │
│  └─────────────┘                                                         │
└────────────────────────────────────────────────────────────────────────┬─┘
                                                                         │
                                                                         ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  STAGE 1 — PLANNER  (gpt-5-mini / Qwen3-VL)                       │
       │  Prompt: prompt_plan_new.txt                                        │
       │  Task: classify image → select agents                              │
       │                                                                    │
       │  ⚠ ISSUE 2 — complex/non-standard graphics                         │
       │    Spectral plots, mechanism diagrams, bio-figures                 │
       │    misclassified as reaction schemes; wrong agents selected        │
       │                                                                    │
       │  ⚠ ISSUE 4 — R-group table type misclassification                  │
       │    Mutual exclusion rule (structure-based vs text-based)           │
       │    may pick the wrong agent when both types are present            │
       └──────────────────────────────┬─────────────────────────────────────┘
                                      │ agent_list
                                      ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  STAGE 2 — PLAN OBSERVER  (gpt-5-mini / Qwen3-VL)                 │
       │  File: get_observer.py → plan_observer_agent()                     │
       │                                                                    │
       │  ⚠ Same LLM as Planner — inherits the same blind spots            │
       │    If Planner misclassified, Observer typically agrees             │
       │    Disabled by default in OS mode — no safety net                 │
       └──────────────────────────────┬─────────────────────────────────────┘
                                      │ plan_to_execute
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — TOOL EXECUTOR LOOP                                               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  PATH A  get_full_reaction_template()  [Reaction Template Parsing]    │ │
│  │                                                                       │ │
│  │   image ──► RxnIM.predict_image_file()  (rxn.ckpt)                   │ │
│  │             ⚠ ISSUE 1a — 3D/stereo notation                           │ │
│  │               Wedge/dash bonds, Newman/Fischer projections cause      │ │
│  │               incorrect template bounding box detection               │ │
│  │             ⚠ ISSUE 5  — multi-step reactions                         │ │
│  │               Single-step model merges or splits cascade steps        │ │
│  │                        │                                              │ │
│  │                        ▼                                              │ │
│  │             ChemIEToolkit  (MolDetector)                              │ │
│  │             ⚠ ISSUE 1b — stereo wedge bonds extend bounding boxes,    │ │
│  │               overlapping adjacent labels or structures               │ │
│  │             ⚠ ISSUE 10 — labels "1a","2b" near structures             │ │
│  │               cause wrong label-SMILES associations                   │ │
│  │                        │                                              │ │
│  │                        ▼                                              │ │
│  │             MolNexTR  Image2Graph → Graph2SMILES                      │ │
│  │             ⚠ ISSUE 1c  ← PRIMARY ROOT CAUSE FOR STEREO               │ │
│  │               Trained on 2D Kekulé structures only.                   │ │
│  │               Wedge/dash → regular bonds; stereocenters erased.       │ │
│  │               Outputs flat SMILES; all @/@@ / \ tokens lost          │ │
│  │                        │                                              │ │
│  │                        ▼                                              │ │
│  │             LLM  (condition role classification)                      │ │
│  │             ⚠ ISSUE 6 — abbreviation SMILES from LLM memory           │ │
│  │               DIPEA, BINAP, Pd/C etc — no database validation         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  PATH B  process_reaction_image_with_product_variant_R_group()        │ │
│  │  [Structure-based R-group Substitution]                               │ │
│  │                                                                       │ │
│  │   All PATH A failures apply, PLUS:                                    │ │
│  │   Variant panel crops ──► MolNexTR per crop                           │ │
│  │   ⚠ ISSUE 1c amplified — stereo lost in every enumerated product      │ │
│  │   normalize_product_variant_output() scaffold substitution            │ │
│  │   ⚠ ISSUE 4b — scaffold SMILES mismatch → invalid product SMILES      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  PATH C  process_reaction_image_with_table_R_group()                  │ │
│  │  [Text-based R-group Substitution]                                    │ │
│  │                                                                       │ │
│  │   All PATH A failures apply, PLUS:                                    │ │
│  │   TableParser  (OCR-based table detection)                            │ │
│  │   ⚠ ISSUE 8a — merged/spanning cells silently dropped or misread      │ │
│  │   TesseractOCR  cell content reading                                  │ │
│  │   ⚠ ISSUE 8b — CF3/NO2 subscripts collapse; degree/mu/eta misread    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  PATH D  text_extraction_agent()                                      │ │
│  │  File: get_text_agent.py                                              │ │
│  │                                                                       │ │
│  │   TesseractOCR  (general-purpose)                                     │ │
│  │   ⚠ ISSUE 8c — IUPAC names, Greek letters, superscripts misread       │ │
│  │                        │                                              │ │
│  │                        ▼                                              │ │
│  │   ChemRxnExtractor  (BioBERT-based NLP, cre_models_v0.1)             │ │
│  │   ⚠ ISSUE 9a — trained on biomedical text; synthetic organic          │ │
│  │                 chemistry jargon reduces extraction recall             │ │
│  │                        │                                              │ │
│  │                        ▼                                              │ │
│  │   ChemNER / MolNER  (ner.ckpt)                                        │ │
│  │   ⚠ ISSUE 9b — proprietary ligand names (BrettPhos, SPhos,            │ │
│  │                 RuPhos) not in training set → missed entities          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ execution_logs + results
                                     ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  STAGE 4 — ACTION OBSERVER  (gpt-5-mini / Qwen3-VL)               │
       │  File: get_observer.py → action_observer_agent()                   │
       │                                                                    │
       │  ⚠ ISSUE 7a — redo signal never acted upon                         │
       │    Returns {"redo":True} but orchestrator exits immediately.       │
       │    README usage example has no retry loop → signal dropped         │
       │                                                                    │
       │  ⚠ ISSUE 7b — stereo blind spot                                    │
       │    Flat SMILES without stereo is syntactically valid —             │
       │    LLM "is SMILES reasonable?" check passes it through             │
       │                                                                    │
       │  ⚠ ISSUE 7c — disabled in OS mode by default                       │
       └──────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │  STAGE 5 — DATA STRUCTURE AGENT  (gpt-5-mini / Qwen3-VL)          │
       │  Prompt: prompt_final_simple_version.txt                           │
       │                                                                    │
       │  ⚠ ISSUE 6b — SMILES hallucination for unknown abbreviations       │
       │    Prompt explicitly asks LLM to convert abbrevs "from knowledge"  │
       │                                                                    │
       │  ⚠ ISSUE 2b — non-chemistry image confusion                         │
       │    LLM may fabricate plausible-looking reactions from garbage input │
       │                                                                    │
       │  ⚠ ISSUE 5b — multi-step label mismatch perpetuated in output      │
       └──────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
                            {"reactions": [...]}
```

---

## Failure Mode Registry

### Issue 1 — 3D / Stereochemical Representations

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 1a | Stage 3 PATH A | **RxnIM** `rxn.ckpt` | Trained on 2D standard schemes; wedge/dash bonds extend bounding regions causing template mis-detection |
| 1b | Stage 3 PATH A | **MolDetector** (ChemIEToolkit) | Stereo bond lines overlap with adjacent labels; wrong crop regions fed to MolNexTR |
| **1c** | Stage 3 ALL PATHS | **MolNexTR** `Image2Graph→Graph2SMILES` | **Primary root cause.** Trained exclusively on 2D Kekulé structures. Wedge=regular bond, stereocenters erased. Outputs flat SMILES with no `@`, `/`, `\` tokens |
| 1d | Stage 4 | **Action Observer** (LLM) | Flat SMILES is syntactically valid; LLM "is SMILES reasonable?" passes it |

Affected input types: wedge/dash bonds, Newman projections, Fischer projections, Haworth projections, ORTEP crystal structure diagrams.

---

### Issue 2 — Complex / Non-Standard Graphics

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 2a | Stage 1 | **Planner** LLM | Prompt only describes reaction categories; mechanism arrows, spectral baselines resemble reaction arrows → wrong agent selected |
| 2b | Stage 2 | **Plan Observer** (same LLM) | Same model, same image — if Planner was wrong, Observer typically agrees |
| 2c | Stage 3 | **RxnIM** | Detects arrow-like features in flow charts, mechanisms, spectra as reaction arrows |
| 2d | Stage 3 | **MolDetector** | Ring patterns in bio-diagrams, hexagons in materials figures detected as benzene rings |
| 2e | Stage 5 | **Data Structure Agent** LLM | Given bad tool output, LLM fabricates plausible-looking SMILES to fill required JSON format |

Affected input types: catalytic cycle diagrams, mechanism arrows, NMR/MS/IR spectral figures, energy diagrams, protein schematics, bar charts with molecule decorations.

---

### Issue 3 — PDF Scanning Includes Non-Chemistry Images

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 3a | PDF extraction | **VisualHeist** (Florence-2 `<OD>`) | Object detection model extracts ALL figures/tables with no chemistry-specific classifier |
| 3b | Main loop | **User integration code** (README) | Recommended usage iterates every `.png` in `output_dir` and passes all to `ChemEagle()` without content-type check |
| 3c | Stages 1–5 | All agents | Every non-chemistry image burns through the full multi-agent pipeline, producing garbage output |

Affected inputs: experimental photographs, TOC art, supporting information plots, PAGE gels, microscopy images — all common in journal PDFs.

---

### Issue 4 — R-group Agent Misclassification

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 4a | Stage 1 | **Planner** LLM | Mutual exclusion rule forces one choice; LLM may choose wrong agent when both table types appear |
| 4b | Stage 3 PATH B | **`normalize_product_variant_output()`** | Core scaffold SMILES from RxnIM may not match variant attachment points (due to 1a/1c errors) → invalid product SMILES |
| 4c | Stage 3 PATH C | **TableParser + LLM** | Footnote symbols and multi-row headers cause column-row misalignment; wrong R-groups assigned |

---

### Issue 5 — Multi-step Reaction Schemes

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 5a | Stage 3 | **RxnIM** `rxn.ckpt` | Designed for single-step; cascade reactions are merged into one or split incorrectly |
| 5b | Stage 5 | **Data Structure Agent** LLM | Intermediates passed without explicit step tagging; LLM infers structure from ambiguous positional cues |
| 5c | Stage 3 | **ChemIEToolkit** | "Step 1:", "Step 2:", roman numeral labels above arrows not parsed as step delimiters |

---

### Issue 6 — Abbreviation / Shorthand SMILES Hallucination

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 6a | Stage 3 | **LLM** condition classifier | Reagent abbreviations resolved from training memory, no database validation |
| 6b | Stage 5 | **Data Structure Agent** LLM | Prompt explicitly instructs "convert chemistry texts to SMILES based on your knowledge" — hallucinates rare/proprietary names |
| **6c** | All stages | *Missing component* | **No SMILES validation anywhere.** No RDKit valence check, no InChI roundtrip — invalid SMILES silently reach the final output |

---

### Issue 7 — Action Observer "Redo" Never Executes

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 7a | Stage 4 | **Orchestrator** `main.py` | When observer returns `True`, `ChemEagle()` immediately returns `{"redo":True,...}` and exits. README example has no retry wrapper |
| 7b | Stage 4 | **Action Observer** LLM | Stereo-erased flat SMILES is valid syntax → passes "is SMILES reasonable?" check |
| 7c | Stage 4 | **`ChemEagle_OS()`** | `use_action_observer=False` by default in OS mode — validator entirely absent |

---

### Issue 8 — OCR Quality Degradation

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 8a | Stage 3 PATH C | **TableParser** | Merged/spanning cells, footnote symbols cause column misalignment in R-group tables |
| 8b | Stage 3 PATH C,D | **TesseractOCR** | Subscripts (CF3, NO2), Greek letters (degree, mu, eta, alpha), ± misread as alphanumerics |
| 8c | Stage 3 PATH D | **TesseractOCR** | Chemical formula subscripts lost; breaks ChemRxnExtractor tokenisation |
| 8d | All paths | **TesseractOCR + MolDetector** | Low-DPI PDF renders cause OCR failure and missed molecule detection; no pre-processing/upscaling step |

---

### Issue 9 — NLP Model Domain Mismatch

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 9a | Stage 3 PATH D | **ChemRxnExtractor** `cre_models_v0.1` | BioBERT-based; biomedical training corpus → synthetic organic chemistry terminology reduces recall |
| 9b | Stage 3 PATH D | **ChemNER** `ner.ckpt` | Proprietary ligand names (BrettPhos, SPhos, RuPhos, Josiphos) absent from training → missed entities |
| 9c | Stage 5 | **Data Structure Agent** LLM | Garbled OCR text breaks text-to-graphical-label alignment in the synthesizer |

---

### Issue 10 — Label vs. Molecule Boundary Confusion

| Sub | Stage | Component | Root cause |
|---|---|---|---|
| 10a | Stage 3 | **MolDetector** bounding boxes | Compound labels ("1a", "cat.", "Ar") adjacent to structures are included in the molecule crop |
| 10b | Stage 3, 5 | **LLM** (template parsing) | Labels inside the crop confuse MolNexTR graph-building; synthesizer cannot match label to SMILES |

---

## Summary Matrix

| # | Failure | Planner | Plan Observer | RxnIM | MolDetector | MolNexTR | OCR / TableParser | LLM Agents | Action Observer | PDF Extractor |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 3D/stereo molecules | | | ✦ | ✦ | **★** | | | ✦ | |
| 2 | Complex non-reaction graphics | **★** | ✦ | ✦ | ✦ | | | ✦ | | |
| 3 | Non-chemistry images from PDF | | | | | | | | | **★** |
| 4 | R-group agent misclassification | **★** | | | | ✦ | ✦ | ✦ | | |
| 5 | Multi-step reaction schemes | | | **★** | | ✦ | | ✦ | | |
| 6 | Abbreviation SMILES hallucination | | | | | | | **★** | ✦ | |
| 7 | Redo loop never executes | | | | | | | | **★** | |
| 8 | OCR quality degradation | | | | ✦ | | **★** | | | |
| 9 | NLP domain mismatch | | | | | | ✦ | **★** | | |
| 10 | Label vs. molecule confusion | | | | **★** | ✦ | | ✦ | | |

**★** = primary responsible component   **✦** = contributing factor

---

## Severity vs. Frequency

```
HIGH freq │ Issue 1  3D/stereo           — most papers use stereo notation
          │ Issue 6  abbreviation SMILES  — conditions always contain abbrevs
          │ Issue 3  PDF non-chem images  — every PDF has non-reaction figs
          │
MED freq  │ Issue 8  OCR quality          — all text-heavy reaction tables
          │ Issue 2  complex graphics      — common in overview/methods figs
          │ Issue 5  multi-step reactions  — cascade reactions are common
          │
LOW freq  │ Issue 4  R-group agent        — only when both table types exist
          │ Issue 10 label confusion       — layout-dependent
          │ Issue 9  NLP mismatch          — partial; common terms are fine
          │ Issue 7  redo loop             — observer rarely triggers redo
          │
          └────────────────────────────────────────────────► HIGH severity
                    7    9    4    10    8     5    2    1    3    6
```

---

## Key Findings

1. **MolNexTR is the single highest-impact failure point** (Issue 1). Every pipeline path routes through it for SMILES generation. Any image with stereochemistry loses that information irreversibly — no downstream LLM can recover stereo from a flat SMILES without re-examining the original image crop.

2. **The PDF extraction layer has zero chemistry awareness** (Issue 3). VisualHeist extracts ALL figures from PDFs. Without a chemistry-content classifier between extraction and ChemEagle, every production run on real journal PDFs includes noisy, irrelevant images.

3. **The Action Observer `redo` mechanism is architecturally incomplete** (Issue 7). The observer can detect failures but the orchestrator only returns the signal as data. The calling code must implement the retry loop — the documented example does not.

4. **No SMILES validation exists at any stage** (Issue 6c). No RDKit valence check, no InChI roundtrip, no cheminformatics validation is applied to any LLM-generated or MolNexTR-generated SMILES string anywhere in the pipeline.

5. **OCR is a systemic weak link** (Issue 8). Both the condition-parsing path (TableParser) and the text-extraction path (TesseractOCR) use general-purpose OCR with no chemistry-aware post-correction. Errors silently propagate into the final JSON.
