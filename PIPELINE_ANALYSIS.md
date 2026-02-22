# ChemEAGLE Reaction Image Agent Pipeline Analysis

## Overview

ChemEAGLE is a multimodal multi-agent system that extracts structured chemical reaction data from images and PDF literature. It orchestrates several specialized agents through a **Planner → Observer → Executor → Synthesizer** pipeline.

---

## Pipeline Architecture

```
Input Image
    │
    ▼
┌─────────────────────┐
│      Planner        │  (LLM call) Analyzes the image and selects
│                     │  which agents to invoke
└────────┬────────────┘
         │ agent_list (e.g. "Reaction Template Parsing Agent, Text Extraction Agent")
         ▼
┌─────────────────────┐
│    Plan Observer    │  (optional) Reviews and may modify the
│                     │  execution plan before any tools run
└────────┬────────────┘
         │ plan_to_execute (list of {id, name, arguments})
         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tool Executor Loop                      │
│                                                             │
│  For each tool call in plan_to_execute:                     │
│    ┌──────────────────────────────────────────────────┐     │
│    │  • process_reaction_image_with_product_variant_R_group  │
│    │  • process_reaction_image_with_table_R_group            │
│    │  • get_full_reaction_template                           │
│    │  • get_multi_molecular_full                             │
│    │  • text_extraction_agent                               │
│    └──────────────────────────────────────────────────┘     │
└────────┬────────────────────────────────────────────────────┘
         │ execution_logs + results
         ▼
┌─────────────────────┐
│   Action Observer   │  (optional) Validates execution results;
│                     │  flags failures (returns {"redo": True})
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Data Structure     │  Final LLM call with all tool outputs;
│  Agent (Synthesizer)│  produces JSON-structured reaction data
└────────┬────────────┘
         │
         ▼
    Output Dict
    {"reactions": [...], "text_extraction": "..."}
```

---

## Step-by-Step Pipeline Description

### Step 1 — Planner
**File:** `main.py` → `ChemEagle()` / `ChemEagle_OS()`
**Prompt:** `prompt/prompt_plan_new.txt`

The Planner receives the raw image (base64-encoded) and a prompt listing all available agents. It returns a **comma-separated list** of agent names to invoke. Agent selection logic:

| Planner Output | Selected Tool Function |
|---|---|
| `Structure-based R-group Substitution Agent` | `process_reaction_image_with_product_variant_R_group` |
| `Text-based R-group Substitution Agent` | `process_reaction_image_with_table_R_group` |
| `Reaction Template Parsing Agent` | `get_full_reaction_template` |
| `Molecular Recognition Agent` | `get_multi_molecular_full` |
| *(fallback / no match)* | `get_full_reaction_template` |

If `Text Extraction Agent` appears in the planner output, it is **always appended** as a second tool call.

---

### Step 2 — Plan Observer (optional)
**File:** `get_observer.py` → `plan_observer_agent()` / `plan_observer_agent_OS()`

Receives the image and the serialized execution plan. May approve, modify, or reorder tool calls before execution. If the observer returns an empty or invalid plan, the original plan is used as fallback.

---

### Step 3 — Tool Executor Loop
**File:** `main.py`

Iterates over `plan_to_execute`. For each item:
1. Resolves `tool_name` → Python function via `TOOL_MAP`
2. Normalizes `image_path` argument (replaces placeholders)
3. Calls the function and appends result to `execution_logs` and `results`

---

### Step 4 — Action Observer (optional)
**File:** `get_observer.py` → `action_observer_agent()` / `action_observer_agent_OS()`

Reviews `execution_logs`. If it detects a failure, returns `True`, causing the pipeline to return `{"redo": True, "plan": ..., "execution_logs": ...}` — signalling the caller to retry.

---

### Step 5 — Data Structure Agent (Synthesizer)
**File:** `main.py`
**Prompt:** `prompt/prompt_final_simple_version.txt`

A final LLM call that receives:
- The original image
- The system prompt
- An `assistant` message summarising selected agents
- All tool results as `role: tool` messages

Returns a JSON object with the structured reaction data.

---

## Agent / Tool Details

### Reaction Template Parsing Agent
**Function:** `get_full_reaction_template` / `get_full_reaction_template_OS`
**File:** `get_R_group_sub_agent.py`
**Use case:** Images with a reaction diagram but **no R-group tables** or product variant sets.
**Pipeline:**
1. `RxnIM.predict_image_file()` — parses reaction image bounding boxes and components
2. `ChemIEToolkit` — molecule detection and SMILES extraction
3. LLM call to assign conditions, reactants, products

---

### Structure-based R-group Substitution Agent
**Function:** `process_reaction_image_with_product_variant_R_group` / `..._OS`
**File:** `get_R_group_sub_agent.py`
**Use case:** Images where product variants are shown as a **set of structures** (structure-based table).
**Pipeline:**
1. Reaction template extraction
2. Molecule cropping and recognition (MolNexTR via `process_reaction_image_with_multiple_products_and_text_correctmultiR`)
3. R-group substitution into reactant SMILES
4. Output normalized via `normalize_product_variant_output()`

---

### Text-based R-group Substitution Agent
**Function:** `process_reaction_image_with_table_R_group` / `..._OS`
**File:** `get_R_group_sub_agent.py`
**Use case:** Images with a **text/table** listing R-group values.
**Pipeline:**
1. Table parsing (OCR + `TableParser`)
2. R-group extraction from text cells
3. SMILES reconstruction for each table row
4. Assembled into reaction list

---

### Molecular Recognition Agent
**Function:** `get_multi_molecular_full`
**File:** `get_R_group_sub_agent.py`
**Use case:** Images of **single or multiple molecules** (no reaction arrow).
**Pipeline:**
1. `MolDetector` segments each molecule
2. `Image2Graph` → `Graph2SMILES` (MolNexTR) per segment
3. Returns list of SMILES

---

### Text Extraction Agent
**Function:** `text_extraction_agent` / `text_extraction_agent_OS`
**File:** `get_text_agent.py`
**Use case:** Always run alongside the primary reaction agent when selected.
**Pipeline:**
1. OCR with TesseractOCR
2. `ChemRxnExtractor` and `MolNER` on extracted text
3. Returns structured text-based reaction/condition info

---

### Molecular Recognition Sub-agent
**Functions:** `process_reaction_image_with_multiple_products_and_text_correctR`, `..._correctmultiR`
**File:** `get_molecular_agent.py`
**Use case:** Called internally by R-group agents for per-molecule SMILES.
**Pipeline:**
1. Crop molecule sub-images from bounding boxes
2. `MolNexTR` graph → SMILES conversion
3. Optionally merge with OCR-corrected symbol list

---

### Reaction Data Sub-agent
**Function:** `get_reaction_withatoms_correctR` / `..._OS`
**File:** `get_reaction_agent.py`
**Use case:** Called internally; extracts atom-mapped reactions from image.
**Pipeline:**
1. `RxnIM.predict_image_file()` for bounding box detection
2. `ChemIEToolkit` molecule detection
3. LLM call for condition classification and role assignment
4. Retry logic on 503/overload errors (3 retries, exponential backoff)

---

## Data Flow Summary

```
image_path
  │
  ├─► [Planner LLM] ──────────────► agent_list
  │
  ├─► [Plan Observer LLM] ─────────► plan_to_execute
  │
  ├─► [RxnIM] ─────────────────────► raw reaction bounding boxes
  │
  ├─► [ChemIEToolkit / MolDetector] ► molecule sub-images + SMILES
  │
  ├─► [MolNexTR: Image2Graph + Graph2SMILES] ► molecular SMILES strings
  │
  ├─► [OCR / TableParser] ──────────► text, R-group table cells
  │
  ├─► [LLM agents] ────────────────► structured reaction dicts per tool
  │
  ├─► [Action Observer LLM] ───────► pass / redo signal
  │
  └─► [Data Structure LLM] ────────► final JSON {"reactions": [...]}
```

---

## Two Deployment Modes

| Feature | `ChemEagle` (Azure) | `ChemEagle_OS` (vLLM) |
|---|---|---|
| LLM backend | Azure OpenAI (`gpt-5-mini`) | Local vLLM (`Qwen3-VL-32B-Instruct-AWQ`) |
| Plan Observer | Enabled by default | Disabled by default |
| Action Observer | Enabled by default | Disabled by default |
| JSON parsing | `json.loads()` direct | Direct + `extract_json_from_text_with_reasoning()` fallback |
| Tool response format | No `name` field required | `name` field required (OpenAI-compatible API) |
