"""get_reaction_agent.py

Reaction-extraction agent.  Optimisations vs. original:
  • ChemIEToolkit / RxnIM are loaded once via _shared_models singletons
    (previously each file instantiated its own copy, loading models 4×).
  • encode_image / read_prompt are cached (no repeated disk I/O per call).
  • retry_api_call defined once in _shared_models (was duplicated here).
  • AzureOpenAI / OpenAI clients are singletons (no new HTTP session per call).
  • get_raw_prediction() is called at most once per image path (thread-safe
    cache).  The original code called model1.predict_image_file() twice inside
    get_reaction_withatoms_correctR — once as the LLM tool result and again
    for the final symbol merge step.
  • First LLM call and local model pre-computation run in parallel inside
    get_reaction_withatoms_correctR / get_reaction_withatoms_correctR_OS.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from molnextr.chemistry import _convert_graph_to_smiles
from openai import AzureOpenAI, OpenAI

from _model_lock import CUDA_MODEL_LOCK  # backward-compat import
from _shared_models import (
    encode_image,
    get_azure_client,
    get_raw_prediction,
    get_rxnim,
    read_prompt,
    resolve_os_client,
    retry_api_call,
)


# ── Structured-output helpers ─────────────────────────────────────────────────

def _raw_to_structured(raw_pred: dict) -> dict:
    """Return a compact structured dict from a single raw RxnIM prediction."""
    structured: dict = {}
    for section_key in ("reactants", "conditions", "products"):
        if section_key not in raw_pred:
            continue
        structured[section_key] = []
        for item in raw_pred[section_key]:
            if section_key in ("reactants", "products"):
                structured[section_key].append({
                    "smiles":  item.get("smiles", ""),
                    "bbox":    item.get("bbox", []),
                    "symbols": item.get("symbols", []),
                })
            else:  # conditions
                entry = {"bbox": item.get("bbox", [])}
                if "smiles" in item:
                    entry["smiles"] = item.get("smiles", "")
                    entry["symbols"] = item.get("symbols", [])
                if "text" in item:
                    entry["text"] = item.get("text", [])
                structured[section_key].append(entry)
    return structured


def _update_input_with_symbols(input1: dict, input2: dict, conversion_function) -> dict:
    """Merge GPT-corrected symbols from *input1* into the raw *input2* prediction."""
    symbol_mapping: dict = {}
    for key in ("reactants", "conditions", "products"):
        for item in input1.get(key, []):
            if "symbols" in item and "bbox" in item:
                symbol_mapping[tuple(item["bbox"])] = item["symbols"]

    for key in ("reactants", "conditions", "products"):
        for item in input2.get(key, []):
            if "bbox" not in item:
                continue
            bbox = tuple(item["bbox"])
            if bbox not in symbol_mapping:
                continue
            updated_symbols = symbol_mapping[bbox]
            item["symbols"] = updated_symbols

            if "atoms" in item:
                atoms = item["atoms"]
                if len(atoms) != len(updated_symbols):
                    print(f"Warning: Mismatched symbols and atoms at bbox {bbox}")
                else:
                    for atom, sym in zip(atoms, updated_symbols):
                        atom["atom_symbol"] = sym

            if "coords" in item and "edges" in item:
                new_smiles, new_molfile, _ = conversion_function(
                    item["coords"], updated_symbols, item["edges"]
                )
                item["smiles"] = new_smiles
                item["molfile"] = new_molfile

    return input2


# ── Public API functions (kept for backward compatibility) ────────────────────

def get_reaction(image_path: str) -> dict:
    """Return a structured reaction dict from the cached raw prediction."""
    raw = get_raw_prediction(image_path)
    if not raw:
        return {}
    return _raw_to_structured(raw[0])


def get_full_reaction(image_path: str) -> str:
    """Return a JSON-serialised, trimmed raw prediction list."""
    raw = get_raw_prediction(image_path)
    for reaction in raw:
        for section in ("reactants", "products", "conditions"):
            for entry in reaction.get(section, []):
                coords = entry.get("coords")
                if isinstance(coords, list):
                    entry["coords"] = [[round(v, 3) for v in pt] for pt in coords]
                for key in ("molfile", "atoms", "bonds"):
                    entry.pop(key, None)
    return json.dumps(raw)


# ── Azure backend ─────────────────────────────────────────────────────────────

def get_reaction_withatoms_correctR(image_path: str) -> list:
    """Extract and symbol-correct reactions using AzureOpenAI + RxnIM.

    Optimisations
    -------------
    • RxnIM.predict_image_file is called once via get_raw_prediction() and
      the cached result is reused for both the tool-call response and the
      final symbol-merge step (was called twice before).
    • The first LLM call and the local model pre-computation run in parallel
      so GPU inference overlaps the network round-trip.
    """
    client = get_azure_client()
    b64_image = encode_image(image_path)
    prompt = read_prompt("./prompt/prompt_Rxn_Tem.txt")

    _TOOL_DEF = [
        {
            "type": "function",
            "function": {
                "name": "get_reaction",
                "description": (
                    "Get a list of reactions from a reaction image. "
                    "A reaction contains data of the reactants, conditions, and products."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "The path to the reaction image.",
                        }
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ],
        },
    ]

    # ── Parallel: LLM call 1  ∥  local model prefetch ────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        llm_future = ex.submit(
            client.chat.completions.create,
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=messages,
            tools=_TOOL_DEF,
        )
        raw_future = ex.submit(get_raw_prediction, image_path)

        response = llm_future.result()
        raw_prediction = raw_future.result()   # likely already done; worst-case a brief wait

    if not response.choices:
        return {}
    tool_calls = response.choices[0].message.tool_calls or []

    # Build tool-result messages using the already-computed raw prediction
    tool_results = []
    for tc in tool_calls:
        if tc.function.name != "get_reaction":
            continue
        tool_result = _raw_to_structured(raw_prediction[0]) if raw_prediction else {}
        tool_results.append({
            "role": "tool",
            "name": tc.function.name,
            "content": json.dumps({"image_path": image_path, "get_reaction": tool_result}),
            "tool_call_id": tc.id,
        })

    # ── LLM call 2: symbol correction ────────────────────────────────────────
    completion_payload_msgs = [
        *messages,
        response.choices[0].message,
        *tool_results,
    ]
    response2 = client.chat.completions.create(
        model="gpt-5-mini",
        messages=completion_payload_msgs,
        response_format={"type": "json_object"},
    )

    if not response2.choices:
        return {}
    gpt_output = json.loads(response2.choices[0].message.content)
    print(f"gpt_output_rxn:{gpt_output}")

    # ── Merge: apply GPT-corrected symbols into raw prediction ───────────────
    if not raw_prediction:
        print("WARNING [Azure]: get_raw_prediction returned empty, skipping symbol update")
        return [gpt_output] if isinstance(gpt_output, dict) else []

    updated_data = [_update_input_with_symbols(gpt_output, raw_prediction[0], _convert_graph_to_smiles)]
    print(f"rxn_agent_output:{updated_data}")
    return updated_data


# ── Open-source backend ───────────────────────────────────────────────────────

def get_reaction_withatoms_correctR_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list:
    """OS equivalent of get_reaction_withatoms_correctR (vLLM / Ollama).

    Applies the same parallelism and deduplication as the Azure variant.
    """
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning  # noqa: PLC0415

    client = resolve_os_client(base_url=base_url, api_key=api_key)
    b64_image = encode_image(image_path)
    prompt = read_prompt("./prompt/prompt_Rxn_Tem.txt")

    _TOOL_DEF = [
        {
            "type": "function",
            "function": {
                "name": "get_reaction",
                "description": (
                    "Get a list of reactions from a reaction image. "
                    "A reaction contains data of the reactants, conditions, and products."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "The path to the reaction image.",
                        }
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ],
        },
    ]

    # ── Parallel: LLM call 1  ∥  local model prefetch ────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        llm_future = ex.submit(
            retry_api_call,
            client.chat.completions.create,
            5, 3, 2,
            model=model_name,
            messages=messages,
            tools=_TOOL_DEF,
            tool_choice="auto",
        )
        raw_future = ex.submit(get_raw_prediction, image_path)

        response = llm_future.result()
        raw_prediction = raw_future.result()

    if not response.choices:
        return {}
    tool_calls = response.choices[0].message.tool_calls or []

    tool_results = []
    for tc in tool_calls:
        if tc.function.name != "get_reaction":
            continue
        tool_result = _raw_to_structured(raw_prediction[0]) if raw_prediction else {}
        tool_results.append({
            "role": "tool",
            "name": tc.function.name,
            "content": json.dumps({"image_path": image_path, "get_reaction": tool_result}),
            "tool_call_id": tc.id,
        })

    # ── LLM call 2: symbol correction ────────────────────────────────────────
    completion_msgs = [
        *messages,
        response.choices[0].message,
        *tool_results,
    ]
    response2 = retry_api_call(
        client.chat.completions.create,
        5, 3, 2,
        model=model_name,
        messages=completion_msgs,
    )

    if not response2.choices:
        return {}
    raw_content = response2.choices[0].message.content
    try:
        gpt_output = json.loads(raw_content)
        print("DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print("WARNING [OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        if gpt_output is not None:
            print("DEBUG [OS]: Successfully extracted JSON from text")
        else:
            print("ERROR [OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                "Could not parse JSON from model response.", raw_content, 0
            )

    print(f"gpt_output_rxn:{gpt_output}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    if not raw_prediction:
        print("WARNING [OS]: get_raw_prediction returned empty, skipping symbol update")
        return [gpt_output] if isinstance(gpt_output, dict) else []

    updated_data = [_update_input_with_symbols(gpt_output, raw_prediction[0], _convert_graph_to_smiles)]
    print(f"rxn_agent_output:{updated_data}")
    return updated_data


# ── Legacy helpers (kept for callers that import them directly) ───────────────

def get_reaction_withatoms(image_path: str) -> list:
    """Legacy wrapper — delegates to get_reaction_withatoms_correctR."""
    return get_reaction_withatoms_correctR(image_path)
