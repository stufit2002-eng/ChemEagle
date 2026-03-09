"""get_molecular_agent.py

Molecular-recognition / R-group correction agents.  Optimisations vs. original:
  • ChemIEToolkit is loaded once via _shared_models (not re-instantiated here).
  • encode_image / read_prompt / retry_api_call come from _shared_models.
  • Azure / OS clients are singletons — no new HTTP session per call.
  • extract_molecule_corefs_from_figures is called once per image path through
    get_coref_results() (thread-safe cache).  The original code called it twice
    per function: once as the LLM tool response and again for the final merge.
  • First LLM call and local model pre-computation run in parallel so GPU
    inference overlaps the network round-trip.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from molnextr.chemistry import _convert_graph_to_smiles

from _shared_models import (
    CUDA_MODEL_LOCK,
    encode_image,
    get_azure_client,
    get_chemiie_toolkit,
    get_coref_results,
    read_prompt,
    resolve_os_client,
    retry_api_call,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _update_symbols_in_atoms(input1: list, input2: list) -> list:
    """Copy corrected *symbols* from *input1* bboxes into *input2* atoms."""
    for item1, item2 in zip(input1, input2):
        bboxes1 = item1.get("bboxes", [])
        bboxes2 = item2.get("bboxes", [])
        if len(bboxes1) != len(bboxes2):
            print("Warning: Mismatched number of bboxes!")
            continue
        for bb1, bb2 in zip(bboxes1, bboxes2):
            if "symbols" not in bb1:
                continue
            bb2["symbols"] = bb1["symbols"]
            if "atoms" in bb2:
                atoms = bb2["atoms"]
                syms = bb1["symbols"]
                if len(syms) != len(atoms):
                    print(f"Warning: Mismatched symbols and atoms in bbox {bb1.get('bbox')}!")
                    continue
                for atom, sym in zip(atoms, syms):
                    atom["atom_symbol"] = sym
    return input2


def _update_symbols_and_corefs(gpt_outputs: list, coref_results: list) -> list:
    """Merge GPT-corrected symbols + corefs into *coref_results*."""
    results = []
    for item1, item2 in zip(gpt_outputs, coref_results):
        orig_bboxes = item2.get("bboxes", [])
        orig_corefs = item2.get("corefs", [])
        coord2idx = {tuple(bb["bbox"]): i for i, bb in enumerate(orig_bboxes)}

        new_bboxes = []
        for bb1 in item1.get("bboxes", []):
            coord = tuple(bb1["bbox"])
            if coord not in coord2idx:
                raise ValueError(f"bbox {coord} not found in original template!")
            bb_new = copy.deepcopy(orig_bboxes[coord2idx[coord]])
            if "symbols" in bb1:
                bb_new["symbols"] = bb1["symbols"]
                if "atoms" in bb_new:
                    for atom, sym in zip(bb_new["atoms"], bb1["symbols"]):
                        atom["atom_symbol"] = sym
            for field in ("text", "sub_text"):
                if field in bb1:
                    bb_new[field] = bb1[field]
            bb_new["bbox"] = bb1["bbox"]
            new_bboxes.append(bb_new)

        coord2new_idxs: dict[tuple, list] = {}
        for idx, bb in enumerate(new_bboxes):
            coord2new_idxs.setdefault(tuple(bb["bbox"]), []).append(idx)

        new_corefs = []
        for group in orig_corefs:
            label_idx = group[-1]
            label_coord = tuple(orig_bboxes[label_idx]["bbox"])
            new_label_idx = coord2new_idxs[label_coord][-1]
            for mol_idx in group[:-1]:
                mol_coord = tuple(orig_bboxes[mol_idx]["bbox"])
                for new_mol_idx in coord2new_idxs[mol_coord]:
                    new_corefs.append([new_mol_idx, new_label_idx])

        new_item = copy.deepcopy(item2)
        new_item["bboxes"] = new_bboxes
        new_item["corefs"] = new_corefs
        results.append(new_item)
    return results


def _update_smiles_and_molfile(input_data: list, conversion_function) -> list:
    """Regenerate smiles / molfile from corrected coords + symbols + edges."""
    for item in input_data:
        for bbox in item.get("bboxes", []):
            if all(k in bbox for k in ("coords", "symbols", "edges")):
                new_smiles, new_molfile, _ = conversion_function(
                    bbox["coords"], bbox["symbols"], bbox["edges"]
                )
                bbox["smiles"] = new_smiles
                bbox["molfile"] = new_molfile
    return input_data


def _coref_for_tool(coref_results: list, strip_keys: tuple) -> str:
    """Return a JSON string of *coref_results* with *strip_keys* removed."""
    data = copy.deepcopy(coref_results)
    for item in data:
        for bb in item.get("bboxes", []):
            for k in strip_keys:
                bb.pop(k, None)
    return json.dumps(data)


_STRIP_KEYS_WITHATOMS = (
    "category", "molfile", "symbols", "atoms", "bonds",
    "category_id", "score", "corefs",
)

_STRIP_KEYS_NO_BBOX = (
    "category", "bbox", "molfile", "symbols", "atoms", "bonds",
    "category_id", "score", "corefs",
)


# ── Azure helpers ─────────────────────────────────────────────────────────────

def _run_mol_llm_round_trip(
    image_path: str,
    prompt_path: str,
    tool_name: str,
    strip_keys: tuple,
    *,
    model: str = "gpt-4o",
) -> tuple[list, list]:
    """Run a 2-call LLM loop with local-model tool in parallel with call 1.

    Returns (gpt_corrected_output: list, raw_coref_results: list).
    The raw coref results are computed via get_coref_results() which caches
    them so they are not recomputed for subsequent calls on the same image.
    """
    client = get_azure_client()
    b64 = encode_image(image_path)
    prompt = read_prompt(prompt_path)

    _tool_def = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "Extracts the SMILES string, the symbols set, and the text coref "
                    "of all molecular images in a table-reaction image and ready to be correct."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to the reaction image."}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    msgs_user = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]

    # ── Parallel: LLM call 1  ∥  local model ─────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        llm_future = ex.submit(
            client.chat.completions.create,
            model=model,
            response_format={"type": "json_object"},
            messages=msgs_user,
            tools=_tool_def,
        )
        coref_future = ex.submit(get_coref_results, image_path)

        response1 = llm_future.result()
        coref_results = coref_future.result()   # cached; may already be done

    if not response1.choices:
        return [], coref_results

    # Build tool-result messages using already-computed coref
    tool_results = []
    for tc in (response1.choices[0].message.tool_calls or []):
        tool_results.append({
            "role": "tool",
            "name": tc.function.name,
            "content": json.dumps({
                "image_path": image_path,
                tc.function.name: _coref_for_tool(coref_results, strip_keys),
            }),
            "tool_call_id": tc.id,
        })

    # ── LLM call 2: symbol correction ────────────────────────────────────────
    response2 = client.chat.completions.create(
        model=model,
        messages=[*msgs_user, response1.choices[0].message, *tool_results],
        response_format={"type": "json_object"},
    )

    if not response2.choices:
        return [], coref_results
    gpt_output = [json.loads(response2.choices[0].message.content)]
    print(f"gpt_output_mol:{gpt_output}")
    return gpt_output, coref_results


# ── Public Azure functions ────────────────────────────────────────────────────

def process_reaction_image_with_multiple_products_and_text(image_path: str) -> list:
    gpt_output, coref_results = _run_mol_llm_round_trip(
        image_path,
        "./prompt/prompt_getmolecular.txt",
        "get_multi_molecular_text_to_correct_withatoms",
        _STRIP_KEYS_WITHATOMS,
    )
    if not gpt_output:
        return coref_results
    merged = _update_symbols_in_atoms(gpt_output, coref_results)
    return _update_smiles_and_molfile(merged, _convert_graph_to_smiles)


def process_reaction_image_with_multiple_products_and_text_correctR(image_path: str) -> list:
    gpt_output, coref_results = _run_mol_llm_round_trip(
        image_path,
        "./prompt/prompt_getmolecular_correctR.txt",
        "get_multi_molecular_text_to_correct_withatoms",
        _STRIP_KEYS_WITHATOMS,
    )
    if not gpt_output:
        return coref_results
    merged = _update_symbols_in_atoms(gpt_output, coref_results)
    result = _update_smiles_and_molfile(merged, _convert_graph_to_smiles)
    print(f"mol_agent_output:{result}")
    return result


def process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path: str) -> list:
    """Like _correctR but uses the multiR prompt (gpt-5-mini) and corefs merge."""
    client = get_azure_client()
    b64 = encode_image(image_path)
    prompt = read_prompt("./prompt/prompt_Mol_Reco.txt")
    tool_name = "get_multi_molecular_text_to_correct_withatoms"

    _tool_def = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "Extracts the SMILES string, the symbols set, and the text coref "
                    "of all molecular images in a table-reaction image and ready to be correct."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to the reaction image."}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    msgs_user = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]

    # ── Parallel: LLM call 1  ∥  local model ─────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        llm_future = ex.submit(
            client.chat.completions.create,
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=msgs_user,
            tools=_tool_def,
        )
        coref_future = ex.submit(get_coref_results, image_path)

        response1 = llm_future.result()
        coref_results = coref_future.result()

    if not response1.choices:
        return coref_results

    tool_results = []
    for tc in (response1.choices[0].message.tool_calls or []):
        tool_results.append({
            "role": "tool",
            "name": tc.function.name,
            "content": json.dumps({
                "image_path": image_path,
                tc.function.name: _coref_for_tool(coref_results, _STRIP_KEYS_WITHATOMS),
            }),
            "tool_call_id": tc.id,
        })

    response2 = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[*msgs_user, response1.choices[0].message, *tool_results],
        response_format={"type": "json_object"},
    )

    if not response2.choices:
        return coref_results
    gpt_output = [json.loads(response2.choices[0].message.content)]
    print(f"gpt_output_mol:{gpt_output}")

    # Merge using corefs (already available from parallel step)
    merged = _update_symbols_and_corefs(gpt_output, coref_results)
    result = _update_smiles_and_molfile(merged, _convert_graph_to_smiles)
    print(f"mol_agent_output:{result}")
    return result


# ── OS (vLLM / Ollama) equivalent ────────────────────────────────────────────

def process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list:
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning  # noqa: PLC0415

    client = resolve_os_client(base_url=base_url, api_key=api_key)
    b64 = encode_image(image_path)
    prompt = read_prompt("./prompt/prompt_Mol_Reco.txt")
    tool_name = "get_multi_molecular_text_to_correct_withatoms"

    _tool_def = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "Extracts the SMILES string, the symbols set, and the text coref "
                    "of all molecular images in a table-reaction image and ready to be correct."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to the reaction image."}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    msgs_user = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]

    # ── Parallel: LLM call 1  ∥  local model ─────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        llm_future = ex.submit(
            retry_api_call,
            client.chat.completions.create,
            5, 3, 2,
            model=model_name,
            messages=msgs_user,
            tools=_tool_def,
            tool_choice="auto",
        )
        coref_future = ex.submit(get_coref_results, image_path)

        response1 = llm_future.result()
        coref_results = coref_future.result()

    if not response1.choices:
        return coref_results

    tool_results = []
    for tc in (response1.choices[0].message.tool_calls or []):
        tool_results.append({
            "role": "tool",
            "name": tc.function.name,
            "content": json.dumps({
                "image_path": image_path,
                tc.function.name: _coref_for_tool(coref_results, _STRIP_KEYS_WITHATOMS),
            }),
            "tool_call_id": tc.id,
        })

    response2 = retry_api_call(
        client.chat.completions.create,
        5, 3, 2,
        model=model_name,
        messages=[*msgs_user, response1.choices[0].message, *tool_results],
    )

    if not response2.choices:
        return coref_results
    raw_content = response2.choices[0].message.content

    try:
        gpt_output = [json.loads(raw_content)]
        print("DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print("WARNING [OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        parsed = extract_json_from_text_with_reasoning(raw_content)
        if parsed is not None:
            gpt_output = [parsed]
            print("DEBUG [OS]: Successfully extracted JSON from text")
        else:
            print("ERROR [OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                "Could not parse JSON from model response.", raw_content, 0
            )

    print(f"gpt_output_mol:{gpt_output}")

    merged = _update_symbols_and_corefs(gpt_output, coref_results)
    result = _update_smiles_and_molfile(merged, _convert_graph_to_smiles)
    print(f"mol_agent_output:{result}")
    return result
