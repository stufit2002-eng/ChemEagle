import json
import os
from typing import Any, List, Optional

from openai import AzureOpenAI, OpenAI

from _shared_models import encode_image as _encode_image_shared, get_azure_client, resolve_os_client, retry_api_call


def _get_client():
    """Return the shared Azure client (or None if env vars are not set)."""
    try:
        return get_azure_client()
    except ValueError:
        return None

PLAN_PROMPT_TEMPLATE = """System Message:
You are a plan observer. Given the graphic and the current list of agent calls (plan), decide whether the plan is sufficient.

User Message:
Given the graphic and the current list of agent calls (plan), please recheck the component of the graphic and decide whether the corresponding agent is called and the plan is sufficient.
Valid agents (exactly 5 — do NOT invent any other agent names):
1. Reaction template parsing agent
Parses the reaction scheme to identify reactants, products, and label mappings, and outputs a structured reaction template.

2. Molecular recognition agent
Detects other molecules in the graphic except in the reaction template, recognizes their structures, and returns normalized representations (e.g., SMILES, labels, positions).

3. Structure-based R-group substitution agent
Uses structure panels / variant images to extract R-group values and generate enumerated products from a core scaffold based on structural information.

4. Text-based R-group substitution agent
Reads R-group tables and enumerates products or substituents on top of a given core scaffold using text information.

5. Text extraction agent
Performs chemical NER and text-based reaction extraction on the text description.

Mutual exclusion rule: Text-based R-group substitution agent and Structure-based R-group substitution agent MUST NOT be used at the same time.

If the plan is acceptable, return the original plan as-is.
If adjustments are required, provide the improved list of agents and briefly explain the changes.

CRITICAL — name field values: Each item in list_of_agents must use the "name" field with EXACTLY one of these strings (copy verbatim, no abbreviations):
  "process_reaction_image_with_product_variant_R_group"  ← Structure-based R-group substitution agent
  "process_reaction_image_with_table_R_group"            ← Text-based R-group substitution agent
  "get_full_reaction_template"                           ← Reaction template parsing agent
  "get_multi_molecular_full"                             ← Molecular recognition agent
  "text_extraction_agent"                                ← Text extraction agent

Always respond in valid JSON with the structure:
{{
  "list_of_agents": [
    {{"id": "tool_call_0", "name": "<exact name from list above>", "arguments": {{"image_path": "IMAGE_PATH"}}}},
    ...
  ],
  "redo": true/false,
  "reason": "If changed is true, give an explanation; otherwise leave blank."
}}

Current plan (JSON):
{plan_json}
"""

ACTION_PROMPT_TEMPLATE = """System Message: 
You are an action observer. Your task is too observe the graphic and the current agent output, decide whether the agent must be rerun.

User Message: 
By observing the image and the current agent output, decide whether the agent must be rerun.
The main focus is on whether the SMILES is reasonable and effective. Is the condition or text classification correct?
If the outcome is acceptable, return redo=false.
If issues are found or corrections are needed, return redo=true with a short explanation.

Always respond in valid JSON with the structure:
{{
  "redo": true/false,
  "reason": "Provide the reasons when redo is true; otherwise leave blank."
}}

Current agent_result (JSON):
{result_json}
"""


def _encode_image(image_path: str) -> str | None:
    if not image_path or not os.path.exists(image_path):
        return None
    return _encode_image_shared(image_path)


def plan_observer_agent(image_path: str, tool_calls: List[Any]) -> List[Any]:
    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        client = _get_client()
        if client is None:
            # 没有 API 配置，返回原始计划
            return tool_calls
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(content.lstrip())
            except json.JSONDecodeError:
                parsed = {}
        # Support both "plan" and "list_of_agents" keys for compatibility
        if not isinstance(parsed, dict):
            return tool_calls
        return parsed.get("list_of_agents", parsed.get("plan", tool_calls))
    except Exception:
        return tool_calls


def action_observer_agent(image_path: str, tool_result: Any) -> bool:
    base64_image = _encode_image(image_path)
    result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    prompt = ACTION_PROMPT_TEMPLATE.format(result_json=result_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        client = _get_client()
        if client is None:
            # 没有 API 配置，返回 False（不重做）
            return False
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(content.lstrip())
            except json.JSONDecodeError:
                parsed = {}
        if not isinstance(parsed, dict):
            return False
        return bool(parsed.get("redo", False))
    except Exception:
        return False


def plan_observer_agent_OS(
    image_path: str,
    tool_calls: List[Any],
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Any]:
    """OS version of plan_observer_agent using a cached vLLM/Ollama client."""
    client_os = resolve_os_client(base_url=base_url, api_key=api_key)

    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        # Note: vLLM may not support response_format
        response = retry_api_call(
            client_os.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            # response_format={"type": "json_object"},  # vLLM 可能不支持
        )
        content = response.choices[0].message.content
        
        # Try to parse JSON directly
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                parsed = extract_json_from_text_with_reasoning(content)
                if parsed is None:
                    raise ValueError("Failed to extract JSON from response")
            except (ImportError, ValueError):
                print(f"⚠️ 警告: plan_observer_agent_OS 无法解析 JSON，返回原始计划")
                return tool_calls
        
        # Support both "plan" and "list_of_agents" keys for compatibility
        if not isinstance(parsed, dict):
            return tool_calls
        return parsed.get("list_of_agents", parsed.get("plan", tool_calls))
    except Exception as e:
        print(f"⚠️ 警告: plan_observer_agent_OS 出错: {e}，返回原始计划")
        return tool_calls


def action_observer_agent_OS(
    image_path: str,
    tool_result: Any,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """OS version of action_observer_agent using a cached vLLM/Ollama client."""
    client_os = resolve_os_client(base_url=base_url, api_key=api_key)

    base64_image = _encode_image(image_path)
    result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    prompt = ACTION_PROMPT_TEMPLATE.format(result_json=result_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        # Note: vLLM may not support response_format
        response = retry_api_call(
            client_os.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            # response_format={"type": "json_object"},  # vLLM 可能不支持
        )
        content = response.choices[0].message.content
        
        # Try to parse JSON directly
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                parsed = extract_json_from_text_with_reasoning(content)
                if parsed is None:
                    raise ValueError("Failed to extract JSON from response")
            except (ImportError, ValueError):
                print(f"⚠️ 警告: action_observer_agent_OS 无法解析 JSON，返回 False（不重做）")
                return False
        
        if not isinstance(parsed, dict):
            return False
        return bool(parsed.get("redo", False))
    except Exception as e:
        print(f"⚠️ 警告: action_observer_agent_OS 出错: {e}，返回 False（不重做）")
        return False
