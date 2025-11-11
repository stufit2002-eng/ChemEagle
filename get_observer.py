import base64
import json
import os
from typing import Any, List

from openai import AzureOpenAI


API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-06-01",
    azure_endpoint=AZURE_ENDPOINT,
)

PLAN_PROMPT_TEMPLATE = """You are a workflow reviewer. Given the image and the current list of tool calls (plan), decide whether the plan is sufficient.
- If the plan is acceptable, return the original plan as-is.
- If adjustments are required, provide the improved plan and briefly explain the changes.

Always respond in valid JSON with the structure:
{
  "plan": [...],   // final list of tool_calls
  "changed": true/false,
  "reason": "If changed is true, give a short explanation; otherwise leave blank."
}

Current plan (JSON):
{plan_json}
"""

ACTION_PROMPT_TEMPLATE = """You are an execution auditor. Using the image and the current tool execution results (tool_result), decide whether the workflow must be rerun.
- If the outcome is acceptable, return redo=false.
- If issues are found or corrections are needed, return redo=true with a short explanation.

Always respond in valid JSON with the structure:
{
  "redo": true/false,
  "reason": "Provide a short reason when redo is true; otherwise leave blank."
}

Current tool_result (JSON):
{result_json}
"""


def _encode_image(image_path: str) -> str | None:
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("plan", tool_calls)
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
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return bool(parsed.get("redo", False))
    except Exception:
        return False
