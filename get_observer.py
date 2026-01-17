import base64
import json
import os
from typing import Any, List

from openai import AzureOpenAI


API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-06-01",
    azure_endpoint=AZURE_ENDPOINT,
)

PLAN_PROMPT_TEMPLATE = """System Message: 
You are a plan observer. Given the graphic and the current list of agent calls (plan), decide whether the plan is sufficient.

User Message: 
Given the graphic and the current list of agent calls (plan), please recheck the component of the graphic and decide whether the corresponding agent is called and the plan is sufficient.
Valid agents:
1. Reaction template parsing agent
Parses the reaction scheme to identify reactants, products, and label mappings, and outputs a structured reaction template.

2. Molecular recognition agent
Detects other molecules in the graphic except in the reaction template, recognizes their structures, and returns normalized representations (e.g., SMILES, labels, positions).

3. Structure-based R-group substitution agent
Uses structure panels / variant images to extract R-group values and generate enumerated products from a core scaffold based on structural information.

4. Text-based R-group substitution agent
Reads R-group tables and enumerates products or substituents on top of a given core scaffold using text information.

5. Condition interpretation agent
Extracts and normalizes reaction conditions (catalysts, reagents, solvent, temperature, time, atmosphere, etc.) from the graphic.

6. Text extraction agent
Performs chmical NER and text-based reaction extraction on the text description.

If the plan is acceptable, return the original plan as-is.
If adjustments are required, provide the improved list of agents and briefly explain the changes.

Always respond in valid JSON with the structure:
{
  "list_of_agents": [...],   // final list of agent calls
  "redo": true/false,
  "reason": "If changed is true, give an explanation; otherwise leave blank."
}

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
{
  "redo": true/false,
  "reason": "Provide the reasons when redo is true; otherwise leave blank."
}

Current agent_result (JSON):
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
            model="gpt-5-mini",
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
            model="gpt-5-mini",
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
