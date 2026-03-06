from __future__ import annotations

import json
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from PIL import Image
import pytesseract
from openai import AzureOpenAI, OpenAI

from _model_lock import CUDA_MODEL_LOCK
from _shared_models import encode_image, get_azure_client, resolve_os_client, retry_api_call

# ── Lazy singletons for text-extraction models ────────────────────────────────
_ner_model = None
_ner_lock = threading.Lock()

def _get_ner_model():
    global _ner_model
    if _ner_model is None:
        with _ner_lock:
            if _ner_model is None:
                from chemiener import ChemNER  # noqa: PLC0415
                _ner_model = ChemNER("./ner.ckpt", device=torch.device("cuda"))
    return _ner_model

_rxn_extractor = None
_rxn_extractor_lock = threading.Lock()

def _get_rxn_extractor():
    global _rxn_extractor
    if _rxn_extractor is None:
        with _rxn_extractor_lock:
            if _rxn_extractor is None:
                from chemrxnextractor import RxnExtractor  # noqa: PLC0415
                _rxn_extractor = RxnExtractor("./cre_models_v0.1")
    return _rxn_extractor


def configure_tesseract():
    """自动检测并配置 Tesseract OCR 可执行文件路径"""
    import sys

    # 如果已经配置过，直接返回
    if hasattr(pytesseract.pytesseract, 'tesseract_cmd') and pytesseract.pytesseract.tesseract_cmd:
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    is_windows = sys.platform == "win32"

    # Linux / macOS paths (ignored on Windows)
    linux_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "/opt/local/bin/tesseract",
    ]

    # Windows paths (ignored on Linux/macOS)
    windows_paths = [
        r"F:\chemeagle\Tesseract-OCR\tesseract.exe",
        os.path.join(script_dir, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.path.dirname(script_dir), "Tesseract-OCR", "tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        r"C:\Users\Administrator\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]

    possible_paths = windows_paths if is_windows else linux_paths

    # 首先尝试从 PATH 中查找
    try:
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            print(f"✓ 从 PATH 中找到 Tesseract: {tesseract_cmd}")
            return
    except Exception:
        pass

    # 如果 PATH 中没有，尝试常见路径
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            pytesseract.pytesseract.tesseract_cmd = normalized_path
            print(f"✓ 找到 Tesseract: {normalized_path}")
            return

    # 如果都没找到，提示用户
    print("⚠️  警告: 未找到 Tesseract OCR 可执行文件")
    print("已尝试的路径:")
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        exists = "✓" if os.path.exists(normalized_path) else "✗"
        print(f"  {exists} {normalized_path}")
    print("\n请执行以下步骤之一:")
    print("1. 确保 Tesseract OCR 已正确安装")
    if is_windows:
        print("2. 或者手动设置路径:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'F:\\chemeagle\\Tesseract-OCR\\tesseract.exe'")
    else:
        print("2. Linux: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")
    raise FileNotFoundError(
        "Tesseract OCR 未安装或不在 PATH 中。"
        "请访问 https://github.com/UB-Mannheim/tesseract/wiki 下载安装。"
    )

# 初始化 Tesseract 配置
configure_tesseract()


def merge_sentences(sentences):
    """
    合并一个句子片段列表为一个连贯的段落字符串。
    """
    # 去除每条片段前后空白，并剔除空串
    cleaned = [s.strip() for s in sentences if s.strip()]
    # 用空格拼接，恢复成完整段落
    paragraph = [" ".join(cleaned)]
    return paragraph


def split_text_into_sentences(text: str) -> list:
    """
    将文本分割成句子，避免文本过长导致的问题。
    使用简单的标点符号分割，保留句子边界。
    """
    # 按句号、问号、感叹号分割，但保留这些标点
    sentences = re.split(r'([.!?]+)', text)
    # 合并标点和前面的文本
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        if sentence:
            result.append(sentence)
    
    # 如果没有找到句子边界，尝试按换行符分割
    if not result:
        result = [line.strip() for line in text.splitlines() if line.strip()]
    
    # 如果还是没有，返回整个文本（但限制长度）
    if not result:
        # 限制单个句子长度，避免超过模型限制
        max_length = 500  # 字符数限制
        if len(text) > max_length:
            # 按空格分割成更小的块
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            result = chunks
        else:
            result = [text]
    
    return result


def extract_reactions_from_text_in_image(image_path: str) -> dict:
    """从化学反应图像中提取文本并识别反应（使用单例 RxnExtractor）。"""
    # 1. OCR 提取文本
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. 将多行文本合并为单段落
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. 将文本分割成句子，避免长度问题
    sentences = split_text_into_sentences(paragraph)

    # 4. 使用单例 RxnExtractor（避免每次调用都重新加载模型）
    rxn_extractor = _get_rxn_extractor()

    # 5. 对每个句子提取反应（避免长度不匹配问题）
    all_reactions = []
    try:
        all_reactions = rxn_extractor.get_reactions(sentences)
    except AssertionError as e:
        print(f"警告: 批量处理失败，尝试逐个句子处理: {e}")
        for sent in sentences:
            try:
                all_reactions.extend(rxn_extractor.get_reactions([sent]))
            except Exception as sent_e:
                print(f"警告: 跳过句子（处理失败）: {sent[:50]}... 错误: {sent_e}")

    return all_reactions

def NER_from_text_in_image(image_path: str) -> dict:
    """OCR + chemical NER using singleton ChemNER model."""
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    with CUDA_MODEL_LOCK:
        predictions = _get_ner_model().predict_strings([paragraph])

    return predictions




def text_extraction_agent(image_path: str) -> dict:
    """
    Agent that calls two tools:
      1) extract_reactions_from_text_in_image
      2) NER_from_text_in_image
    to perform OCR, reaction extraction, and chemical NER on a single image.
    Returns a merged JSON result.

    The two local tools are now executed in parallel when the LLM requests them.
    """
    client = get_azure_client()
    b64_image = encode_image(image_path)

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`(“NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst B27" in text to the molecule B27 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates and oxaziridines was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes were used as precursors of azolium enolates. A bifunctional NHC precatalyst B27 bearing a free hydroxyl group was employed."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "NHC-bound azolium enolates" -> Chemical (Plural/Class)
    * "oxaziridines" -> Chemical (Systematic Name)
    * "Aryl(alkyl)-disubstituted ketenes" -> Chemical (Reactant precursor)
    * "NHC" -> Abbreviation
    * "B27" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "B27" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for B27 from the tool/image context: `CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F`.
    * "azolium enolates" and "oxaziridines" are participating reactants in a [3+2] cycloaddition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates [reactant][MULTIPLE] and oxaziridines [reactant][SYSTEMATIC] was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes [reactant] were used as precursors of azolium enolates. A bifunctional NHC [ABBREVIATION] precatalyst B27 [reagent][IDENTIFIERS][SMILES:CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F] bearing a free hydroxyl group was employed."
}
"""

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    # Note: response_format=json_object is omitted — on o-series/gpt-5-mini it
    # internally enforces temperature=0 which those models reject (HTTP 400).
    response1 = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=tools,
    )

    # Get assistant message with tool calls
    if not response1.choices:
        print("WARNING [Azure]: text_extraction_agent got empty choices on first call, returning {}")
        return {}
    assistant_message = response1.choices[0].message

    # Execute each requested tool
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        # If no tool calls, parse response with the same robust fallback
        raw_content1 = assistant_message.content or ""
        if not raw_content1:
            return {}
        from get_R_group_sub_agent import extract_json_from_text_with_reasoning
        try:
            return json.loads(raw_content1)
        except json.JSONDecodeError:
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw_content1.lstrip())
                return obj
            except json.JSONDecodeError:
                pass
            try:
                result = extract_json_from_text_with_reasoning(raw_content1)
            except Exception:
                result = None
            return result if result is not None else {"content": raw_content1}
    
    # Execute all tool calls in parallel (extract_reactions and NER are independent)
    _TOOL_FN = {
        "extract_reactions_from_text_in_image": extract_reactions_from_text_in_image,
        "NER_from_text_in_image": NER_from_text_in_image,
    }

    def _run_tool(call):
        fn = _TOOL_FN.get(call.function.name)
        if fn is None:
            return None
        return call, fn(image_path)

    tool_results_msgs = []
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as _ex:
        for call, result in _ex.map(_run_tool, tool_calls):
            if result is None:
                continue
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # Second API call: pass tool outputs back to GPT for final response
    messages.append(assistant_message)
    messages.extend(tool_results_msgs)

    response2 = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
    )

    # Parse JSON with fallback (response_format removed to avoid implicit temperature=0)
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
    if not response2.choices:
        print("WARNING [Azure]: text_extraction_agent got empty choices on second call, returning {}")
        return {}
    raw2 = response2.choices[0].message.content or ""
    try:
        return json.loads(raw2)
    except json.JSONDecodeError:
        # "Extra data" case: valid JSON followed by trailing non-JSON text
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw2.lstrip())
            return obj
        except json.JSONDecodeError:
            pass
        # JSON embedded inside reasoning/explanatory text
        try:
            result = extract_json_from_text_with_reasoning(raw2)
        except Exception:
            result = None
        return result if result is not None else {"content": raw2}


def text_extraction_agent_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    client = resolve_os_client(base_url=base_url, api_key=api_key)
    b64_image = encode_image(image_path)

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`("NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst B27" in text to the molecule B27 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates and oxaziridines was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes were used as precursors of azolium enolates. A bifunctional NHC precatalyst B27 bearing a free hydroxyl group was employed."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "NHC-bound azolium enolates" -> Chemical (Plural/Class)
    * "oxaziridines" -> Chemical (Systematic Name)
    * "Aryl(alkyl)-disubstituted ketenes" -> Chemical (Reactant precursor)
    * "NHC" -> Abbreviation
    * "B27" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "B27" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for B27 from the tool/image context: `CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F`.
    * "azolium enolates" and "oxaziridines" are participating reactants in a [3+2] cycloaddition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates [reactant][MULTIPLE] and oxaziridines [reactant][SYSTEMATIC] was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes [reactant] were used as precursors of azolium enolates. A bifunctional NHC [ABBREVIATION] precatalyst B27 [reagent][IDENTIFIERS][SMILES:CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F] bearing a free hydroxyl group was employed."
}
```

"""

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    # Note: vLLM may not support response_format and tools simultaneously
    try:
        response1 = retry_api_call(
            client.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            # response_format={"type": "json_object"},  # vLLM 不支持同时使用 response_format 和 tools
        )
    except Exception as e:
        error_msg = str(e)
        if "tool" in error_msg.lower() or "tool-call" in error_msg.lower():
            print(f"⚠️ 警告: vLLM 不支持工具调用: {e}")
            print("提示: 请重新启动 vLLM 容器，添加以下参数:")
            print("  --enable-auto-tool-choice --tool-call-parser auto")
            print("或者继续使用 Ollama（原生支持工具调用）")
            raise
        else:
            raise

    # Get assistant message with tool calls
    if not response1.choices:
        print("WARNING [OS]: text_extraction_agent got empty choices on first call, returning {}")
        return {}
    assistant_message = response1.choices[0].message

    # Execute each requested tool
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        # If no tool calls, try to parse response directly
        raw_content = assistant_message.content
        if raw_content:
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                try:
                    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                    result = extract_json_from_text_with_reasoning(raw_content)
                    if result is not None:
                        return result
                except ImportError:
                    pass
                return {"content": raw_content}
        return {}
    
    # Execute all tool calls in parallel
    _TOOL_FN = {
        "extract_reactions_from_text_in_image": extract_reactions_from_text_in_image,
        "NER_from_text_in_image": NER_from_text_in_image,
    }

    def _run_tool_os(call):
        fn = _TOOL_FN.get(call.function.name)
        if fn is None:
            return None
        return call, fn(image_path)

    tool_results_msgs = []
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as _ex:
        for item in _ex.map(_run_tool_os, tool_calls):
            if item is None:
                continue
            call, result = item
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # Second API call: pass tool outputs back to GPT for final response
    messages.append(assistant_message)
    messages.extend(tool_results_msgs)

    response2 = retry_api_call(
        client.chat.completions.create,
        5, 3, 2,
        model=model_name,
        messages=messages,
        temperature=0,
    )

    # Parse response (support extracting JSON from text with reasoning)
    if not response2.choices:
        print("WARNING [OS]: text_extraction_agent got empty choices on second call, returning {}")
        return {}
    raw_content = response2.choices[0].message.content
    
    try:
        # First try direct JSON parsing
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from text
        try:
            from get_R_group_sub_agent import extract_json_from_text_with_reasoning
            result = extract_json_from_text_with_reasoning(raw_content)
            if result is not None:
                return result
        except ImportError:
            pass
        
        # If all else fails, return raw content wrapped in dict
        print(f"⚠️ 警告: 无法解析 JSON，返回原始内容")
        print(f"Raw content (last 500 chars):\n{raw_content[-500:]}")
        return {"content": raw_content}

