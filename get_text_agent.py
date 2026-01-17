from PIL import Image
import pytesseract
from chemrxnextractor import RxnExtractor
from openai import AzureOpenAI
model_dir = "./cre_models_v0.1"
rxn_extractor = RxnExtractor(model_dir)
import json
import torch
from chemiener import ChemNER
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("Ozymandias314/ChemNERCkpt", "best.ckpt")
model2 = ChemNER(ckpt_path, device=torch.device('cpu'))
import base64
import os

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

def merge_sentences(sentences):

    cleaned = [s.strip() for s in sentences if s.strip()]
    paragraph = [" ".join(cleaned)]
    return paragraph


def extract_reactions_from_text_in_image(image_path: str) -> dict:


    model_dir = "./cre_models_v0.1"
    device = "cpu"

    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)
）
    reactions = rxn_extractor.get_reactions([paragraph])

    return reactions 

def NER_from_text_in_image(image_path: str) -> dict:
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)
）
    predictions = model2.predict_strings([paragraph])

    return predictions 




def text_extraction_agent(image_path: str) -> dict:
    """
    Agent that calls two tools:
      1) extract_reactions_from_text_in_image
      2) NER_from_text_in_image
    to perform OCR, reaction extraction, and chemical NER on a single image.
    Returns a merged JSON result.
    """
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

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
System Message: 
You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation.

User Message: 
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
        {"role": "system", "content": "You are an expert assistant for chemical text analysis."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        temperature=0,
        response_format={"type": "json_object"}
    )

    # Execute each requested tool
    tool_calls = response1.choices[0].message.tool_calls
    tool_results_msgs = []
    for call in tool_calls:
        name = call.function.name
        if name == "extract_reactions_from_text_in_image":
            result = extract_reactions_from_text_in_image(image_path)
        elif name == "NER_from_text_in_image":
            result = NER_from_text_in_image(image_path)
        else:
            continue
        tool_results_msgs.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

    # Second API call: pass tool outputs back to GPT for final response
    response2 = client.chat.completions.create(
        model="gpt-4o",
        messages=messages + tool_results_msgs,
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response2.choices[0].message.content)
