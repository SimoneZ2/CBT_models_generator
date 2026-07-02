import os
import re
import json
import csv
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Directory with individual CBT files like 1-2.json, 2-2.json, ...
INPUT_DIR = "outputs_pipeline_random_sit1_2/generated_cbts"  # <-- change to your folder

# NEW: vignettes file (JSON list of objects like your example)
VIGNETTES_JSON_PATH = "vignettes_sit1.json"  # <-- set correct path (see note below)

OUTPUT_CSV_PATH = "outputs_pipeline_random_sit1_2/cbt_ratings_gpt41.csv"
OUTPUT_JSONL_PATH = "outputs_pipeline_random_sit1_2/cbt_ratings_gpt41.jsonl"

MAX_RETRIES = 5
BASE_BACKOFF_SEC = 1.5
MAX_OUTPUT_TOKENS = 400

# -----------------------------
# SYSTEM PROMPT (revised)
# -----------------------------
SYSTEM_PROMPT = """You are an expert CBT supervisor and researcher.

You must evaluate CBT cognitive case formulations for a specific patient case vignette.

Quality is defined as a parsimonious, coherent, and meaningful CBT account of the client’s presenting problems.
A high-quality formulation links CBT elements (situations/triggers, automatic thoughts, emotions/physiology, behaviors, coping/compensatory strategies, underlying assumptions, core beliefs, and developmental/learning history) into plausible maintaining mechanisms across situations or over time, grounded in the provided patient vignette.

The quality scale has four levels:
- Good
- Good enough
- Poor
- Very poor

ANCHORS
- Good enough: includes most relevant information (e.g., developmental factors, core beliefs, assumptions, coping/compensatory strategies) at an appropriate level of detail AND links it to prototypical problematic situations with at least a minimally coherent maintaining story consistent with the vignette.
- Very poor: minimal integration of CBT elements, largely irrelevant content, or clear misunderstanding of CBT/CCD concepts.
- Good: stronger than “good enough” due to particularly insightful integration, clarity of maintaining mechanisms, or superior clinical utility (clear targets for intervention) that fits the vignette well.
- Poor: not “very poor,” but still substantially limited (weak/unclear maintaining mechanisms, patchy links across levels, major gaps or mis-specified CBT elements).

IMPORTANT CONSTRAINTS
- Do NOT evaluate formatting, diagram layout, headings, or whether the CCD structure is followed.
- The structure may be incomplete or unconventional: ignore this. Focus ONLY on the semantic content of the fields.
- Core beliefs can be abstract and not personalized; do not penalize abstraction when conceptually appropriate.
- Parsimony is a virtue: do not reward narrative richness unless it improves conceptual integration.
- Judge primarily:
  1) functional integration among beliefs/assumptions/coping and maintaining situations
  2) clarity and plausibility of maintaining mechanisms
  3) coherence across levels (surface problems ↔ underlying beliefs/history)
  4) consistency with the patient case vignette (do not reward content that is not grounded in the case)

ANTI-BIAS / USE FULL SCALE
- Do not default to “Good enough.” Actively discriminate between adjacent levels.
- If mechanisms are clearly specified and intervention targets are obvious consider “Good.”
- If links are vague, inconsistent, mostly unintegrated, or not grounded in the vignette consider “Poor” or “Very poor.”
"""



RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "cbt_quality_rating",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cbt_id": {"type": "string"},
            "quality_rating": {
                "type": "string",
                "enum": ["Good", "Good enough", "Poor", "Very poor"],
            },
            "supervisory_rationale": {
                "type": "string",
                "description": "2–3 sentences rationale in CBT/CCD terms.",
            },
        },
        "required": ["cbt_id", "quality_rating", "supervisory_rationale"],
    },
    "strict": True,
}


def natural_key(filename: str) -> Tuple:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = re.split(r"(\d+)", base)
    key: List[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def load_cbts_from_dir(dir_path: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(dir_path):
        raise ValueError(f"INPUT_DIR does not exist or is not a directory: {dir_path}")

    files = [
        os.path.join(dir_path, fn)
        for fn in os.listdir(dir_path)
        if fn.lower().endswith(".json")
    ]
    if not files:
        raise ValueError(f"No .json files found in: {dir_path}")

    files.sort(key=natural_key)

    cbts: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        items: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            if any(k in data for k in ("formulation", "cbt_text", "text", "content", "diagram_text", "meta", "id", "cbt_id", "source_id")):
                items = [data]
            else:
                for key in ("items", "cbts", "data"):
                    if key in data and isinstance(data[key], list):
                        items = data[key]
                        break
                if not items:
                    items = [data]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(f"Unrecognized JSON content in file: {fp}")

        file_id = os.path.splitext(os.path.basename(fp))[0]
        for it in items:
            if isinstance(it, dict):
                it.setdefault("_filename_id", file_id)
                cbts.append(it)
            else:
                cbts.append({"_filename_id": file_id, "raw": it})

    return cbts


def load_vignettes_map(path: str) -> Dict[str, str]:
    """
    Returns mapping: source_id (e.g., '1-2') -> vignette_en string
    """
    if not os.path.isfile(path):
        raise ValueError(f"VIGNETTES_JSON_PATH does not exist or is not a file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Vignettes JSON must be a list of objects.")

    m: Dict[str, str] = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        sid = obj.get("source_id")
        vtxt = obj.get("vignette_en")
        if isinstance(sid, str) and sid.strip() and isinstance(vtxt, str) and vtxt.strip():
            m[sid.strip()] = vtxt.strip()

    if not m:
        raise ValueError("No valid vignettes found (need source_id + vignette_en).")

    return m


def extract_id(cbt: Dict[str, Any]) -> str:
    for k in ("id", "source_id", "cbt_id"):
        if k in cbt and isinstance(cbt[k], str) and cbt[k].strip():
            return cbt[k].strip()

    if "meta" in cbt and isinstance(cbt["meta"], dict):
        for k in ("id", "source_id", "cbt_id"):
            if k in cbt["meta"] and isinstance(cbt["meta"][k], str) and cbt["meta"][k].strip():
                return cbt["meta"][k].strip()

    if "_filename_id" in cbt and isinstance(cbt["_filename_id"], str) and cbt["_filename_id"].strip():
        return cbt["_filename_id"].strip()

    raise ValueError("Could not find CBT id in item and no filename fallback.")


def format_cbt_for_model(cbt: Dict[str, Any]) -> str:
    for k in ("formulation", "cbt_text", "text", "content", "diagram_text"):
        if k in cbt and isinstance(cbt[k], str) and cbt[k].strip():
            return cbt[k].strip()

    cbt_clean = dict(cbt)
    cbt_clean.pop("_filename_id", None)
    return json.dumps(cbt_clean, ensure_ascii=False, indent=2)


def call_with_retries(client: OpenAI, **kwargs) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("Empty model response content.")
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                raise
            sleep_s = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
            time.sleep(sleep_s)
    raise last_err or RuntimeError("Unknown error")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)

    cbts = load_cbts_from_dir(INPUT_DIR)
    vignettes_map = load_vignettes_map(VIGNETTES_JSON_PATH)

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv, \
         open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f_jsonl:

        writer = csv.DictWriter(
            f_csv,
            fieldnames=["cbt_id", "quality_rating", "supervisory_rationale"],
        )
        writer.writeheader()

        for idx, cbt in enumerate(cbts, start=1):
            cbt_id = extract_id(cbt)
            cbt_text = format_cbt_for_model(cbt)

            vignette = vignettes_map.get(cbt_id)
            if not vignette:
                vignette = "MISSING VIGNETTE FOR THIS CBT_ID. Rate formulation quality based on internal coherence only, but note mismatch/absence."

            user_prompt = f"""TASK
You will be given:
(1) a PATIENT CASE VIGNETTE (this is the patient's case history + index situation)
(2) a CBT case formulation purportedly describing that patient.

Evaluate how well the CBT formulation explains the patient's presenting problems and maintaining mechanisms GIVEN the vignette.

OUTPUT REQUIREMENTS
- Assign exactly one rating: Good, Good enough, Poor, or Very poor.
- Provide a 2–3 sentence supervisory rationale in CBT/CCD terms.
- Focus ONLY on the semantic content of the fields; do NOT judge formatting or whether the CCD structure is perfectly followed.

CBT_ID: {cbt_id}

PATIENT CASE VIGNETTE:
{vignette}

CBT_FORMULATION:
{cbt_text}
"""

            result = call_with_retries(
                client,
                model=MODEL,
                temperature=0.2,
                top_p=0.9,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": RESPONSE_SCHEMA,
                },
            )

            if result.get("cbt_id") != cbt_id:
                result["cbt_id"] = cbt_id

            writer.writerow(result)
            f_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"[{idx}/{len(cbts)}] {cbt_id} -> {result['quality_rating']}")

    print("\nDone.")
    print(f"CSV:   {OUTPUT_CSV_PATH}")
    print(f"JSONL: {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    main()