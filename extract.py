import json
import re
from typing import Any, Dict, List
from transformers import pipeline
import requests
import re
from openai import OpenAI
import time
from classify_belief_utils import (
    predict_core_beliefs_hard_gated)

import json
from pathlib import Path


openai_client = OpenAI()

_TRANSCRIPT_GARBAGE_PATTERNS = [
    r"TRANSCRIPT OF AUDIO FILE\s*:?",
    r"BEGIN TRANSCRIPT\s*:?",
    r"END TRANSCRIPT\s*:?",
    r"_{5,}",
]

MAX_CHARS = 16000
MAJOR_THRESHOLD = 0.5
MAJOR_THRESHOLD_WORTHLESS = 0.6
FINE_THRESHOLD = 0.3
MAX_LEN = 512
USE_FAST = False

def clean_therapy_transcript(raw: str) -> str:
    if not raw:
        return ""

    text = raw
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\[\s*\d{2}:\d{2}:\d{2}\s*\]", "", text)

    for pat in _TRANSCRIPT_GARBAGE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"(^|\n)\s*CLIENT\s*:\s*", r"\1C: ", text, flags=re.IGNORECASE)
    text = re.sub(r"(^|\n)\s*THERAPIST\s*:\s*", r"\1T: ", text, flags=re.IGNORECASE)

    text = re.sub(r"(^|\n)\s*Client\s*:\s*", r"\1C: ", text)
    text = re.sub(r"(^|\n)\s*Therapist\s*:\s*", r"\1T: ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _call_gpt4_chat(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = MAX_CHARS,
    temperature: float = 0.2,
    top_p: float = 0.9,
):

    start = time.perf_counter()

    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    elapsed = time.perf_counter() - start

    usage = getattr(resp, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    print(f"time={elapsed:.2f}s in={prompt_tokens} out={completion_tokens} total={total_tokens}")
    content = resp.choices[0].message.content.strip()
    return content



def extract_psychological_evidence(
    raw_description: str,
    raw_transcript: str,
    model: str = "gpt-4o",

) -> str:
    try:
        clean_description = clean_therapy_transcript(raw_description)
    except NameError:
        clean_description = raw_description.strip()
    
    try:
        clean_transcript = clean_therapy_transcript(raw_transcript)
    except NameError:
        clean_transcript = raw_transcript.strip()
    
    prompt = f"""
You are an expert clinical psychologist with strong experience in Cognitive Behavioral Therapy (CBT) Beck style.

TASK:
Write a psychological report based on the DESCRIPTION of the patient and TRANSCRIPTION of SIMILAR (but DIFFERENT) THERAPY session, with the following constraints.

IMPORTANT NOTES: The TRANSCRIPTION is of a DIFFERENT PATIENT with some similarities of the patient in the description. IT MUST NOT be used as factual evidence but as psychological reference only.

STRICT FACTUAL CONSTRAINTS (VERY IMPORTANT):
- The ONLY factual events, situations, and life circumstances you are allowed to mention are those explicitly listed in the DESCRIPTION.
- You MUST NOT introduce any new facts, events, diagnoses, medications, incidents, relationships, or life details that are not present in the DESCRIPTION.
- The TRANSCRIPT DOES NOT describe the same patient and MUST NOT be used as a source of factual information.

USE OF THE TRANSCRIPT:
- The transcript may ONLY be used as a source of psychological patterns and mechanisms
  (e.g., typical emotional reactions, thought styles, behavioral responses, coping tendencies).
- You may use the transcript ONLY to support or justify psychological interpretations of the DESCRIPTION facts.
- Never state or imply that an event described in the transcript happened to the patient.

PSYCHOLOGICAL INTERPRETATION RULES:
- Every psychological interpretation must be explicitly grounded in one or more facts from the DESCRIPTION.
- You may infer emotional states, cognitive tendencies, behavioral responses, and coping styles,
  but ONLY as psychological reactions to the described facts.
- Do NOT speculate about unmentioned traumas, incidents, medical conditions, legal issues, or future plans.

STYLE REQUIREMENTS:
- Write in a CBT case-formulation style similar to Beck-style therapy notes:
  concrete, descriptive, focused on the patient's inner experience (thoughts, emotions, behaviors),
  with minimal theoretical language
- Do NOT list facts from the transcript as if they were patient history.
- Focus on how a person with these DESCRIPTION facts might psychologically experience them.
- Avoid introducing concrete details not present in the DESCRIPTION.

CONTENT GOAL:
Produce a detailed psychological report that:
- Explains the likely emotional, cognitive, and behavioral impacts of the DESCRIPTION facts.
- When describing behaviors or coping, use concrete action verbs
  (e.g., withdraws, avoids, ruminates, procrastinates, seeks reassurance, fallbacks, scrolls social media,
  calls for support...).
- When describing emotions, prefer clustered, everyday labels similar to:
  "anxious, worried, fearful, scared, tense";
  "sad, down, lonely, unhappy";
  "ashamed, embarrassed, humiliated";
  "angry, mad, irritated, annoyed";
  "guilty", "disappointed", "hurt".
- Uses the TRANSCRIPT only as indirect psychological support for interpreting these impacts
- Remains strictly faithful to the DESCRIPTION as the sole source of facts

DESCRIPTION:
\"\"\"{clean_description}\"\"\"

TRANSCRIPT (for psychological reference only, NOT factual evidence):
\"\"\"{clean_transcript}\"\"\"
""".strip()

    raw = _call_gpt4_chat(
        prompt=prompt,
        model=model,
        max_tokens=16000,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    
    print("This is the report" , raw.strip())
    return raw.strip()


def extract_psychological_evidence_random(
    raw_description: str,
    raw_transcript: str,
    model: str = "gpt-4o",

) -> str:
    try:
        clean_description = clean_therapy_transcript(raw_description)
    except NameError:
        clean_description = raw_description.strip()
    
    try:
        clean_transcript = clean_therapy_transcript(raw_transcript)
    except NameError:
        clean_transcript = raw_transcript.strip()
    
    prompt2 = f"""
You are an expert clinical psychologist with strong experience in Cognitive Behavioral Therapy (CBT).

TASK:
Write ONE single continuous psychological report (no headings, no bullet points, no sections, no labels).
The report must integrate BOTH:
- DESCRIPTION (background history + stated current situation)
- TRANSCRIPT (what the client actually talks about, reacts to, repeats, avoids, and the session-specific themes)

CRITICAL INTEGRATION RULES:
1) HISTORY and the stated SITUATION must come from the DESCRIPTION, but you must weave them into the narrative naturally (do not present them as a separate section).
2) For every paragraph you write, you MUST include:
   - at least ONE concrete element from the DESCRIPTION (history/situation fact), AND
   - at least ONE concrete element from the TRANSCRIPT (session content, themes, reactions, behaviors, or client wording).
   Do not write any paragraph that can be supported by only one source.
3) Use the TRANSCRIPT heavily: it should contribute most of the content about emotions, automatic thoughts, cognitive patterns, behavioral responses, coping strategies, and interpersonal dynamics.

EVIDENCE / ANCHOR REQUIREMENT (must be in-line):
- You must include 6–10 short verbatim quotes from the TRANSCRIPT (max 8 words each),
  embedded naturally inside sentences using quotation marks.
- Spread them across the report. Do NOT group them in a separate list.

STYLE REQUIREMENTS:
- Beck-style CBT case formulation tone: concrete, descriptive, low jargon.
- Focus on the client's moment-to-moment experience: triggers → thoughts → emotions → behaviors → coping.
- Use everyday emotion clusters (anxious/worried/tense; sad/down/lonely; ashamed/humiliated; angry/irritated; guilty/hurt).
- Use concrete action verbs (avoids, withdraws, ruminates, procrastinates, checks, seeks reassurance, scrolls, argues, shuts down, numbs).

OUTPUT LENGTH:
- One continuous text block.

DESCRIPTION:
\"\"\"{clean_description}\"\"\"

TRANSCRIPT:
\"\"\"{clean_transcript}\"\"\"

""".strip()

    print("Prompt length:", len(prompt2))

    raw = _call_gpt4_chat(
        prompt=prompt2,
        model=model,
        max_tokens=MAX_CHARS,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    
    print("This is the report" , raw.strip())
    return raw.strip()




def ccd_from_psychological_evidence(
    report: str,
    description: str,
    model: str = "gpt-4o",
    max_chars: int = MAX_CHARS,
) -> dict:

    prompt = f"""
You are a senior CBT clinician and case formulator (Beck-style).

You are given:
1) A DESCRIPTION of concrete facts about the patient case. 
2) A complete REPORT with facts and psychological evidence about a trascription of a SIMILAR patient case but not the SAME. To use as primary source of knowledge. 


YOUR TASK:
Construct a CBT Cognitive Conceptualization Model (CCD-like) for the CLIENT, as if YOU (the therapist)
are asking yourself key case-formulation questions using the REPORT. 

ABSOLUTE RULES NON-NEGOTIABLE:
- Use situation as described in the DESCRIPTION.
- ADD to every fields aspects from the REPORT even if it don't fit with the situation/history.
- Psychological evidence means: the client’s reported thoughts, emotions, behaviors, patterns, or strongly implied themes — not generic clinical theory.
- You MUST keep the JSON schema provided.

BECK-STYLE SELF-QUESTIONS (use transcript evidence to answer internally before producing JSON):
- What is the main real-life triggering situation the client is dealing with?
- When the client deals with the situation, what was going through their mind (hot thought)?
- What did that thought mean about them / others / the future?
- What emotions followed, and what did they do next (behavior)?
- What rule/assumption would generate that thought in that situation (intermediate belief)?
- What coping strategies keep the cycle going?



DEFINITIONS:

1) situation:
  Real-life triggering context. DO NOT CHANGE IT, even if you think you could refine it.

2) auto_thought:
- Definition: an AUTOMATIC THOUGHT is a  “hot thought” (2 linked sentences) in first person, present/near-future, that pops up immediately in that moment as a response to the SITUATION.
[IMPORTANT] - It must be tied ONLY to the specific situation (in the CCD situation field) and MUST include core theme from the situation.
- It is an interpretation / judgment / prediction / meaning, not a fact and not a summary of emotions.
- Use the patient’s inner voice (can be distorted, absolute, catastrophic); no therapy language, no explanations, no coping strategies.


3) emotion:
   - Definition: the emotional reaction that follows the automatic thought(s).
   - Output 3–7 emotion labels/phrases (some examples: "anxious, worried, fearful, tense, sad, down, lonely, unhappy, angry, irritated, ashamed, embarrassed, humiliated, guilty, disappointed, hurt, jealous, envious, ...").


4) behavior:
   - Definition: the behavioral concrete reaction response in response to the SITUATION
     (including avoidance, safety behaviors, reassurance seeking, substance use, conflict behaviors).
   - Avoid future plans or general patterns, but focus only on concrete actions done by the client in response to the SITUATION.
   - They are CONCRETE short actions that the client does in response to the SITUATION, so don't describe feelings or general patterns.
   - It is specific to the SITUATION and linked to the thought/emotion chain, to talk about the SITUATION.
   - 2–3 short sentences that describe a concrete behavior, to the situation / auto-thought

5) core beliefs:
   - Definition: core beliefs are deeply held, global, rigid, overgeneralized beliefs about self/others/world.
   - Must be chosen ONLY from the standard menu above.
   - Output in first person. 0–3 items per category.
   
6) intermediate_belief:
   - Definition: intermediate beliefs are rules/assumptions/attitudes derived from core beliefs
     that shape automatic thoughts (often in if–then form).
   - Output 2-3 sentences. That act as a bridge between core beliefs and automatic thoughts.
   - They MUST connect a selected core belief to the auto_thought, don't be shorty and don't copy the auto thought.

7) intermediate_belief_depression:
   - Definition: a depression/hopelessness-leaning variant of the intermediate belief
      if a depressed/hopeless pattern is evident in the session.

8) history:
   - Definition: relevant background the client reports that helps explain the development/maintenance
     of the belief system and coping patterns.
   - 2–6 sentences. Only what is clearly reported or strongly implied.

9) coping_strategies:
   - Definition: recurring strategies the client uses to manage distress (adaptive or maladaptive),
     especially those that maintain problems (avoidance, substance use, withdrawal, overcontrol, etc.).
   - 2–3 sentences. Concrete patterns, not generic.

10) rationale:
   - Cite the exact phrase of the transcript that JUSTIFIES your choice for each field above. 
   REMEMBER: use ONLY psychological evidence from the transcript, not concrete facts.

    
OUTPUT:
Return ONLY valid JSON, with EXACTLY the following keys and structure (no extra keys):


DATA YOU RECEIVE:
1) DESCRIPTION (concrete facts):
\"\"\"{description}\"\"\"

2) REPORT:
\"\"\"{report}\"\"\"




JSON SCHEMA:

{{
  "name": "<string>",
  "id": "<string>",
  "type": ["<string>", "..."],
  "history": "<string>",
  "helpless_belief": ["<string>", "..."],
  "unlovable_belief": ["<string>", "..."],
  "worthless_belief": ["<string>", "..."],
  "intermediate_belief": "<string>",
  "intermediate_belief_depression": "<string>",
  "coping_strategies": "<string>",
  "situation": "<string>",
  "auto_thought": "<string>",
  "emotion": ["<string>", "..."],
  "behavior": "<string>",
  "rationale": "<string>"
}}
""".strip()
    

    
    
    print("len(prompt) =", len(prompt))
    print("len(summary_text) =", len(report))
    
    raw = _call_gpt4_chat(
        prompt=prompt,
        model=model,
        max_tokens=MAX_CHARS,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    def _ensure_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    refined = {
        "name": str(obj.get("name", "") or "").strip(),
        "id": str(obj.get("id", "") or "").strip(),
        "type": _ensure_list(obj.get("type", [])),
        "history": str(obj.get("history", "") or "").strip(),
        "helpless_belief": _ensure_list(obj.get("helpless_belief", []))[:3],
        "unlovable_belief": _ensure_list(obj.get("unlovable_belief", []))[:3],
        "worthless_belief": _ensure_list(obj.get("worthless_belief", []))[:3],
        "intermediate_belief": str(obj.get("intermediate_belief", "") or "").strip(),
        "intermediate_belief_depression": str(obj.get("intermediate_belief_depression", "") or "").strip(),
        "coping_strategies": str(obj.get("coping_strategies", "") or "").strip(),
        "situation": str(obj.get("situation", "") or "").strip(),
        "auto_thought": str(obj.get("auto_thought", "") or "").strip(),
        "emotion": _ensure_list(obj.get("emotion", []))[:6],
        "behavior": str(obj.get("behavior", "") or "").strip(),
        "rationale": str(obj.get("rationale", "") or "").strip(),
    }

    try:
        preds = predict_core_beliefs_hard_gated(
            situation=refined.get("situation", ""),
            auto_thought=refined.get("auto_thought", ""),
            intermediate_belief=refined.get("intermediate_belief", ""),
            major_threshold=MAJOR_THRESHOLD,
            fine_threshold=FINE_THRESHOLD,
            major_threshold_worthless=MAJOR_THRESHOLD_WORTHLESS,
            max_len=MAX_LEN,
            use_fast=USE_FAST,
        )

        old_helpless = refined["helpless_belief"]
        old_unlovable = refined["unlovable_belief"]
        old_worthless = refined["worthless_belief"]

        refined["helpless_belief"] = preds["helpless_belief"]
        refined["unlovable_belief"] = preds["unlovable_belief"]
        refined["worthless_belief"] = preds["worthless_belief"]

        print(
            f"\n[refine_ccd_with_transcript] Core Beliefs (before):\n"
            f"  helpless={old_helpless}\n"
            f"  unlovable={old_unlovable}\n"
            f"  worthless={old_worthless}\n"
            f"→ (after):\n"
            f"  helpless={refined['helpless_belief']} {preds['probs']['helpless']}\n"
            f"  unlovable={refined['unlovable_belief']} {preds['probs']['unlovable']}\n"
            f"  worthless={refined['worthless_belief']} {preds['probs']['worthless']}\n"
            f"Major active: {preds['major_active']}"
        )
    except NameError:
        pass

    return refined



def ccd_from_psychological_evidence_random(
    report: str,
    description: str,
    model: str = "gpt-4o",
    max_chars: int = MAX_CHARS,
) -> dict:
    """
    Valida e adatta un CCD ipotetico (derivato da descrizione) usando la TRASCRIZIONE completa.

    - NON modifica il campo `situation` (rimane quello ipotizzato dalla descrizione).
    - Può modificare / raffinare tutti gli altri campi, usando:
        - la trascrizione come evidenza principale;
    - Aggiorna anche i livelli di confidenza dopo aver integrato la trascrizione.
    - Ricalcola i core beliefs con `predict_core_beliefs_hard_gated` se disponibile.
    - Restituisce anche un `diff_report` che sintetizza cosa è cambiato e con quali confidence.
    - Include in `meta["rationales"]` brevi giustificazioni testuali per i campi chiave
      (es. auto_thought, core beliefs, coping, behavior).

    Ritorna un dict con la stessa struttura del CCD ipotetico, ma validato, con un campo extra:
      - diff_report: dict per confronto prima/dopo.
    """



    prompt = f"""
You are a senior CBT clinician and case formulator (Beck-style).

You are given:
1) A DESCRIPTION of concrete facts about the patient case. 
2) A complete REPORT with facts and psychological evidence about a trascription of a SIMILAR patient case but not the SAME. To use as primary source of knowledge. 


YOUR TASK:
Construct a CBT Cognitive Conceptualization Model (CCD-like) for the CLIENT, as if YOU (the therapist)
are asking yourself key case-formulation questions and answering them using the REPORT. 

ABSOLUTE RULES NON-NEGOTIABLE:
- If the fact of the REPORT contradicts the description, ALWAYS trust the REPORT.
- Use situation as described in the DESCRIPTION.
- IMPORTANT: Rely on the REPORT to fill in all other fields.
- Psychological evidence means: the client’s reported thoughts, emotions, behaviors, patterns, or strongly implied themes — NOT generic clinical theory but very specific for this client.
- You MUST keep the JSON schema provided.

BECK-STYLE SELF-QUESTIONS (use transcript evidence to answer internally before producing JSON):
- What is the main real-life triggering situation the client is dealing with?
- When the client deals with the situation, what was going through their mind (hot thought)?
- What did that thought mean about them / others / the future?
- What emotions followed, and what did they do next (behavior)?
- What rule/assumption would generate that thought in that situation (intermediate belief)?
- What coping strategies keep the cycle going?



DEFINITIONS:

1) situation:
  Real-life triggering context. DO NOT CHANGE IT, even if you think you could refine it.

2) auto_thought:
- Definition: an AUTOMATIC THOUGHT is a  “hot thought” (2 linked sentences) in first person, present/near-future, that pops up immediately in that moment as a response to the SITUATION.
[IMPORTANT] - It must be tied ONLY to the specific situation (in the CCD situation field) and MUST include core theme from the situation.
- It is an interpretation / judgment / prediction / meaning, not a fact and not a summary of emotions.
- Use the patient’s inner voice (can be distorted, absolute, catastrophic); no therapy language, no explanations, no coping strategies.


3) emotion:
   - Definition: the emotional reaction that follows the automatic thought(s).
   - Output 3–7 emotion labels/phrases (some examples: "anxious, worried, fearful, tense, sad, down, lonely, unhappy, angry, irritated, ashamed, embarrassed, humiliated, guilty, disappointed, hurt, jealous, envious, ...").


4) behavior:
   - Definition: the behavioral concrete reaction response in response to the SITUATION
     (including avoidance, safety behaviors, reassurance seeking, substance use, conflict behaviors).
   - Avoid future plans or general patterns, but focus only on concrete actions done by the client in response to the SITUATION.
   - They are CONCRETE short actions that the client does in response to the SITUATION, so don't describe feelings or general patterns.
   - It is specific to the SITUATION and linked to the thought/emotion chain, to talk about the SITUATION.
   - 2–3 short sentences that describe a concrete behavior, to the situation / auto-thought

5) core beliefs:
   - Definition: core beliefs are deeply held, global, rigid, overgeneralized beliefs about self/others/world.
   - Must be chosen ONLY from the standard menu above.
   - Output in first person. 0–3 items per category.
   
6) intermediate_belief:
   - Definition: intermediate beliefs are rules/assumptions/attitudes derived from core beliefs
     that shape automatic thoughts (often in if–then form).
   - Output 2-3 sentences. That act as a bridge between core beliefs and automatic thoughts.
   - They MUST connect a selected core belief to the auto_thought, don't be shorty and don't copy the auto thought.

7) intermediate_belief_depression:
   - Definition: a depression/hopelessness-leaning variant of the intermediate belief
      if a depressed/hopeless pattern is evident in the session.

8) history:
   - Definition: relevant background the client reports that helps explain the development/maintenance
     of the belief system and coping patterns.
   - 2–6 sentences. Only what is clearly reported or strongly implied.

9) coping_strategies:
   - Definition: recurring strategies the client uses to manage distress (adaptive or maladaptive),
     especially those that maintain problems (avoidance, substance use, withdrawal, overcontrol, etc.).
   - 2–3 sentences. Concrete patterns, not generic.

10) rationale:
   - Cite the exact phrase of the transcript that JUSTIFIES your choice for each field above. 
   REMEMBER: use ONLY psychological evidence from the transcript, not concrete facts.

    
OUTPUT:
Return ONLY valid JSON, with EXACTLY the following keys and structure (no extra keys):


DATA YOU RECEIVE:
1) DESCRIPTION (concrete facts):
\"\"\"{description}\"\"\"

2) REPORT:
\"\"\"{report}\"\"\"




JSON SCHEMA:

{{
  "name": "<string>",
  "id": "<string>",
  "type": ["<string>", "..."],
  "history": "<string>",
  "helpless_belief": ["<string>", "..."],
  "unlovable_belief": ["<string>", "..."],
  "worthless_belief": ["<string>", "..."],
  "intermediate_belief": "<string>",
  "intermediate_belief_depression": "<string>",
  "coping_strategies": "<string>",
  "situation": "<string>",
  "auto_thought": "<string>",
  "emotion": ["<string>", "..."],
  "behavior": "<string>",
  "rationale": "<string>"
}}
""".strip()
    

    
    print("I'm using random extraction")
    print("len(prompt) =", len(prompt))
    print("len(summary_text) =", len(report))
    
    raw = _call_gpt4_chat(
        prompt=prompt,
        model=model,
        max_tokens=MAX_CHARS,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    def _ensure_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    refined = {
        "name": str(obj.get("name", "") or "").strip(),
        "id": str(obj.get("id", "") or "").strip(),
        "type": _ensure_list(obj.get("type", [])),
        "history": str(obj.get("history", "") or "").strip(),
        "helpless_belief": _ensure_list(obj.get("helpless_belief", []))[:3],
        "unlovable_belief": _ensure_list(obj.get("unlovable_belief", []))[:3],
        "worthless_belief": _ensure_list(obj.get("worthless_belief", []))[:3],
        "intermediate_belief": str(obj.get("intermediate_belief", "") or "").strip(),
        "intermediate_belief_depression": str(obj.get("intermediate_belief_depression", "") or "").strip(),
        "coping_strategies": str(obj.get("coping_strategies", "") or "").strip(),
        "situation": str(obj.get("situation", "") or "").strip(),
        "auto_thought": str(obj.get("auto_thought", "") or "").strip(),
        "emotion": _ensure_list(obj.get("emotion", []))[:6],
        "behavior": str(obj.get("behavior", "") or "").strip(),
        "rationale": str(obj.get("rationale", "") or "").strip(),
    }

    try:
        preds = predict_core_beliefs_hard_gated(
            situation=refined.get("situation", ""),
            auto_thought=refined.get("auto_thought", ""),
            intermediate_belief=refined.get("intermediate_belief", ""),
            major_threshold=MAJOR_THRESHOLD,
            fine_threshold=FINE_THRESHOLD,
            major_threshold_worthless=MAJOR_THRESHOLD_WORTHLESS,
            max_len=MAX_LEN,
            use_fast=USE_FAST,
        )

        old_helpless = refined["helpless_belief"]
        old_unlovable = refined["unlovable_belief"]
        old_worthless = refined["worthless_belief"]

        refined["helpless_belief"] = preds["helpless_belief"]
        refined["unlovable_belief"] = preds["unlovable_belief"]
        refined["worthless_belief"] = preds["worthless_belief"]

        print(
            f"\n[refine_ccd_with_transcript] Core Beliefs (before):\n"
            f"  helpless={old_helpless}\n"
            f"  unlovable={old_unlovable}\n"
            f"  worthless={old_worthless}\n"
            f"→ (after):\n"
            f"  helpless={refined['helpless_belief']} {preds['probs']['helpless']}\n"
            f"  unlovable={refined['unlovable_belief']} {preds['probs']['unlovable']}\n"
            f"  worthless={refined['worthless_belief']} {preds['probs']['worthless']}\n"
            f"Major active: {preds['major_active']}"
        )
    except NameError:
        pass

    return refined




def gpt4_cbt(
    raw_description: str,
    model: str = "gpt-4o-mini",
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """
    Estrae un CBT/CCD-like model da una descrizione e same transcript.

    Output schema (NO extra keys):
    {
      "name": str,
      "id": str,
      "type": [str, ...],
      "history": str,
      "helpless_belief": [str, ...],
      "unlovable_belief": [str, ...],
      "worthless_belief": [str, ...],
      "intermediate_belief": str,
      "intermediate_belief_depression": str,
      "coping_strategies": str,
      "situation": str,
      "auto_thought": str,
      "emotion": [str, ...],
      "behavior": str,
    }
    """

    def _ensure_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    def _ensure_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()

    if not raw_description or not raw_description.strip():
        return {
            "name": "",
            "id": "",
            "type": [],
            "history": "",
            "helpless_belief": [],
            "unlovable_belief": [],
            "worthless_belief": [],
            "intermediate_belief": "",
            "intermediate_belief_depression": "",
            "coping_strategies": "",
            "situation": "",
            "auto_thought": "",
            "emotion": [],
            "behavior": ""
        }

    clean = raw_description.strip()
    if len(clean) > max_chars:
        clean = clean[:max_chars]

    CORE_BELIEF_MENU = """
HELPlESS (ineffective / vulnerable):
- I am helpless
- I am trapped
- I am defective
- I am incompetent
- I am powerless, weak, vulnerable
- I am a failure, loser

UNLOVABLE (unwanted / rejected):
- I am unlovable
- I am undesirable, unwanted
- I am unattractive
- I am bound to be rejected
- I am bound to be abandoned
- I am bound to be alone

WORTHLESS (bad / defective / dangerous / immoral):
- I am bad - dangerous, toxic, evil
- I am worthless, waste
- I am immoral
- I don't deserve to live
""".strip()

    prompt = f"""

You are given ONE  patient case description.
Your job is to produce a CBT Cognitive Conceptualization Diagram (CCD-like) by reading a the DESCRIPTION.

You can infer the psychological aspects of the patient, from your knowledge but without changing the real facts of the case.
As history and situation refer only to the case description.
The other fields can refer to psychological evidence from the similar case transcription.
STRICT RULES: 
- Core beliefs MUST be chosen ONLY from the menu below (do not invent new wordings): you can choose as many core beliefs (from the 3 major categories)
and fill them with the sub-beliefs. You are not limited to fill all the slots if not needed.
- Return ONLY valid JSON that matches the schema exactly (no extra keys).

CORE BELIEF MENU (strict):
{CORE_BELIEF_MENU}


JSON SCHEMA (exact):
{{
  "name": "<string>",
  "id": "<string>",
  "type": ["<string>", "..."],
  "history": "<string>",
  "helpless_belief": ["<string>", "..."],
  "unlovable_belief": ["<string>", "..."],
  "worthless_belief": ["<string>", "..."],
  "intermediate_belief": "<string>",
  "intermediate_belief_depression": "<string>",
  "coping_strategies": "<string>",
  "situation": "<string>",
  "auto_thought": "<string>",
  "emotion": ["<string>", "..."],
  "behavior": "<string>"
}}

CASE DESCRIPTION:
\"\"\"{clean}\"\"\"
""".strip()

    raw = _call_gpt4_chat(
        prompt=prompt,
        model=model,
        max_tokens=16000,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    out = {
        "name": _ensure_str(obj.get("name", "")),
        "id": _ensure_str(obj.get("id", "")),
        "type": _ensure_list(obj.get("type", [])),
        "history": _ensure_str(obj.get("history", "")),
        "helpless_belief": _ensure_list(obj.get("helpless_belief", []))[:3],
        "unlovable_belief": _ensure_list(obj.get("unlovable_belief", []))[:3],
        "worthless_belief": _ensure_list(obj.get("worthless_belief", []))[:3],
        "intermediate_belief": _ensure_str(obj.get("intermediate_belief", "")),
        "intermediate_belief_depression": _ensure_str(obj.get("intermediate_belief_depression", "")),
        "coping_strategies": _ensure_str(obj.get("coping_strategies", "")),
        "situation": _ensure_str(obj.get("situation", "")),
        "auto_thought": _ensure_str(obj.get("auto_thought", "")),
        "emotion": _ensure_list(obj.get("emotion", []))[:6],
        "behavior": _ensure_str(obj.get("behavior", "")),
    }

    return out


def gpt4_cbt_w_same_transcription(
    raw_description: str,
    raw_transcript: str,
    model: str = "gpt-4o-mini",
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """
    Estrae un CBT/CCD-like model SOLO da una descrizione e una TRASCRIZIONE di terapia simile.

    Output schema (NO extra keys):
    {
      "name": str,
      "id": str,
      "type": [str, ...],
      "history": str,
      "helpless_belief": [str, ...],
      "unlovable_belief": [str, ...],
      "worthless_belief": [str, ...],
      "intermediate_belief": str,
      "intermediate_belief_depression": str,
      "coping_strategies": str,
      "situation": str,
      "auto_thought": str,
      "emotion": [str, ...],
      "behavior": str,
    }
    """

    def _ensure_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    def _ensure_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()

    if not raw_description or not raw_description.strip():
        return {
            "name": "",
            "id": "",
            "type": [],
            "history": "",
            "helpless_belief": [],
            "unlovable_belief": [],
            "worthless_belief": [],
            "intermediate_belief": "",
            "intermediate_belief_depression": "",
            "coping_strategies": "",
            "situation": "",
            "auto_thought": "",
            "emotion": [],
            "behavior": "",
        }

    clean_description = raw_description.strip()
    if len(clean_description) > max_chars:
        clean_description = clean_description[:max_chars]

    clean_transcript = clean_therapy_transcript(raw_transcript)
    CORE_BELIEF_MENU = """
HELPlESS (ineffective / vulnerable):
- I am helpless
- I am trapped
- I am defective
- I am incompetent
- I am powerless, weak, vulnerable
- I am a failure, loser

UNLOVABLE (unwanted / rejected):
- I am unlovable
- I am undesirable, unwanted
- I am unattractive
- I am bound to be rejected
- I am bound to be abandoned
- I am bound to be alone

WORTHLESS (bad / defective / dangerous / immoral):
- I am bad - dangerous, toxic, evil
- I am worthless, waste
- I am immoral
- I don't deserve to live
""".strip()

    prompt = f"""

You are given ONE pre-therapy case description AND a transcription of a therapy session of a similar case.
Your job is to produce a CBT Cognitive Conceptualization Diagram (CCD-like) by reading a the DESCRIPTION and the transcription of a THERAPY SESSION with a SIMILAR patient.

To understand the psychological aspects of the patient, you can use, the transcription.
As history and situation refer only to the case description.
The other fileds can refer to psychological evidence from the similar case transcription.

STRICT RULES: 
- Core beliefs MUST be chosen ONLY from the menu below (do not invent new wordings): you can choose as many core beliefs (from the 3 major categories)
and fill them with the sub-beliefs. You are not limited to fill all the slots if not needed.
- Return ONLY valid JSON that matches the schema exactly (no extra keys).

CORE BELIEF MENU (strict):
{CORE_BELIEF_MENU}


JSON SCHEMA (exact):
{{
  "name": "<string>",
  "id": "<string>",
  "type": ["<string>", "..."],
  "history": "<string>",
  "helpless_belief": ["<string>", "..."],
  "unlovable_belief": ["<string>", "..."],
  "worthless_belief": ["<string>", "..."],
  "intermediate_belief": "<string>",
  "intermediate_belief_depression": "<string>",
  "coping_strategies": "<string>",
  "situation": "<string>",
  "auto_thought": "<string>",
  "emotion": ["<string>", "..."],
  "behavior": "<string>",
}}


TRANSCRIPTION:
\"\"\"{clean_transcript}\"\"\"

CASE DESCRIPTION:
\"\"\"{clean_description}\"\"\"
""".strip()

    print("len(prompt) =", len(prompt))
    print("len(similar case transcript) =", len(clean_transcript))

    raw = _call_gpt4_chat(
        prompt=prompt,
        model=model,
        max_tokens=16384,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    out = {
        "name": _ensure_str(obj.get("name", "")),
        "id": _ensure_str(obj.get("id", "")),
        "type": _ensure_list(obj.get("type", [])),
        "history": _ensure_str(obj.get("history", "")),
        "helpless_belief": _ensure_list(obj.get("helpless_belief", []))[:3],
        "unlovable_belief": _ensure_list(obj.get("unlovable_belief", []))[:3],
        "worthless_belief": _ensure_list(obj.get("worthless_belief", []))[:3],
        "intermediate_belief": _ensure_str(obj.get("intermediate_belief", "")),
        "intermediate_belief_depression": _ensure_str(obj.get("intermediate_belief_depression", "")),
        "coping_strategies": _ensure_str(obj.get("coping_strategies", "")),
        "situation": _ensure_str(obj.get("situation", "")),
        "auto_thought": _ensure_str(obj.get("auto_thought", "")),
        "emotion": _ensure_list(obj.get("emotion", []))[:6],
        "behavior": _ensure_str(obj.get("behavior", "")),
    }

    return out


def gpt4_cbt_w_random_transcription(
    raw_description: str,
    raw_transcript: str,
    model: str = "gpt-4o",
    max_chars: int = 16000,
) -> Dict[str, Any]:
    """
    Estrae un CBT/CCD-like model SOLO da una descrizione e una trascrizione scelta casualmente.

    Output schema (NO extra keys):
    {
      "name": str,
      "id": str,
      "type": [str, ...],
      "history": str,
      "helpless_belief": [str, ...],
      "unlovable_belief": [str, ...],
      "worthless_belief": [str, ...],
      "intermediate_belief": str,
      "intermediate_belief_depression": str,
      "coping_strategies": str,
      "situation": str,
      "auto_thought": str,
      "emotion": [str, ...],
      "behavior": str,
    }
    """

    def _ensure_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    def _ensure_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()

    if not raw_description or not raw_description.strip():
        return {
            "name": "",
            "id": "",
            "type": [],
            "history": "",
            "helpless_belief": [],
            "unlovable_belief": [],
            "worthless_belief": [],
            "intermediate_belief": "",
            "intermediate_belief_depression": "",
            "coping_strategies": "",
            "situation": "",
            "auto_thought": "",
            "emotion": [],
            "behavior": "",
        }

    clean_description = raw_description.strip()
    if len(clean_description) > max_chars:
        clean_description = clean_description[:max_chars]

    clean_transcript = clean_therapy_transcript(raw_transcript)

    CORE_BELIEF_MENU = """
HELPlESS (ineffective / vulnerable):
- I am helpless
- I am trapped
- I am defective
- I am incompetent
- I am powerless, weak, vulnerable
- I am a failure, loser

UNLOVABLE (unwanted / rejected):
- I am unlovable
- I am undesirable, unwanted
- I am unattractive
- I am bound to be rejected
- I am bound to be abandoned
- I am bound to be alone

WORTHLESS (bad / defective / dangerous / immoral):
- I am bad - dangerous, toxic, evil
- I am worthless, waste
- I am immoral
- I don't deserve to live
""".strip()

    prompt = f"""

You are given ONE pre-therapy case description AND a transcription of a therapy session.
Your job is to produce a CBT Cognitive Conceptualization Diagram (CCD-like) by reading a the DESCRIPTION and the TRANSCRIPTION of a THERAPY SESSION.

To understand the psychological aspects of the patient, you can use, the transcription.
As history and situation refer only to the case description.
The other fileds can refer to psychological evidence from the case transcription.

STRICT RULES: 
- Core beliefs MUST be chosen ONLY from the menu below (do not invent new wordings): you can choose as many core beliefs (from the 3 major categories)
and fill them with the sub-beliefs. You are not limited to fill all the slots if not needed.
- Return ONLY valid JSON that matches the schema exactly (no extra keys).

CORE BELIEF MENU (strict):
{CORE_BELIEF_MENU}


JSON SCHEMA (exact):
{{
  "name": "<string>",
  "id": "<string>",
  "type": ["<string>", "..."],
  "history": "<string>",
  "helpless_belief": ["<string>", "..."],
  "unlovable_belief": ["<string>", "..."],
  "worthless_belief": ["<string>", "..."],
  "intermediate_belief": "<string>",
  "intermediate_belief_depression": "<string>",
  "coping_strategies": "<string>",
  "situation": "<string>",
  "auto_thought": "<string>",
  "emotion": ["<string>", "..."],
  "behavior": "<string>",
}}


TRANSCRIPTION:
\"\"\"{clean_transcript}\"\"\"

CASE DESCRIPTION:
\"\"\"{clean_description}\"\"\"
""".strip()

    print("len(prompt) =", len(prompt))
    print("len(similar case transcript) =", len(clean_transcript))

    raw = _call_gpt4_chat(
        prompt=prompt,
        model=model,
        max_tokens=16000,
        temperature=0.2,
        top_p=0.9,
    ).strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    out = {
        "name": _ensure_str(obj.get("name", "")),
        "id": _ensure_str(obj.get("id", "")),
        "type": _ensure_list(obj.get("type", [])),
        "history": _ensure_str(obj.get("history", "")),
        "helpless_belief": _ensure_list(obj.get("helpless_belief", []))[:3],
        "unlovable_belief": _ensure_list(obj.get("unlovable_belief", []))[:3],
        "worthless_belief": _ensure_list(obj.get("worthless_belief", []))[:3],
        "intermediate_belief": _ensure_str(obj.get("intermediate_belief", "")),
        "intermediate_belief_depression": _ensure_str(obj.get("intermediate_belief_depression", "")),
        "coping_strategies": _ensure_str(obj.get("coping_strategies", "")),
        "situation": _ensure_str(obj.get("situation", "")),
        "auto_thought": _ensure_str(obj.get("auto_thought", "")),
        "emotion": _ensure_list(obj.get("emotion", []))[:6],
        "behavior": _ensure_str(obj.get("behavior", "")),
    }

    return out



def main():
    description = """History (facts):
- substance abuse treatment in the past
- long-term obesity
- childhood bullying
- rehab participation

Situation (facts):
- overslept on workday
- insurance payments stopped
- considering not going to work""".strip()
    cbt = gpt4_cbt(description, model="gpt-4o")
    print(json.dumps(cbt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
