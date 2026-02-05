import re
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG CLASSIFICATORE
# =========================
MAJOR_DIR = "./cbt_pc_major_deberta_final"
FINE_DIR  = "./cbt_fc_fine_deberta_final"

PAPER_MAPPING = {
    "helpless": [
        "I am incompetent",
        "I am helpless",
        "I am powerless, weak, vulnerable",
        "I am a victim",
        "I am needy",
        "I am trapped",
        "I am out of control",
        "I am a failure, loser",
        "I am defective",
    ],
    "unlovable": [
        "I am unlovable",
        "I am unattractive",
        "I am undesirable, unwanted",
        "I am bound to be rejected",
        "I am bound to be abandoned",
        "I am bound to be alone",
    ],
    "worthless": [
        "I am worthless, waste",
        "I am immoral",
        "I am bad - dangerous, toxic, evil",
        "I don’t deserve to live",
    ],
}

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def norm_label(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_fine_to_major_map(model_fine_id2label: dict) -> dict:
    canon = {}
    for major, fine_list in PAPER_MAPPING.items():
        for fl in fine_list:
            canon[norm_label(fl)] = major

    fine_to_major = {}
    for i, lbl in model_fine_id2label.items():
        key = norm_label(lbl)
        if key in canon:
            fine_to_major[lbl] = canon[key]
    return fine_to_major

# =========================
# CACHING MODELLI
# =========================
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MAJOR_TOK = None
_MAJOR_MODEL = None
_FINE_TOK = None
_FINE_MODEL = None

def _load_major(use_fast: bool = False):
    global _MAJOR_TOK, _MAJOR_MODEL
    if _MAJOR_TOK is None or _MAJOR_MODEL is None:
        _MAJOR_TOK = AutoTokenizer.from_pretrained(MAJOR_DIR, use_fast=use_fast)
        _MAJOR_MODEL = AutoModelForSequenceClassification.from_pretrained(MAJOR_DIR).to(_DEVICE)
        _MAJOR_MODEL.eval()
    return _MAJOR_TOK, _MAJOR_MODEL

def _load_fine(use_fast: bool = False):
    global _FINE_TOK, _FINE_MODEL
    if _FINE_TOK is None or _FINE_MODEL is None:
        _FINE_TOK = AutoTokenizer.from_pretrained(FINE_DIR, use_fast=use_fast)
        _FINE_MODEL = AutoModelForSequenceClassification.from_pretrained(FINE_DIR).to(_DEVICE)
        _FINE_MODEL.eval()
    return _FINE_TOK, _FINE_MODEL

@torch.no_grad()
def _predict_probs(tok, model, text: str, max_len: int = 512):
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(_DEVICE) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits.detach().cpu().numpy()[0]
    probs = sigmoid(logits)

    id2label = model.config.id2label
    id2label = {int(k): v for k, v in id2label.items()}
    return probs, id2label

def build_classifier_text(situation: str, auto_thought: str, intermediate_belief: str) -> str:
    """
    Formato coerente col tuo CLI: testo con tag [SITUATION] e [THOUGHTS]
    """
    situation = (situation or "").strip()
    auto_thought = (auto_thought or "").strip()
    intermediate_belief = (intermediate_belief or "").strip()
    return f"[SITUATION]\n{situation}\n\n[THOUGHTS]\n{auto_thought}, {intermediate_belief}".strip()

def predict_core_beliefs_hard_gated(
    situation: str,
    auto_thought: str,
    intermediate_belief: str,
    major_threshold: float = 0.5,
    major_threshold_worthless: float | None = None,  # NEW
    fine_threshold: float = 0.25,
    max_len: int = 512,
    use_fast: bool = False,
) -> dict:
    """
    Ritorna:
      {
        "helpless_belief": [...],
        "unlovable_belief": [...],
        "worthless_belief": [...],
        "probs": {
            "helpless": [{"label": ..., "prob": ...}, ...],
            "unlovable": [...],
            "worthless": [...],
        },
        "major_active": [...]
      }

    Le liste contengono TUTTE le fine-label sopra soglia DOPO hard-gating
    (nessun limite top-k), ordinate per probabilità decrescente.
    """
    text = build_classifier_text(situation, auto_thought, intermediate_belief)
    if not text.strip():
        return {
            "helpless_belief": [],
            "unlovable_belief": [],
            "worthless_belief": [],
            "probs": {"helpless": [], "unlovable": [], "worthless": []},
            "major_active": [],
        }

    # 1) Major
    major_tok, major_model = _load_major(use_fast=use_fast)
    major_probs, major_id2label = _predict_probs(major_tok, major_model, text, max_len=max_len)

    major_items = [(major_id2label[i], float(major_probs[i])) for i in range(len(major_probs))]
    major_items.sort(key=lambda x: x[1], reverse=True)

    # NEW: soglia separata per 'worthless' (come nel main)
    t_default = major_threshold
    t_w = major_threshold_worthless if major_threshold_worthless is not None else t_default

    major_active = set()
    for lbl, p in major_items:
        thr = t_w if lbl == "worthless" else t_default
        if p >= thr:
            major_active.add(lbl)

    # 2) Fine
    fine_tok, fine_model = _load_fine(use_fast=use_fast)
    fine_probs, fine_id2label = _predict_probs(fine_tok, fine_model, text, max_len=max_len)

    fine_to_major = build_fine_to_major_map(fine_id2label)

    # 3) HARD GATE
    hard_probs = fine_probs.copy()
    for i in range(len(hard_probs)):
        lbl = fine_id2label[i]
        maj = fine_to_major.get(lbl, None)
        if maj is None:
            continue  # non mappata: lasciata intatta
        if maj not in major_active:
            hard_probs[i] = 0.0

    # 4) Filtra sopra soglia fine_threshold e raggruppa per major (SENZA top-k)
    out = {
        "helpless_belief": [],
        "unlovable_belief": [],
        "worthless_belief": [],
        "probs": {"helpless": [], "unlovable": [], "worthless": []},
        "major_active": sorted(list(major_active)),
    }

    buckets = {"helpless": [], "unlovable": [], "worthless": []}

    hard_items = [(fine_id2label[i], float(hard_probs[i])) for i in range(len(hard_probs))]
    hard_items.sort(key=lambda x: x[1], reverse=True)

    for lbl, p in hard_items:
        if p < fine_threshold:
            continue
        maj = fine_to_major.get(lbl, None)
        if maj in buckets:
            buckets[maj].append((lbl, p))

    for maj in ["helpless", "unlovable", "worthless"]:
        out[f"{maj}_belief"] = [lbl for lbl, _ in buckets[maj]]
        out["probs"][maj] = [{"label": lbl, "prob": float(p)} for lbl, p in buckets[maj]]

    return out
