# run_vignette_eval_resume.py
import os
import json
import hashlib
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import random
from pathlib import Path

from extract import gpt4_cbt_w_random_transcription

from compare_models import (
    compare_cbt_models_embeddings_only,
    compare_cbt_models_crossencoder_sts,
    compare_cbt_models_crossencoder_nli,
    aggregate_core_beliefs_fine_metrics,
    compare_core_beliefs_major_and_fine,
)

BASE_DIR = "./Psychotherapy_Transcripts"

# ----------------------------
# Helpers JSON/JSONL + resume
# ----------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # linea corrotta: skip
                continue
    return rows

def _index_existing_by_source_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for r in records:
        sid = str(r.get("source_id", "") or "").strip()
        if sid:
            idx[sid] = r
    return idx

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ----------------------------
# Metrics helpers (uguali)
# ----------------------------
def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _extract_nli_contradiction_overall(nli_out: Dict[str, Any]) -> Optional[float]:
    by_field = nli_out.get("by_field", []) or []
    vals: List[float] = []
    for item in by_field:
        c1 = _safe_float(item.get("contr_g2p", None))
        c2 = _safe_float(item.get("contr_p2g", None))
        if c1 is None and c2 is None:
            continue
        if c1 is None:
            vals.append(c2)
        elif c2 is None:
            vals.append(c1)
        else:
            vals.append((c1 + c2) / 2.0)
    if not vals:
        return None
    return sum(vals) / len(vals)

def compute_final_means(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    emb, sts, ent, contr = [], [], [], []
    for r in records:
        s = r.get("scores", {}) or {}
        if s.get("embeddings_cosine") is not None:
            emb.append(float(s["embeddings_cosine"]))
        if s.get("crossencoder_sts") is not None:
            sts.append(float(s["crossencoder_sts"]))
        if s.get("nli_entailment_overall") is not None:
            ent.append(float(s["nli_entailment_overall"]))
        if s.get("nli_contradiction_overall") is not None:
            contr.append(float(s["nli_contradiction_overall"]))

    def mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "n": len(records),
        "means": {
            "embeddings_cosine": mean(emb),
            "crossencoder_sts": mean(sts),
            "nli_entailment_overall": mean(ent),
            "nli_contradiction_overall": mean(contr),
        },
        "counts": {
            "embeddings_cosine": len(emb),
            "crossencoder_sts": len(sts),
            "nli_entailment_overall": len(ent),
            "nli_contradiction_overall": len(contr),
        },
    }

def _normalize_gold_model(gold: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(gold)
    if "helpless_belief" not in g and "helpless_belief_current" in g:
        g["helpless_belief"] = g.get("helpless_belief_current", [])
    if "unlovable_belief" not in g and "unlovable_belief_current" in g:
        g["unlovable_belief"] = g.get("unlovable_belief_current", [])
    if "worthless_belief" not in g and "worthless_belief_current" in g:
        g["worthless_belief"] = g.get("worthless_belief_current", [])
    for k in ["helpless_belief", "unlovable_belief", "worthless_belief", "emotion", "type"]:
        if k in g and g[k] is None:
            g[k] = []
        if k in g and isinstance(g[k], str):
            g[k] = [g[k]]
    return g

def _index_gold_by_id(gold_models: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for m in gold_models:
        mid = str(m.get("id", "") or "").strip()
        if mid:
            idx[mid] = m
    return idx

def aggregate_major_metrics(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_space = ["helpless", "unlovable", "worthless"]
    tp = fp = fn = 0
    subset_ok = 0
    jaccs = []

    def jacc(a, b):
        u = a | b
        return 1.0 if not u else len(a & b) / len(u)

    for ex in pairs:
        g = set(ex.get("gold_set", []) or [])
        p = set(ex.get("pred_set", []) or [])

        if g == p:
            subset_ok += 1
        jaccs.append(jacc(g, p))

        for l in label_space:
            in_g = l in g
            in_p = l in p
            if in_g and in_p:
                tp += 1
            elif (not in_g) and in_p:
                fp += 1
            elif in_g and (not in_p):
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    n = len(pairs)
    return {
        "n": n,
        "micro": {"precision": precision, "recall": recall, "f1": f1},
        "subset_accuracy": (subset_ok / n) if n else 0.0,
        "jaccard_mean": (sum(jaccs) / n) if n else 0.0,
        "label_space_size": len(label_space),
    }

def _pairs_from_records_fine(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        det = r.get("details", {}) or {}
        out.append({
            "gold_set": det.get("fine_gold_set", []) or [],
            "pred_set": det.get("fine_pred_set", []) or [],
            "unknown_pred": det.get("fine_unknown_pred", []) or [],
        })
    return out

def _pairs_from_records_major(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        det = r.get("details", {}) or {}
        out.append({
            "gold_set": det.get("major_gold_set", []) or [],
            "pred_set": det.get("major_pred_set", []) or [],
        })
    return out

# ----------------------------
# Transcript helpers
# ----------------------------
def list_transcript_files(base_dir: str) -> List[str]:
    p = Path(base_dir)
    if not p.exists():
        return []
    return sorted([x.name for x in p.glob("*.txt") if x.is_file()])

def load_transcription(file_name: str) -> str:
    fname = (file_name or "").strip()
    if not fname:
        return ""
    path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def pick_random_transcript(*, all_files: List[str], used: List[str], rng: random.Random) -> Optional[str]:
    if not all_files:
        return None
    available = [f for f in all_files if f not in used]
    if not available:
        used.clear()
        available = list(all_files)
    return rng.choice(available)

# ----------------------------
# Resume state (opzionale)
# ----------------------------
def _load_resume_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}

def _save_resume_state(path: str, obj: Dict[str, Any]) -> None:
    _save_json(path, obj)

# ----------------------------
# Main run
# ----------------------------
def run(
    *,
    vignette_path: str,
    gold_path: str,
    out_dir: str,
    max_n: Optional[int],
    model: str,
    verbose: bool = True,
    resume: bool = True,
) -> None:
    vignettes = _load_json(vignette_path)
    gold_models = _load_json(gold_path)
    gold_by_id = _index_gold_by_id(gold_models)

    generated_dir = os.path.join(out_dir, "generated_cbts")
    results_jsonl = os.path.join(out_dir, "compare_results.jsonl")
    state_path = os.path.join(out_dir, "resume_state.json")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    # logging
    log_path = os.path.join(out_dir, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # RESUME: carico records già valutati
    existing_records: List[Dict[str, Any]] = []
    done_ids = set()
    if resume:
        existing_records = _read_jsonl(results_jsonl)
        done_ids = set(_index_existing_by_source_id(existing_records).keys())
        logger.info(f"[RESUME] Found {len(done_ids)} already-evaluated source_id in {results_jsonl}")
    else:
        # run pulita
        if os.path.exists(results_jsonl):
            os.remove(results_jsonl)
        logger.info("[RESUME] disabled -> compare_results.jsonl reset")

    # transcripts + stato RNG (opzionale)
    all_transcripts = list_transcript_files(BASE_DIR)
    if not all_transcripts:
        logger.warning(f"[WARN] Nessun file .txt trovato in BASE_DIR={BASE_DIR}")

    used_transcripts: List[str] = []
    rng_seed: int = int(datetime.now().timestamp())

    if resume:
        st = _load_resume_state(state_path)
        if isinstance(st.get("used_transcripts"), list):
            used_transcripts = [str(x) for x in st["used_transcripts"]]
        if isinstance(st.get("rng_seed"), int):
            rng_seed = st["rng_seed"]

    rng = random.Random(rng_seed)

    new_records: List[Dict[str, Any]] = []

    count_new = 0
    for v in vignettes:
        if max_n is not None and count_new >= max_n:
            break

        source_id = str(v.get("source_id", "") or "").strip()
        vignette_en = str(v.get("vignette_en", "") or "").strip()

        if not source_id or not vignette_en:
            continue

        if resume and source_id in done_ids:
            if verbose:
                logger.info(f"[SKIP] source_id={source_id}: already in compare_results.jsonl")
            continue

        gold = gold_by_id.get(source_id)
        if not gold:
            if verbose:
                logger.info(f"[SKIP] source_id={source_id}: gold non trovato in {gold_path}")
            continue

        gold_norm = _normalize_gold_model(gold)

        if verbose:
            logger.info(f"SOURCE_ID:{source_id}")
            logger.info(f"GOLD_ID:{gold_norm.get('id')}")
            for k in ["helpless_belief", "unlovable_belief", "worthless_belief"]:
                logger.info(f"GOLD {k} ={gold_norm.get(k)}")

        logger.info(f"Found {len(all_transcripts)} transcript file(s) in {BASE_DIR}")

        transcript_file = pick_random_transcript(
            all_files=all_transcripts,
            used=used_transcripts,
            rng=rng,
        )
        if not transcript_file:
            logger.info(f"[STOP] nessun transcript disponibile in {BASE_DIR}")
            break

        used_transcripts.append(transcript_file)
        transcript = load_transcription(transcript_file)

        # 1) genera CBT
        predicted = gpt4_cbt_w_random_transcription(
            vignette_en,
            transcript,
            model=model,
        )

        logger.info("CBT generated")

        def _safe_list(x):
            return x if isinstance(x, list) else ([] if x is None else [x])

        output = {
            "history": (predicted.get("history") or "").strip()
            if isinstance(predicted.get("history"), str) else (predicted.get("history") or ""),
            "helpless_belief": predicted.get("helpless_belief", ""),
            "unlovable_belief": predicted.get("unlovable_belief", ""),
            "worthless_belief": predicted.get("worthless_belief", ""),
            "intermediate_belief": predicted.get("intermediate_belief", ""),
            "intermediate_belief_depression": predicted.get("intermediate_belief_depression", ""),
            "coping_strategies": _safe_list(predicted.get("coping_strategies", [])),
            "situation": predicted.get("situation", ""),
            "auto_thought": predicted.get("auto_thought", ""),
            "emotion": predicted.get("emotion", ""),
            "behavior": predicted.get("behavior", ""),
        }

        logger.info(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info(f"TRANSCRIPT_FILE:{transcript_file}")
        logger.info(f"PRED_MD5:{hashlib.md5(json.dumps(predicted, sort_keys=True).encode('utf-8')).hexdigest()}")
        for k in ["helpless_belief", "unlovable_belief", "worthless_belief"]:
            logger.info(f"PRED {k} ={predicted.get(k)}")

        # 2) salva CBT generato
        pred_path = os.path.join(generated_dir, f"{source_id}.json")
        _save_json(pred_path, predicted)

        # 3) ricarica
        predicted_loaded = _load_json(pred_path)

        # 4) compare
        out_emb = compare_cbt_models_embeddings_only(gold_norm, predicted_loaded, verbose_print=verbose)
        out_sts = compare_cbt_models_crossencoder_sts(gold_norm, predicted_loaded, verbose_print=verbose)
        out_nli = compare_cbt_models_crossencoder_nli(gold_norm, predicted_loaded)

        beliefs_cmp = compare_core_beliefs_major_and_fine(gold_norm, predicted_loaded)

        record = {
            "source_id": source_id,
            "patient_id": str(v.get("patient_id", "") or "").strip(),
            "generated_cbt_path": pred_path,
            "meta": {
                "transcript_file": transcript_file,
                "rng_seed": rng_seed,
            },
            "scores": {
                "embeddings_cosine": out_emb.get("overall", None),
                "crossencoder_sts": out_sts.get("overall", None),
                "nli_entailment_overall": out_nli.get("overall", None),
                "nli_contradiction_overall": _extract_nli_contradiction_overall(out_nli),
                "major_exact": beliefs_cmp["major"]["exact_match"],
                "major_jaccard": beliefs_cmp["major"]["jaccard"],
                "major_micro_f1": beliefs_cmp["major"]["micro"]["f1"],
                "fine_exact": beliefs_cmp["fine"]["exact_match"],
                "fine_subset_gold_in_pred": beliefs_cmp["fine"]["subset_gold_in_pred"],
                "fine_jaccard": beliefs_cmp["fine"]["jaccard"],
                "fine_micro_f1": beliefs_cmp["fine"]["micro"]["f1"],
            },
            "details": {
                "major_gold_set": beliefs_cmp["major"]["gold_set"],
                "major_pred_set": beliefs_cmp["major"]["pred_set"],
                "fine_gold_set": beliefs_cmp["fine"]["gold_set"],
                "fine_pred_set": beliefs_cmp["fine"]["pred_set"],
                "fine_unknown_pred": beliefs_cmp["fine"]["unknown_pred"],
            },
            "methods": {
                "embeddings": out_emb.get("method", ""),
                "sts": out_sts.get("method", ""),
                "nli": out_nli.get("method", ""),
                "beliefs": beliefs_cmp.get("method", ""),
            },
        }

        _append_jsonl(results_jsonl, record)
        new_records.append(record)
        done_ids.add(source_id)
        count_new += 1

        # salva stato resume (così a crash riparti *anche* con used_transcripts)
        if resume:
            _save_resume_state(state_path, {
                "rng_seed": rng_seed,
                "used_transcripts": used_transcripts,
                "updated_at": datetime.now().isoformat(),
            })

        if verbose:
            logger.info("\n========================================")
            logger.info(f"[DONE] {source_id} | saved={pred_path}")
            logger.info("Scores:")
            logger.info(f"  embeddings_cosine: {record['scores']['embeddings_cosine']}")
            logger.info(f"  crossencoder_sts:  {record['scores']['crossencoder_sts']}")
            logger.info(f"  nli_entailment:    {record['scores']['nli_entailment_overall']}")
            logger.info(f"  nli_contradiction: {record['scores']['nli_contradiction_overall']}")
            logger.info(f"  fine_jaccard:      {record['scores']['fine_jaccard']}")
            logger.info(f"  major_jaccard:     {record['scores']['major_jaccard']}")
            unk = record.get("details", {}).get("fine_unknown_pred")
            if unk:
                logger.warning(f"unknown_pred beliefs: {unk}")

    logger.info(f"\nCompleted. NEW processed {count_new} vignette(s).")
    logger.info(f"Generated CBTs in: {generated_dir}")
    logger.info(f"Compare results JSONL: {results_jsonl}")

    # ----------------------------
    # Metriche cumulative (existing + new) come nel primo codice
    # ----------------------------
    all_records_total = existing_records + new_records
    logger.info(f"[TOTAL] records existing={len(existing_records)} new={len(new_records)} total={len(all_records_total)}")

    pairs_fine_total = _pairs_from_records_fine(all_records_total)
    pairs_major_total = _pairs_from_records_major(all_records_total)

    beliefs_report = aggregate_core_beliefs_fine_metrics(pairs_fine_total)
    major_report = aggregate_major_metrics(pairs_major_total)

    final_report = compute_final_means(all_records_total)
    final_report["core_beliefs_fine"] = beliefs_report
    final_report["core_beliefs_major"] = major_report
    final_report["resume"] = {
        "enabled": resume,
        "existing": len(existing_records),
        "new": len(new_records),
        "total": len(all_records_total),
    }

    final_path = os.path.join(out_dir, "final_means.json")
    _save_json(final_path, final_report)

    logger.info("\n=== FINAL MEANS (CUMULATIVE) ===")
    logger.info(json.dumps(final_report, ensure_ascii=False, indent=2))
    logger.info(f"\nSaved final means to: {final_path}")
    logger.info("\n=== CORE BELIEFS (fine-grained) METRICS (CUMULATIVE) ===")
    logger.info(json.dumps(beliefs_report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", type=str, default="vignette_mistral_current_sit1.json")
    ap.add_argument("--gold", type=str, default="Patient_Psi_CM_Dataset.json")
    ap.add_argument("--out_dir", type=str, default="outputs_gpt4_w_random_transcript_sit1")
    ap.add_argument("--max_n", type=int, default=3)
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--verbose", default=True, action="store_true")
    ap.add_argument("--resume", default=True, action="store_true", help="Se true, riprende da compare_results.jsonl")
    ap.add_argument("--no_resume", dest="resume", action="store_false", help="Disabilita resume e resetta jsonl")
    args = ap.parse_args()

    run(
        vignette_path=args.vignettes,
        gold_path=args.gold,
        out_dir=args.out_dir,
        max_n=args.max_n,
        model=args.model,
        verbose=args.verbose,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()
