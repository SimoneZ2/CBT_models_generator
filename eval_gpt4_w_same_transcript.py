# run_vignette_eval.py
import os
import json
import hashlib
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from extract import gpt4_cbt_w_same_transcription

from compare_models import (
    compare_cbt_models_embeddings_only,
    compare_cbt_models_crossencoder_sts,
    compare_cbt_models_crossencoder_nli,
    compare_core_beliefs_fine,
    aggregate_core_beliefs_fine_metrics,
    compare_core_beliefs_major_and_fine
)

BASE_DIR = "./Psychotherapy_Transcripts"
#TRANSCRIPT_FILES = ['1001084005.txt', '1000370274.txt', '1000134003.txt', '1002565897.txt', '1004538622.txt', '1004687820.txt', '1000013338.txt', '1002565945.txt', '1004824673.txt', '1004538612.txt', '1001084034.txt', '1001084028.txt', '1001083958.txt', '1004538610.txt', '1004522810.txt', '1003009271.txt', '1000126013.txt', '1003269462.txt', '1000094811.txt', '1004522800.txt', '1004472416.txt', '1000016540.txt', '1002565853.txt', '1004687594.txt', '1003009240.txt', '1000056536.txt', '1004687750.txt', '1004532444.txt', '1004824713.txt', '1004600954.txt', '1001083994.txt', '1004565013.txt', '1002565915.txt', '1002565888.txt', '1000013261.txt', '1004532428.txt', '1004798935.txt', '1003009283.txt', '1000016558.txt', '1000122009.txt', '1000329602.txt', '1000393712.txt', '1004824757.txt', '1004478010.txt', '1004824799.txt', '1000122292.txt', '1004687590.txt', '1003009239.txt', '1000325703.txt', '1000122292.txt'] # situation 2
TRANSCRIPT_FILES = ['1001084005.txt', '1000370274.txt', '1000134003.txt', '1002565897.txt', '1004538622.txt', '1004687820.txt', '1000013338.txt', '1002565945.txt', '1004824673.txt', '1004538612.txt', '1001084034.txt', '1001084028.txt', '1001083958.txt', '1004538610.txt', '1004522810.txt', '1003009271.txt', '1000126013.txt', '1003269462.txt', '1000094811.txt', '1004522800.txt', '1004472416.txt', '1000016540.txt', '1002565853.txt', '1004687594.txt', '1003009240.txt', '1000056536.txt', '1004687750.txt', '1004532444.txt', '1004824713.txt', '1004600954.txt', '1001083994.txt', '1004565013.txt', '1002565915.txt', '1002565888.txt', '1000013261.txt', '1004532428.txt', '1004798935.txt', '1003009283.txt', '1000016558.txt', '1000122009.txt', '1000329602.txt', '1000393712.txt', '1004824757.txt', '1004478010.txt', '1004824799.txt', '1000122292.txt', '1004687590.txt', '1003009239.txt', '1000325703.txt', '1000122292.txt'] # situation 1
def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default
    

def _extract_nli_contradiction_overall(nli_out: Dict[str, Any]) -> Optional[float]:
    """
    Calcola un "overall contradiction" dal risultato NLI.
    Usa la media, per ciascun field, di:
        (contr_g2p + contr_p2g) / 2
    poi fa la media sui field validi.
    """
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
    """
    Dati i record (uno per confronto), calcola le medie finali
    di embeddings cosine, sts, entailment overall, contradiction overall.
    """
    emb = []
    sts = []
    ent = []
    contr = []

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


def _index_gold_by_id(gold_models: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for m in gold_models:
        mid = str(m.get("id", "") or "").strip()
        if mid:
            idx[mid] = m
    return idx


def load_transcription(file_name: str) -> str:
    fname = (file_name or "").strip()
    if not fname:
        return ""
    path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run(
    *,
    vignette_path: str,
    gold_path: str,
    out_dir: str,
    max_n: Optional[int],
    model: str,
    verbose: bool = True,
) -> None:
    vignettes = _load_json(vignette_path)
    gold_models = _load_json(gold_path)

    gold_by_id = _index_gold_by_id(gold_models)

    generated_dir = os.path.join(out_dir, "generated_cbts")
    results_jsonl = os.path.join(out_dir, "compare_results.jsonl")  # <- jsonl

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    if os.path.exists(results_jsonl):
        os.remove(results_jsonl)

    all_records: List[Dict[str, Any]] = []
    beliefs_pairs: List[Dict[str, Any]] = []
    beliefs_pairs_major: List[Dict[str, Any]] = []

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()  # opzionale: anche console
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)


    count = 0
    for v in vignettes:
        if max_n is not None and count >= max_n:
            break

        source_id = str(v.get("source_id", "") or "").strip()
        vignette_en = str(v.get("vignette_en", "") or "").strip()

        if not source_id or not vignette_en:
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

        if count >= len(TRANSCRIPT_FILES):
            logger.info(f"[STOP] finiti transcript hard-coded (count={count})")
            break

        transcript_file = TRANSCRIPT_FILES[count]
        transcript = load_transcription(transcript_file)

        # 1) genera CBT
        predicted = gpt4_cbt_w_same_transcription(
            vignette_en,
            transcript,
            model=model,
        )

        logger.info("CBT generated")
        def _safe_list(x):
            return x if isinstance(x, list) else ([] if x is None else [x])
        output = {
            "history": (predicted.get("history") or "").strip() if isinstance(predicted.get("history"), str) else (predicted.get("history") or ""),
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

        # 4) compare (testo)
        out_emb = compare_cbt_models_embeddings_only(gold_norm, predicted_loaded, verbose_print=verbose)
        out_sts = compare_cbt_models_crossencoder_sts(gold_norm, predicted_loaded, verbose_print=verbose)
        out_nli = compare_cbt_models_crossencoder_nli(gold_norm, predicted_loaded)

        # 5) compare core beliefs (Case A)
        beliefs_cmp = compare_core_beliefs_major_and_fine(gold_norm, predicted_loaded)

        beliefs_pairs.append({
            "gold_set": beliefs_cmp["fine"]["gold_set"],
            "pred_set": beliefs_cmp["fine"]["pred_set"],
            "unknown_pred": beliefs_cmp["fine"]["unknown_pred"],
        })

        beliefs_pairs_major.append({
            "gold_set": beliefs_cmp["major"]["gold_set"],
            "pred_set": beliefs_cmp["major"]["pred_set"],
        })


        record = {
            "source_id": source_id,
            "patient_id": str(v.get("patient_id", "") or "").strip(),
            "generated_cbt_path": pred_path,
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
        all_records.append(record)

    
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

        count += 1

    logger.info(f"\nCompleted. Processed {count} vignette(s).")
    logger.info(f"Generated CBTs in: {generated_dir}")
    logger.info(f"Compare results JSONL: {results_jsonl}")


    # dataset-level beliefs metrics
    pairs_fine_total = _pairs_from_records_fine(all_records)
    pairs_major_total = _pairs_from_records_major(all_records)

    beliefs_report = aggregate_core_beliefs_fine_metrics(pairs_fine_total)
    major_report = aggregate_major_metrics(pairs_major_total)

    final_report = compute_final_means(all_records)
    final_report["core_beliefs_fine"] = beliefs_report
    final_report["core_beliefs_major"] = major_report
    final_report["resume"] = {
    "total": len(all_records),
    }

    final_path = os.path.join(out_dir, "final_means.json")
    _save_json(final_path, final_report)

    logger.info("\n=== FINAL MEANS ===")
    logger.info(json.dumps(final_report, ensure_ascii=False, indent=2))
    logger.info(f"\nSaved final means to: {final_path}")

    logger.info("\n=== CORE BELIEFS (fine-grained) METRICS ===")
    logger.info(json.dumps(beliefs_report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", type=str, default="vignette_mistral_current_sit1.json", help="Path a vignette.json")
    ap.add_argument("--gold", type=str, default="Patient_Psi_CM_Dataset.json", help="Path al file CBT gold")
    ap.add_argument("--out_dir", type=str, default="output_gpt4_w_same_transcription_sit1_2", help="Cartella output")
    ap.add_argument("--max_n", type=int, default=3, help="Massimo numero di vignette da processare (default: 3)")
    ap.add_argument("--model", type=str, default="gpt-4o", help="Modello da usare")
    ap.add_argument("--verbose", default=True, action="store_true", help="Stampa output dettagliato dei compare")
    args = ap.parse_args()

    run(
        vignette_path=args.vignettes,
        gold_path=args.gold,
        out_dir=args.out_dir,
        max_n=args.max_n,
        model=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
