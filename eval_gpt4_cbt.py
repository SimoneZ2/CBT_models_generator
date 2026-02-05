# run_vignette_eval.py
from __future__ import annotations

import os
import json
import hashlib
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


from extract import gpt4_cbt  # 
# ==== IMPORTA I COMPARE + CORE BELIEFS (major+fine) ====
from compare_models import (
    compare_cbt_models_embeddings_only,
    compare_cbt_models_crossencoder_sts,
    compare_cbt_models_crossencoder_nli,
    aggregate_core_beliefs_fine_metrics,
    compare_core_beliefs_major_and_fine,
)


# ----------------------------
# Utils
# ----------------------------
def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _extract_nli_contradiction_overall(nli_out: Dict[str, Any]) -> Optional[float]:
    """
    Calcola un "overall contradiction" dal risultato NLI.
    Media, per ciascun field, di (contr_g2p + contr_p2g)/2, poi media sui field validi.
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
    Medie finali di:
      - embeddings_cosine
      - crossencoder_sts
      - nli_entailment_overall
      - nli_contradiction_overall
      + counts per campo
    """
    emb: List[float] = []
    sts: List[float] = []
    ent: List[float] = []
    contr: List[float] = []

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
    """
    Normalizza le chiavi del gold per compatibilità:
    - helpless_belief_current -> helpless_belief
    - unlovable_belief_current -> unlovable_belief
    - worthless_belief_current -> worthless_belief
    Inoltre garantisce list per belief/emotion/type se sono string/None.
    """
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
    idx: Dict[str, Dict[str, Any]] = {}
    for m in gold_models:
        mid = str(m.get("id", "") or "").strip()
        if mid:
            idx[mid] = m
    return idx


def aggregate_major_metrics(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dataset-level metriche MAJOR (helpless/unlovable/worthless):
      - micro precision/recall/f1
      - subset_accuracy (exact set match)
      - jaccard_mean
    """
    label_space = ["helpless", "unlovable", "worthless"]
    tp = fp = fn = 0
    subset_ok = 0
    jaccs: List[float] = []

    def jacc(a: set, b: set) -> float:
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
    out: List[Dict[str, Any]] = []
    for r in records:
        det = r.get("details", {}) or {}
        out.append(
            {
                "gold_set": det.get("fine_gold_set", []) or [],
                "pred_set": det.get("fine_pred_set", []) or [],
                "unknown_pred": det.get("fine_unknown_pred", []) or [],
            }
        )
    return out


def _pairs_from_records_major(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        det = r.get("details", {}) or {}
        out.append(
            {
                "gold_set": det.get("major_gold_set", []) or [],
                "pred_set": det.get("major_pred_set", []) or [],
            }
        )
    return out


def _setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"[LOG] Writing logs to: {log_path}")
    return logger


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
    verbose: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logger = _setup_logger(out_dir)

    vignettes = _load_json(vignette_path)
    gold_models = _load_json(gold_path)
    gold_by_id = _index_gold_by_id(gold_models)

    generated_dir = os.path.join(out_dir, "generated_cbts")
    os.makedirs(generated_dir, exist_ok=True)

    results_jsonl = os.path.join(out_dir, "compare_results.jsonl")

    # reset risultati per run pulita (voluto)
    if os.path.exists(results_jsonl):
        os.remove(results_jsonl)
        logger.info(f"[RESET] Removed existing results file: {results_jsonl}")

    all_records: List[Dict[str, Any]] = []

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
            logger.info(f"SOURCE_ID: {source_id}")
            logger.info(f"GOLD_ID: {gold_norm.get('id')}")
            for k in ["helpless_belief", "unlovable_belief", "worthless_belief"]:
                logger.info(f"GOLD {k}: {gold_norm.get(k)}")
            logger.info(f"vignette_en: {vignette_en}")

        # 1) genera CBT dal solo testo vignette_en
        predicted = gpt4_cbt(vignette_en, model=model)

        pred_md5 = hashlib.md5(json.dumps(predicted, sort_keys=True).encode("utf-8")).hexdigest()
        if verbose:
            logger.info(f"PRED_MD5: {pred_md5}")

        # 2) salva CBT generato
        pred_path = os.path.join(generated_dir, f"{source_id}.json")
        _save_json(pred_path, predicted)

        # 3) ricarica (come richiesto: “reperiamo il CBT generato”)
        predicted_loaded = _load_json(pred_path)

        # 4) compare testuale
        out_emb = compare_cbt_models_embeddings_only(gold_norm, predicted_loaded, verbose_print=verbose)
        out_sts = compare_cbt_models_crossencoder_sts(gold_norm, predicted_loaded, verbose_print=verbose)
        out_nli = compare_cbt_models_crossencoder_nli(gold_norm, predicted_loaded)

        # 5) compare core beliefs (MAJOR + FINE)
        beliefs_cmp = compare_core_beliefs_major_and_fine(gold_norm, predicted_loaded)

        record = {
            "source_id": source_id,
            "patient_id": str(v.get("patient_id", "") or "").strip(),
            "generated_cbt_path": pred_path,
            "pred_md5": pred_md5,
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
            logger.info("========================================")
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

    # ---- Aggregazioni dataset-level ----
    pairs_fine_total = _pairs_from_records_fine(all_records)
    pairs_major_total = _pairs_from_records_major(all_records)

    beliefs_report = aggregate_core_beliefs_fine_metrics(pairs_fine_total)
    major_report = aggregate_major_metrics(pairs_major_total)

    final_report = compute_final_means(all_records)
    final_report["core_beliefs_fine"] = beliefs_report
    final_report["core_beliefs_major"] = major_report
    final_report["resume"] = {"total": len(all_records)}

    final_path = os.path.join(out_dir, "final_means.json")
    _save_json(final_path, final_report)

    logger.info("\n=== FINAL MEANS ===")
    logger.info(json.dumps(final_report, ensure_ascii=False, indent=2))
    logger.info(f"\nSaved final means to: {final_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", type=str, default="vignette_mistral_current_sit1.json", help="Path a vignette.json")
    ap.add_argument("--gold", type=str, default="Patient_Psi_CM_Dataset.json", help="Path al file CBT gold")
    ap.add_argument("--out_dir", type=str, default="outputs_gpt4_basic_sit1", help="Cartella output")
    ap.add_argument("--max_n", type=int, default=5, help="Massimo numero di vignette da processare (default: 5)")
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
