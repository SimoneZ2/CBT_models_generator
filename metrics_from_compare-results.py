import json
import argparse
from typing import Iterable, Set, Dict, Tuple, Any, List
import math


def safe_set(x) -> Set[str]:
    """Convert list/None to set of strings."""
    if x is None:
        return set()
    if isinstance(x, list):
        return set(str(v) for v in x)
    raise TypeError(f"Expected list or None, got {type(x)}")


def subset_accuracy(gold: Set[str], pred: Set[str]) -> float:
    return 1.0 if gold == pred else 0.0


def jaccard(gold: Set[str], pred: Set[str]) -> float:
    union = gold | pred
    if not union:
        # both empty -> define as 1.0 (perfect match)
        return 1.0
    return len(gold & pred) / len(union)


def micro_counts(all_gold: Iterable[Set[str]], all_pred: Iterable[Set[str]]) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for g, p in zip(all_gold, all_pred):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn


def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    # micro precision/recall
    p_den = tp + fp
    r_den = tp + fn

    precision = (tp / p_den) if p_den > 0 else 1.0  # if nothing predicted -> define as 1.0 (like "perfect abstention")
    recall = (tp / r_den) if r_den > 0 else 1.0      # if nothing to retrieve -> define as 1.0

    f1_den = 2 * tp + fp + fn
    f1 = (2 * tp / f1_den) if f1_den > 0 else 1.0

    return precision, recall, f1


def _per_label_stats(gold_list: List[Set[str]], pred_list: List[Set[str]]) -> Dict[str, Dict[str, float]]:
    """
    Per ogni label nel union(gold,pred):
      - support = #occ in gold (tp + fn)
      - tp/fp/fn per one-vs-rest
      - precision/recall/f1 per label
    """
    label_universe = set()
    for g, p in zip(gold_list, pred_list):
        label_universe |= g
        label_universe |= p

    stats: Dict[str, Dict[str, float]] = {}
    for label in sorted(label_universe):
        tp = fp = fn = 0
        for g, p in zip(gold_list, pred_list):
            g_has = label in g
            p_has = label in p
            if g_has and p_has:
                tp += 1
            elif (not g_has) and p_has:
                fp += 1
            elif g_has and (not p_has):
                fn += 1

        support = tp + fn  # gold positives

        # precision: if never predicted (tp+fp==0), set to 1 only if also never present in gold, else 0
        if (tp + fp) == 0:
            precision = 1.0 if support == 0 else 0.0
        else:
            precision = tp / (tp + fp)

        # recall: if never present in gold (tp+fn==0), define as 1, else normal
        if (tp + fn) == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)

        f1_den = 2 * tp + fp + fn
        f1 = (2 * tp / f1_den) if f1_den > 0 else 1.0

        stats[label] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "support": float(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return stats


def _macro_prf(label_stats: Dict[str, Dict[str, float]]) -> Tuple[float, float, float]:
    """
    Macro: media semplice su tutte le label nel universe (gold ∪ pred).
    """
    if not label_stats:
        return 1.0, 1.0, 1.0

    ps = [v["precision"] for v in label_stats.values()]
    rs = [v["recall"] for v in label_stats.values()]
    f1s = [v["f1"] for v in label_stats.values()]
    return sum(ps) / len(ps), sum(rs) / len(rs), sum(f1s) / len(f1s)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e


def compute_metrics_extended(entries, gold_key: str, pred_key: str) -> Dict[str, Any]:
    gold_sets: List[Set[str]] = []
    pred_sets: List[Set[str]] = []

    subset_sum = 0.0
    jacc_sum = 0.0
    n = 0

    for obj in entries:
        details = obj.get("details", {})
        gold = safe_set(details.get(gold_key))
        pred = safe_set(details.get(pred_key))

        gold_sets.append(gold)
        pred_sets.append(pred)

        subset_sum += subset_accuracy(gold, pred)
        jacc_sum += jaccard(gold, pred)
        n += 1

    if n == 0:
        return {
            "count": 0,
            "subset_accuracy": 0.0,
            "jaccard_mean": 0.0,
            "micro": {"p": 0.0, "r": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0},
            "macro": {"p": 0.0, "r": 0.0, "f1": 0.0},
            "label_stats": {},
            "label_universe": [],
        }

    tp, fp, fn = micro_counts(gold_sets, pred_sets)
    micro_p, micro_r, micro_f1 = precision_recall_f1_from_counts(tp, fp, fn)

    label_stats = _per_label_stats(gold_sets, pred_sets)
    macro_p, macro_r, macro_f1 = _macro_prf(label_stats)

    label_universe = list(label_stats.keys())

    return {
        "count": n,
        "subset_accuracy": subset_sum / n,
        "jaccard_mean": jacc_sum / n,
        "micro": {"p": micro_p, "r": micro_r, "f1": micro_f1, "tp": tp, "fp": fp, "fn": fn},
        "macro": {"p": macro_p, "r": macro_r, "f1": macro_f1},
        "label_stats": label_stats,
        "label_universe": label_universe,
    }


def _is_number(x: Any) -> bool:
    # bool è subclass di int -> lo escludiamo per non “contare” True/False
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def compute_scores_summary(entries) -> Dict[str, Dict[str, float]]:
    """
    Per ogni key numerica dentro obj["scores"], calcola mean/min/max e count_valid.
    Ignora None e non-numerici.
    """
    acc: Dict[str, Dict[str, float]] = {}  # key -> {"sum":..., "count":..., "min":..., "max":...}

    for obj in entries:
        scores = obj.get("scores", {}) or {}
        if not isinstance(scores, dict):
            continue

        for k, v in scores.items():
            if not _is_number(v):
                continue

            if k not in acc:
                acc[k] = {"sum": 0.0, "count": 0.0, "min": float(v), "max": float(v)}
            acc[k]["sum"] += float(v)
            acc[k]["count"] += 1.0
            if float(v) < acc[k]["min"]:
                acc[k]["min"] = float(v)
            if float(v) > acc[k]["max"]:
                acc[k]["max"] = float(v)

    out: Dict[str, Dict[str, float]] = {}
    for k, st in acc.items():
        cnt = int(st["count"])
        out[k] = {
            "count_valid": cnt,
            "mean": (st["sum"] / st["count"]) if st["count"] else 0.0,
            "min": st["min"],
            "max": st["max"],
        }

    return out


def _print_block(title: str, metrics: Dict[str, Any], top_k: int):
    labels = metrics.get("label_universe", [])
    label_stats = metrics.get("label_stats", {})

    # titolo con lista label (come nel tuo esempio)
    label_part = "/".join(labels) if labels else ""
    if label_part:
        print(f"==== {title} ({label_part}) ====")
    else:
        print(f"==== {title} ====")

    micro = metrics["micro"]
    macro = metrics["macro"]

    print(f"micro  P={micro['p']:.4f} R={micro['r']:.4f} F1={micro['f1']:.4f}")
    print(f"macro  P={macro['p']:.4f} R={macro['r']:.4f} F1={macro['f1']:.4f}")
    print(f"subset accuracy = {metrics['subset_accuracy']:.4f}")
    print(f"jaccard mean    = {metrics['jaccard_mean']:.4f}")

    # Top-K per support
    items = []
    for lab, st in label_stats.items():
        items.append((int(st["support"]), lab, float(st["f1"])))
    items.sort(key=lambda x: (-x[0], x[1]))

    print()
    print(f"Top-{top_k} label per support (support, F1):")
    if not items:
        print(" - (no labels)")
        return

    for sup, lab, f1 in items[:top_k]:
        print(f" - {lab:<40} sup={sup:4d}  F1={f1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute set metrics (major/fine) from details + PRF (micro/macro) + top-k labels + average of all numeric fields in scores from JSONL."
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default="outputs_pipeline_unralated_transcription_sit2/compare_results.jsonl",
        help="Path to input .jsonl file",
    )
    parser.add_argument("--topk_major", type=int, default=3, help="Top-K label per support per MAJOR")
    parser.add_argument("--topk_fine", type=int, default=20, help="Top-K label per support per FINE")
    parser.add_argument(
        "--fine_title",
        type=str,
        default="FINE",
        help='Titolo per FINE (es: "FINE (HARD GATE)" -> passa --fine_title "FINE (HARD GATE)")',
    )
    args = parser.parse_args()

    data = list(read_jsonl(args.jsonl_path))

    major = compute_metrics_extended(data, gold_key="major_gold_set", pred_key="major_pred_set")
    fine = compute_metrics_extended(data, gold_key="fine_gold_set", pred_key="fine_pred_set")

    _print_block("MAJOR", major, top_k=args.topk_major)
    print()
    _print_block(args.fine_title, fine, top_k=args.topk_fine)

    # Medie di cosine/sts/nli/... (tutti i campi numerici in "scores")
    scores_summary = compute_scores_summary(data)
    print("\n=== SCORES (mean/min/max over numeric fields in obj['scores']) ===")
    if not scores_summary:
        print("No numeric scores found.")
    else:
        for k in sorted(scores_summary.keys()):
            s = scores_summary[k]
            print(
                f"{k}: mean={s['mean']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}  count_valid={s['count_valid']}"
            )


if __name__ == "__main__":
    main()
