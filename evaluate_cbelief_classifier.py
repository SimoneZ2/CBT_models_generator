import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MAJOR_DIR = "./cbt_pc_major_deberta_final"
#MAJOR_DIR = "./cbt_pc_major_roberta_final_oversampled"
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


def load_dataset(path: str, fmt: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset non trovato: {path}")

    if fmt == "jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if fmt == "json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError("Formato json: atteso una LISTA di elementi.")
        return obj

    raise ValueError("fmt deve essere 'jsonl' oppure 'json'")


def build_input_text(ex: Dict[str, Any]) -> str:
    situation = (ex.get("situation") or "").strip()
    inter = (ex.get("intermediate_belief") or "").strip()
    auto = (ex.get("auto_thought") or "").strip()

    #thoughts = " ".join([t for t in [inter, auto] if t])
    thoughts = " ".join([t for t in [auto] if t])
    return f"[SITUATION] {situation}\n[THOUGHTS] {thoughts}".strip()


def get_gold_major(ex: Dict[str, Any]) -> Set[str]:
    gold = set()
    if ex.get("helpless_belief"):
        gold.add("helpless")
    if ex.get("unlovable_belief"):
        gold.add("unlovable")
    if ex.get("worthless_belief"):
        gold.add("worthless")
    return gold


def get_gold_fine(ex: Dict[str, Any]) -> List[str]:
    fine = []
    for k in ["helpless_belief", "unlovable_belief", "worthless_belief"]:
        vals = ex.get(k) or []
        for v in vals:
            if isinstance(v, str) and v.strip():
                fine.append(v.strip())
    return fine


def load_model(model_dir: str, use_fast: bool, device: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: i for i, v in id2label.items()}
    return tok, model, id2label, label2id


def _expected_f1_random(N: int, s: int, q: float) -> float:
    """
    Expected F1 for a random classifier for ONE label, where:
      - N = total examples
      - s = support (positives in gold)
      - q = probability of predicting positive (Bernoulli)
    Uses expectation of TP/FP/FN:
      TP = s*q
      FP = (N-s)*q
      FN = s*(1-q)
    and plugs into F1 formula.
    """
    if N <= 0:
        return 0.0
    s = max(0, min(int(s), int(N)))
    q = float(q)
    q = max(0.0, min(1.0, q))

    tp = s * q
    fp = (N - s) * q
    fn = s * (1.0 - q)

    denom = (2.0 * tp + fp + fn)
    return float((2.0 * tp / denom) if denom > 0 else 0.0)


def estimate_random_baseline_f1(
    y_true: List[Set[str]],
    label_space: List[str],
    q_mode: str = "coinflip",
) -> Dict[str, Any]:
    """
    Computes expected per-label F1 for a random classifier.

    q_mode:
      - "coinflip": q=0.5 for every label (random yes/no)
      - "prevalence": q = prevalence = support/N (best naive random for F1-ish)
      - "always_positive": q=1.0
      - "always_negative": q=0.0

    Returns:
      {
        "N": int,
        "per_label": {lbl: {"support": s, "prevalence": pi, "q": q, "f1_rand": f1}},
        "macro_f1_rand": float
      }
    """
    N = len(y_true)
    if N == 0:
        return {"N": 0, "per_label": {}, "macro_f1_rand": 0.0}

    # supports
    support = {lbl: 0 for lbl in label_space}
    for t in y_true:
        for lbl in t:
            if lbl in support:
                support[lbl] += 1

    per_label = {}
    f1s = []
    for lbl in label_space:
        s = support[lbl]
        pi = s / N

        if q_mode == "coinflip":
            q = 0.5
        elif q_mode == "prevalence":
            q = pi
        elif q_mode == "always_positive":
            q = 1.0
        elif q_mode == "always_negative":
            q = 0.0
        else:
            raise ValueError("q_mode must be one of: coinflip, prevalence, always_positive, always_negative")

        f1 = _expected_f1_random(N, s, q)
        per_label[lbl] = {"support": s, "prevalence": pi, "q": q, "f1_rand": f1}
        f1s.append(f1)

    macro = float(np.mean(f1s)) if f1s else 0.0
    return {"N": N, "per_label": per_label, "macro_f1_rand": macro}


def compare_model_vs_random(
    model_metrics: Dict[str, Any],
    random_baseline: Dict[str, Any],
    label_space: List[str],
    eps: float = 1e-12,
    topk: int = 15,
    title: str = "MODEL vs RANDOM (per-label F1)"
) -> Dict[str, Any]:
    """
    Compares model per-label F1 to random baseline per-label F1.

    Expects:
      model_metrics["per_label"][lbl]["f1"]
      random_baseline["per_label"][lbl]["f1_rand"]

    Prints summary + top improvements/regressions.
    Returns a dict with counts and deltas.
    """
    better, worse, equal = 0, 0, 0
    deltas = []

    for lbl in label_space:
        f1_m = float(model_metrics["per_label"].get(lbl, {}).get("f1", 0.0))
        f1_r = float(random_baseline["per_label"].get(lbl, {}).get("f1_rand", 0.0))
        d = f1_m - f1_r
        deltas.append((lbl, f1_m, f1_r, d))

        if d > eps:
            better += 1
        elif d < -eps:
            worse += 1
        else:
            equal += 1

    deltas_sorted = sorted(deltas, key=lambda x: x[3], reverse=True)

    print(f"\n==== {title} ====")
    print(f"Random baseline macro-F1 (expected) = {random_baseline.get('macro_f1_rand', 0.0):.4f}")
    print(f"Labels better: {better} | worse: {worse} | equal: {equal} (out of {len(label_space)})")

    print(f"\nTop-{topk} improvements (label | model_f1 | rand_f1 | delta):")
    for lbl, f1_m, f1_r, d in deltas_sorted[:topk]:
        print(f" - {lbl:40s}  {f1_m:.4f}  {f1_r:.4f}  {d:+.4f}")

    print(f"\nTop-{topk} regressions (label | model_f1 | rand_f1 | delta):")
    for lbl, f1_m, f1_r, d in deltas_sorted[-topk:][::-1]:
        print(f" - {lbl:40s}  {f1_m:.4f}  {f1_r:.4f}  {d:+.4f}")

    return {
        "better": better,
        "worse": worse,
        "equal": equal,
        "deltas": deltas_sorted,
        "random_macro_f1": random_baseline.get("macro_f1_rand", 0.0),
    }


@torch.no_grad()
def batched_predict_probs(
    tok, model, texts: List[str], batch_size: int, max_len: int, device: str
) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(
            batch,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs = sigmoid(logits)
        all_probs.append(probs)
    return np.vstack(all_probs)


def build_fine_to_major_map(fine_id2label: Dict[int, str]) -> Dict[str, str]:
    canon = {}
    for major, fine_list in PAPER_MAPPING.items():
        for fl in fine_list:
            canon[norm_label(fl)] = major

    fine_to_major = {}
    missing = []
    for i, lbl in fine_id2label.items():
        key = norm_label(lbl)
        if key in canon:
            fine_to_major[lbl] = canon[key]
        else:
            missing.append(lbl)

    if missing:
        print("\n[WARN] Alcune fine-grained label del modello NON matchano PAPER_MAPPING.")
        print("       Nel gating verranno ignorate (rimangono valutabili se predette, ma non gated correttamente).")
        for m in missing[:50]:
            print("   -", m)
        if len(missing) > 50:
            print(f"   ... +{len(missing)-50} altre")

    return fine_to_major


def normalize_gold_fine_to_model_labels(gold_fine: List[str], model_labels: List[str]) -> Set[str]:
    # mappa norm -> label_model
    norm2model = {norm_label(l): l for l in model_labels}

    matched = set()
    unmatched = []
    for g in gold_fine:
        ng = norm_label(g)
        if ng in norm2model:
            matched.add(norm2model[ng])
        else:
            unmatched.append(g)

    return matched, unmatched


def multilabel_metrics(
    y_true: List[Set[str]],
    y_pred: List[Set[str]],
    label_space: List[str],
) -> Dict[str, Any]:
    # per-label TP/FP/FN
    tp = {l: 0 for l in label_space}
    fp = {l: 0 for l in label_space}
    fn = {l: 0 for l in label_space}

    subset_ok = 0
    jaccs = []

    for t, p in zip(y_true, y_pred):
        if t == p:
            subset_ok += 1

        inter = len(t & p)
        union = len(t | p)
        jaccs.append(inter / union if union else 1.0)

        for l in label_space:
            in_t = l in t
            in_p = l in p
            if in_t and in_p:
                tp[l] += 1
            elif (not in_t) and in_p:
                fp[l] += 1
            elif in_t and (not in_p):
                fn[l] += 1

    # micro
    TP = sum(tp.values())
    FP = sum(fp.values())
    FN = sum(fn.values())
    micro_p = TP / (TP + FP) if (TP + FP) else 0.0
    micro_r = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # macro
    per_label = {}
    ps, rs, f1s = [], [], []
    for l in label_space:
        P = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        R = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        F1 = (2 * P * R / (P + R)) if (P + R) else 0.0
        support = tp[l] + fn[l]
        per_label[l] = {"precision": P, "recall": R, "f1": F1, "support": support}
        ps.append(P); rs.append(R); f1s.append(F1)

    macro_p = float(np.mean(ps)) if ps else 0.0
    macro_r = float(np.mean(rs)) if rs else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "subset_accuracy": subset_ok / len(y_true) if y_true else 0.0,
        "jaccard_mean": float(np.mean(jaccs)) if jaccs else 0.0,
        "per_label": per_label
    }


def print_metrics(title: str, m: Dict[str, Any], topk_labels: int = 15):
    print(f"\n==== {title} ====")
    print(f"micro  P={m['micro']['precision']:.4f} R={m['micro']['recall']:.4f} F1={m['micro']['f1']:.4f}")
    print(f"macro  P={m['macro']['precision']:.4f} R={m['macro']['recall']:.4f} F1={m['macro']['f1']:.4f}")
    print(f"subset accuracy = {m['subset_accuracy']:.4f}")
    print(f"jaccard mean    = {m['jaccard_mean']:.4f}")

    # top label by support
    per_label = m["per_label"]
    items = sorted(per_label.items(), key=lambda x: x[1]["support"], reverse=True)
    print(f"\nTop-{topk_labels} label per support (support, F1):")
    for lbl, st in items[:topk_labels]:
        print(f" - {lbl:40s}  sup={st['support']:4d}  F1={st['f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Patient_Psi_CM_Dataset.json", help="Path dataset (.jsonl o .json)")
    #ap.add_argument("--data", default="outputs_pipeline_current/merged_cbts.json", help="Path dataset (.jsonl o .json)")
    ap.add_argument("--fmt", choices=["jsonl", "json"], default="json")
    ap.add_argument("--major_dir", default=MAJOR_DIR)
    ap.add_argument("--fine_dir", default=FINE_DIR)
    ap.add_argument("--major_threshold", type=float, default=0.5)
    ap.add_argument("--major_threshold_worthless", type=float, default=0.6, help="Override threshold only for 'worthless' (e.g. 0.35). If None, uses --major_threshold.")
    ap.add_argument("--fine_threshold", type=float, default=0.30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--use_fast", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--save_report", default=None, help="Salva report JSON qui (opzionale)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    data = load_dataset(args.data, args.fmt)
    print(f"[INFO] loaded {len(data)} examples")

    texts = [build_input_text(ex) for ex in data]

    # Load models once
    major_tok, major_model, major_id2label, major_label2id = load_model(args.major_dir, args.use_fast, device)
    fine_tok, fine_model, fine_id2label, fine_label2id = load_model(args.fine_dir, args.use_fast, device)

    major_labels = [major_id2label[i] for i in range(len(major_id2label))]
    fine_labels  = [fine_id2label[i] for i in range(len(fine_id2label))]

    # Predict probs (batched)
    major_probs = batched_predict_probs(major_tok, major_model, texts, args.batch_size, args.max_len, device)
    fine_probs  = batched_predict_probs(fine_tok, fine_model, texts, args.batch_size, args.max_len, device)

    # Mapping for gating
    fine_to_major = build_fine_to_major_map(fine_id2label)

    # Build gold sets
    y_true_major: List[Set[str]] = []
    y_pred_major: List[Set[str]] = []

    y_true_fine: List[Set[str]] = []
    y_pred_fine_gated: List[Set[str]] = []

    unmatched_gold_fine_total = 0

    for idx, ex in enumerate(data):
        # GOLD
        gold_major = get_gold_major(ex)
        gold_fine_raw = get_gold_fine(ex)
        gold_fine, unmatched = normalize_gold_fine_to_model_labels(gold_fine_raw, fine_labels)
        unmatched_gold_fine_total += len(unmatched)

        # PRED major
       # mp = major_probs[idx]
       # pred_major = {major_id2label[i] for i, p in enumerate(mp) if p >= args.major_threshold}
        mp = major_probs[idx]

        t_default = args.major_threshold
        t_w = args.major_threshold_worthless if args.major_threshold_worthless is not None else t_default

        pred_major = set()
        for i, p in enumerate(mp):
          lbl = major_id2label[i]
          thr = t_w if lbl == "worthless" else t_default
          if p >= thr:
            pred_major.add(lbl)
        # PRED fine + HARD GATE
        fp = fine_probs[idx].copy()
        # hard gate: se major del label non è attivo -> 0
        for i in range(len(fp)):
            lbl = fine_id2label[i]
            maj = fine_to_major.get(lbl, None)
            if maj is None:
                continue
            if maj not in pred_major:
                fp[i] = 0.0

        pred_fine = {fine_id2label[i] for i, p in enumerate(fp) if p >= args.fine_threshold}

        y_true_major.append(gold_major)
        y_pred_major.append(pred_major)

        y_true_fine.append(gold_fine)
        y_pred_fine_gated.append(pred_fine)

    # Metriche
    major_metrics = multilabel_metrics(y_true_major, y_pred_major, label_space=major_labels)
    fine_metrics  = multilabel_metrics(y_true_fine, y_pred_fine_gated, label_space=fine_labels)

    print_metrics("MAJOR (helpless/unlovable/worthless)", major_metrics, topk_labels=3)
    print_metrics("FINE (HARD GATE)", fine_metrics, topk_labels=20)


    rand_major = estimate_random_baseline_f1(y_true_major, major_labels, q_mode="coinflip")
    compare_model_vs_random(
        model_metrics=major_metrics,
        random_baseline=rand_major,
        label_space=major_labels,
        topk=3,
        title="MAJOR: model vs random (coinflip q=0.5)"
    )

    # Fine random baseline
    rand_fine = estimate_random_baseline_f1(y_true_fine, fine_labels, q_mode="coinflip")
    compare_model_vs_random(
        model_metrics=fine_metrics,
        random_baseline=rand_fine,
        label_space=fine_labels,
        topk=20,
        title="FINE: model vs random (coinflip q=0.5)"
    )

    if unmatched_gold_fine_total:
        print(f"\n[WARN] Gold fine-grained non matchate alle label del modello (dopo normalizzazione): {unmatched_gold_fine_total}")
        print("       Queste vengono ignorate nella valutazione fine-grained (altrimenti falsano il recall).")

    report = {
        "config": {
            "data": args.data,
            "fmt": args.fmt,
            "major_dir": args.major_dir,
            "fine_dir": args.fine_dir,
            "major_threshold": args.major_threshold,
            "fine_threshold": args.fine_threshold,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "device": device,
        },
        "major": major_metrics,
        "fine_hard_gate": fine_metrics,
        "unmatched_gold_fine_total": unmatched_gold_fine_total,
        "n_examples": len(data),
    }

    if args.save_report:
        Path(args.save_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] Report salvato in: {args.save_report}")


if __name__ == "__main__":
    main()
