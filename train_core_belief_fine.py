import random
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, precision_score, recall_score

# -----------------------
# Config
# -----------------------
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 512

TRAIN_CONFIG = "core_fine_test"   # 112 esempi (nel paper è test/benchmark)
HOLDOUT_CONFIG = "core_fine_seed" # 20 esempi (seed few-shot nel paper)

TRESHOLD = 0.25  # per multilabel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_text(ex):
    # Input consigliato: situation + thoughts (come major)
    sit = ex.get("situation", "") or ""
    th = ex.get("thoughts", "") or ""
    return f"[SITUATION]\n{sit}\n\n[THOUGHTS]\n{th}"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= TRESHOLD).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


def main():
    set_seed(SEED)

    # 112 esempi -> useremo per train/val split
    ds_112 = load_dataset("Psychotherapy-LLM/CBT-Bench", TRAIN_CONFIG)["train"]

    # 20 esempi -> holdout esterno finale
    ds_20 = load_dataset("Psychotherapy-LLM/CBT-Bench", HOLDOUT_CONFIG)["train"]

    # Costruisci label set dinamicamente (robusto a nomi esatti nel dataset)
    all_labels = set()
    for ex in ds_112:
        for l in ex.get("core_belief_fine_grained", []):
            all_labels.add(l)
    labels = sorted(list(all_labels))
    print(f"Detected {len(labels)} fine-grained labels:\n{labels}\n")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    def encode_labels_fine(ex):
        y = np.zeros(len(labels), dtype=np.float32)
        for l in ex.get("core_belief_fine_grained", []):
            if l in label2id:
                y[label2id[l]] = 1.0
        ex["labels"] = y
        return ex

    # Split 80/20 sui 112
    split = ds_112.train_test_split(test_size=0.2, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]

    # Salva colonne originali PRIMA di aggiungere labels (fix fondamentale)
    train_orig_cols = train_ds.column_names
    val_orig_cols = val_ds.column_names
    test_orig_cols = ds_20.column_names

    train_ds = train_ds.map(encode_labels_fine)
    val_ds = val_ds.map(encode_labels_fine)
    test_ds = ds_20.map(encode_labels_fine)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize(ex):
        return tokenizer(build_text(ex), truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize, remove_columns=train_orig_cols)
    val_ds = val_ds.map(tokenize, remove_columns=val_orig_cols)
    test_ds = test_ds.map(tokenize, remove_columns=test_orig_cols)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="./cbt_fc_fine_deberta",
        seed=SEED,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=12,  # piccolo dataset -> poche epoche ma controlla overfit
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        save_only_model=True,
        save_total_limit=1,

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,  # warning deprecazione ok
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Eval on VAL split (from the 112 examples) ===")
    val_metrics = trainer.evaluate()
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    print("\n=== Final TEST on 20 seed examples (external holdout) ===")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    out_dir = "./cbt_fc_fine_deberta_final"
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
