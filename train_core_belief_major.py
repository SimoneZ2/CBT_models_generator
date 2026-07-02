import os
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
MODEL_NAME = "roberta-base"
MAX_LEN = 512

# Major core belief labels (CBT-PC)
LABELS = ["helpless", "unlovable", "worthless"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


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
    # Input consigliato: situation + thoughts (più pulito e mirato)
    sit = ex.get("situation", "") or ""
    th = ex.get("thoughts", "") or ""
    return f"[SITUATION]\n{sit}\n\n[THOUGHTS]\n{th}"


def encode_labels_major(ex):
    # core_belief_major è una lista di stringhe
    y = np.zeros(len(LABELS), dtype=np.float32)
    for l in ex.get("core_belief_major", []):
        if l in label2id:
            y[label2id[l]] = 1.0
    ex["labels"] = y
    return ex


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


def main():
    set_seed(SEED)

    # 184 esempi (benchmark/test nel paper) -> usati per train/val split
    ds_184 = load_dataset("Psychotherapy-LLM/CBT-Bench", "core_major_test")["train"]

    # 20 esempi (seed nel paper) -> usati come test esterno finale
    ds_20 = load_dataset("Psychotherapy-LLM/CBT-Bench", "core_major_seed")["train"]

    # Train/val split sugli 184
    split = ds_184.train_test_split(test_size=0.2, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]

    # --- Attenzione: salviamo le colonne originali PRIMA di aggiungere labels ---
    train_orig_cols = train_ds.column_names
    val_orig_cols = val_ds.column_names
    test_orig_cols = ds_20.column_names

    # Add labels
    train_ds = train_ds.map(encode_labels_major)
    val_ds = val_ds.map(encode_labels_major)
    test_ds = ds_20.map(encode_labels_major)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize(ex):
        return tokenizer(build_text(ex), truncation=True, max_length=MAX_LEN)

    # Tokenize e rimuovi SOLO colonne originali (labels resta!)
    train_ds = train_ds.map(tokenize, remove_columns=train_orig_cols)
    val_ds = val_ds.map(tokenize, remove_columns=val_orig_cols)
    test_ds = test_ds.map(tokenize, remove_columns=test_orig_cols)

    # Data collator (padding dinamico)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="./cbt_pc_major_deberta",
        seed=SEED,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,  # con pochi dati, spesso 6-12 è ok
        weight_decay=0.01,
        # nuova API:
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,  # warning deprecazione ok per ora
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Eval on VAL split (from the 184 examples) ===")
    val_metrics = trainer.evaluate()
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    print("\n=== Final TEST on 20 seed examples (external holdout) ===")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # Salva modello + tokenizer
    trainer.save_model("./cbt_pc_major_roberta_final")
    tokenizer.save_pretrained("./cbt_pc_major_roberta_final")
    print("\nSaved to: ./cbt_pc_major_roberta_final")


if __name__ == "__main__":
    main()
