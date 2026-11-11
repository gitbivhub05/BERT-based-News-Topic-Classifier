import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

class ArticleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = [int(x) for x in labels]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", default="backend/artifacts/bert-news-classifier")
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV requires text and label columns.")
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df.text.astype(str).str.strip()
    df = df[df.text.str.len() > 0]
    df["label"] = pd.to_numeric(df.label, errors="raise").astype(int)
    classes = sorted(df.label.unique().tolist())
    if classes != list(range(len(classes))) or len(classes) < 2:
        raise ValueError(f"Labels must be contiguous IDs starting at 0; found {classes}")
    if df.label.value_counts().min() < 10:
        raise ValueError("Need at least 10 rows per class for this split.")
    duplicate_count = int(df.text.duplicated().sum())
    print(f"Exact duplicate texts after first occurrence: {duplicate_count}")
    if duplicate_count:
        print("WARNING: inspect duplicates/leakage. They are not automatically removed.")

    train, temp = train_test_split(df, test_size=.20, random_state=a.seed, stratify=df.label)
    val, test = train_test_split(temp, test_size=.50, random_state=a.seed, stratify=temp.label)
    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_name, num_labels=len(classes))
    train_ds = ArticleDataset(train.text.tolist(), train.label.tolist(), tokenizer, a.max_length)
    val_ds = ArticleDataset(val.text.tolist(), val.label.tolist(), tokenizer, a.max_length)
    test_ds = ArticleDataset(test.text.tolist(), test.label.tolist(), tokenizer, a.max_length)

    def compute_metrics(pred):
        logits, y = pred
        guess = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(y, guess),
                "macro_f1": f1_score(y, guess, average="macro")}

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        learning_rate=a.learning_rate, weight_decay=.01,
        evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=2, logging_steps=100,
        report_to="none", seed=a.seed, fp16=torch.cuda.is_available()
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    pred = trainer.predict(test_ds)
    y_true = pred.label_ids; y_hat = np.argmax(pred.predictions, axis=1)
    report = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro")),
        "classification_report": classification_report(y_true, y_hat, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_hat).tolist(),
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "settings": vars(a), "exact_duplicate_count": duplicate_count
    }
    trainer.save_model(str(out)); tokenizer.save_pretrained(str(out))
    (out / "id_to_label.json").write_text(json.dumps({str(i): i for i in classes}, indent=2))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({"text": test.text.tolist(), "true_label": y_true,
                  "predicted_label": y_hat}).to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps({k: report[k] for k in ["accuracy", "macro_f1", "split_counts"]}, indent=2))
    print(f"Artifacts saved to {out.resolve()}")

if __name__ == "__main__": main()
