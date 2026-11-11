import argparse, json
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", default="evaluation_external.json")
    a = p.parse_args()
    df = pd.read_csv(a.csv).dropna(subset=["text", "label"])
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval(); guesses = []
    with torch.inference_mode():
        for i in range(0, len(df), 16):
            texts = df.text.astype(str).iloc[i:i+16].tolist()
            batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            batch = {k:v.to(device) for k,v in batch.items()}
            guesses.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    y = df.label.astype(int).tolist()
    result = {"sample_count": len(df), "accuracy": accuracy_score(y, guesses),
              "macro_f1": f1_score(y, guesses, average="macro"),
              "classification_report": classification_report(y, guesses, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(y, guesses).tolist()}
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["sample_count", "accuracy", "macro_f1"]}, indent=2))
if __name__ == "__main__": main()
