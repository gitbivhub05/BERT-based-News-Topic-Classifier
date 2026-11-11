import json, time
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self.model = None
        self.id_to_label = {}
        self.model_name = None
        self._load()

    def _load(self):
        if not self.model_dir.exists() or not (self.model_dir / "config.json").exists():
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self.model.to(self.device).eval()
            mapping = self.model_dir / "id_to_label.json"
            if mapping.exists():
                self.id_to_label = json.loads(mapping.read_text(encoding="utf-8"))
            self.model_name = self.model.config.name_or_path
        except Exception:
            self.tokenizer = self.model = None

    @property
    def loaded(self):
        return self.model is not None and self.tokenizer is not None

    def info(self):
        return {"loaded": self.loaded, "model_name": self.model_name,
                "device": str(self.device),
                "num_labels": self.model.config.num_labels if self.loaded else None,
                "id_to_label": self.id_to_label}

    def predict(self, texts: List[str]):
        if not self.loaded:
            raise RuntimeError("Model artifacts are unavailable. Train BERT first.")
        started = time.perf_counter()
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**batch).logits, dim=-1).cpu()
        elapsed = (time.perf_counter() - started) * 1000 / len(texts)
        output = []
        for row in probabilities:
            output.append({
                "predicted_label": int(row.argmax().item()),
                "probabilities": {str(i): round(float(v), 6) for i, v in enumerate(row)},
                "inference_ms": round(elapsed, 2)
            })
        return output
