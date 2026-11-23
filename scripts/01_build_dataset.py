import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

from punct.data import build_documents, make_windows, save_jsonl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw", "train.csv")
OUT = os.path.join(ROOT, "data", "processed")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True); os.makedirs(TAB, exist_ok=True)

splits, rep = build_documents(RAW, min_words=5, seed=42)
keys = ["id", "raw", "clean", "words", "labels", "input_text", "target_text"]
for name, docs in splits.items():
    save_jsonl(docs, os.path.join(OUT, f"{name}.jsonl"), keys)

windows = make_windows(splits["train"], min_len=24, max_len=96, passes=2, seed=0)
save_jsonl(windows, os.path.join(OUT, "train_windows.jsonl"))

rows = []
for name, docs in splits.items():
    for d in docs:
        rows.append({"split": name, "id": d["id"], "n_words": len(d["words"]),
                     "input_text": d["input_text"], "target_text": d["target_text"],
                     "labels": " ".join(d["labels"])})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "synthetic_dataset.csv"), index=False)

report = rep.__dict__ | {"train_windows": len(windows)}
with open(os.path.join(TAB, "dataset_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
