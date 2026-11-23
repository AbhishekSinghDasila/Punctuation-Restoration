import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from punct.data import load_jsonl
from punct.labels import LABELS
from punct.metrics import evaluate
from punct.tagger import Tagger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = lambda *a: os.path.join(ROOT, *a)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--train", action="store_true")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--limit_train", type=int, default=0, help="debug: use only N windows")
ap.add_argument("--threads", type=int, default=0)
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)
os.makedirs(P("results", "preds"), exist_ok=True); os.makedirs(P("results", "logs"), exist_ok=True)

log_path = P("results", "logs", f"{args.tag}.log")


def log(msg):
    print(msg, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


log(f"== {args.tag}: {args.model}  train={args.train}  device={'cuda' if torch.cuda.is_available() else 'cpu'}")
tagger = Tagger(args.model)
log(f"native labels: {tagger.native_id2label}  ->  fold {tagger.fold.tolist()}")
n_params = sum(p.numel() for p in tagger.model.parameters())
log(f"parameters: {n_params / 1e6:.1f} M")

val = load_jsonl(P("data", "processed", "val.jsonl"))
test = load_jsonl(P("data", "processed", "test.jsonl"))

meta = {"tag": args.tag, "model": args.model, "params_M": n_params / 1e6, "trained": args.train}
if args.train:
    windows = load_jsonl(P("data", "processed", "train_windows.jsonl"))
    if args.limit_train:
        windows = windows[:args.limit_train]
    t0 = time.time()
    hist = tagger.train(windows, P("models", args.tag), epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, log_fn=log)
    meta.update(train_seconds=time.time() - t0, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, train_windows=len(windows))
    json.dump(hist, open(P("results", "logs", f"{args.tag}_loss.json"), "w"))

out = {"meta": meta}
for name, docs in (("val", val), ("test", test)):
    t0 = time.time()
    probs = tagger.predict_proba([d["words"] for d in docs])
    out[name] = probs
    s, _ = evaluate([d["labels"] for d in docs], [[LABELS[i] for i in p.argmax(1)] for p in probs])
    meta[f"{name}_infer_seconds"] = time.time() - t0
    log(f"{name}: micro-F1 {s['micro']['f1']:.4f}  COMMA {s['COMMA']['f1']:.3f}  PERIOD {s['PERIOD']['f1']:.3f}"
        f"  QUESTION {s['QUESTION']['f1']:.3f}  SER {s['SER']:.3f}  ({time.time() - t0:.0f}s)")
with open(P("results", "preds", f"{args.tag}.pkl"), "wb") as f:
    pickle.dump(out, f)
log("done")
