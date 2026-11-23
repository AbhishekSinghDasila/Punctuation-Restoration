import argparse
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from punct.data import load_jsonl
from punct.labels import LABELS, end_of_input
from punct.lm_rescore import ExternalLM, build_nbest_cache, decode_from_cache
from punct.metrics import evaluate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = lambda *a: os.path.join(ROOT, *a)

ap = argparse.ArgumentParser()
ap.add_argument("--tagger", default="distilbert_finetuned")
ap.add_argument("--lm_epochs", type=int, default=2)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--seg_len", type=int, default=32)
args = ap.parse_args()

log_path = P("results", "logs", f"external_lm_{args.tagger}.log")


def log(m):
    print(m, flush=True)
    open(log_path, "a", encoding="utf-8").write(m + "\n")


train = load_jsonl(P("data", "processed", "train.jsonl"))
val = load_jsonl(P("data", "processed", "val.jsonl"))
test = load_jsonl(P("data", "processed", "test.jsonl"))

dom_dir = P("models", "distilgpt2_domain")
lm_info = {}
if not os.path.exists(os.path.join(dom_dir, "config.json")):
    lm = ExternalLM("distilgpt2")
    t0 = time.time()
    hist = lm.finetune([d["target_text"] for d in train], dom_dir, epochs=args.lm_epochs, log_fn=log)
    lm_info["finetune_seconds"] = time.time() - t0
    json.dump(hist, open(P("results", "logs", "distilgpt2_domain_loss.json"), "w"))
lms = {"generic": ExternalLM("distilgpt2"), "domain": ExternalLM(dom_dir)}
val_text = [d["target_text"] for d in val]
for name, lm in lms.items():
    lm_info[f"ppl_val_{name}"] = lm.perplexity(val_text)
    log(f"perplexity on val ({name} LM): {lm_info[f'ppl_val_{name}']:.2f}")

preds = pickle.load(open(P("results", "preds", f"{args.tagger}.pkl"), "rb"))
refs = {"val": [d["labels"] for d in val], "test": [d["labels"] for d in test]}
words = {"val": [d["words"] for d in val], "test": [d["words"] for d in test]}
base = {s: evaluate(refs[s], [end_of_input([LABELS[i] for i in p.argmax(1)], p) for p in preds[s]])[0]["micro"]["f1"]
        for s in ("val", "test")}
log(f"tagger alone (with end-of-input constraint): val micro-F1 {base['val']:.4f}  test {base['test']:.4f}")

LAMS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0]
summary = {"tagger": args.tagger, "k": args.k, "seg_len": args.seg_len, "lm": lm_info,
           "tagger_val_f1": base["val"], "tagger_test_f1": base["test"], "grid": {}}
for name, lm in lms.items():
    t0 = time.time()
    cache_file = P("results", "preds", f"nbest_cache_{args.tagger}_{name}.pkl")
    if os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, "rb"))
    else:
        cache = {s: build_nbest_cache(words[s], preds[s], lm, seg_len=args.seg_len, k=args.k) for s in ("val", "test")}
        pickle.dump(cache, open(cache_file, "wb"))
    log(f"[{name}] n-best + LM scoring: {time.time() - t0:.0f}s")
    grid = []
    for lam, beta in itertools.product(LAMS, BETAS):
        f = evaluate(refs["val"], decode_from_cache(cache["val"], lam, beta))[0]["micro"]["f1"]
        grid.append({"lambda": lam, "beta": beta, "val_f1": f})
    best = max(grid, key=lambda g: g["val_f1"])
    hyp_test = decode_from_cache(cache["test"], best["lambda"], best["beta"])
    s_test = evaluate(refs["test"], hyp_test)[0]
    log(f"[{name}] best on val: lambda={best['lambda']} beta={best['beta']} val F1 {best['val_f1']:.4f}"
        f" -> test micro-F1 {s_test['micro']['f1']:.4f}  QUESTION {s_test['QUESTION']['f1']:.3f}")
    summary["grid"][name] = {"best": best, "test_micro_f1": s_test["micro"]["f1"], "all": grid}

    onehot = {s: [np.eye(len(LABELS))[[LABELS.index(l) for l in h]] for h in hyps]
              for s, hyps in (("val", decode_from_cache(cache["val"], best["lambda"], best["beta"])), ("test", hyp_test))}
    pickle.dump({"meta": {"tag": f"{args.tagger}+lm_{name}", "lambda": best["lambda"], "beta": best["beta"]},
                 **onehot}, open(P("results", "preds", f"{args.tagger}+lm_{name}.pkl"), "wb"))
json.dump(summary, open(P("results", "tables", f"external_lm_{args.tagger}.json"), "w"), indent=2)
log("done")
