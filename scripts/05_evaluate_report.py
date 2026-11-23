import collections
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from punct.data import load_jsonl
from punct.labels import LABELS, PUNCT_LABELS, end_of_input, render
from punct.metrics import confusion, evaluate, flat_row, scores_from_confusion

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = lambda *a: os.path.join(ROOT, *a)
FIG = P("results", "figures"); TAB = P("results", "tables")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

SYSTEMS = [
    ("rule_end_period", "Rule: period at end only", "baseline"),
    ("punctall_zeroshot", "XLM-R base punct. model (zero-shot)", "pre-trained"),
    ("fullstop_zeroshot", "XLM-R large FullStop (zero-shot)", "pre-trained"),
    ("punctall_finetuned", "XLM-R base punct. model + domain FT", "fine-tuned"),
    ("distilbert_finetuned", "DistilBERT + domain FT", "fine-tuned"),
    ("distilbert_finetuned+lm_generic", "DistilBERT FT + GPT-2 rescoring (generic)", "fine-tuned + ext. LM"),
    ("distilbert_finetuned+lm_domain", "DistilBERT FT + GPT-2 rescoring (domain)", "fine-tuned + ext. LM"),
    ("punctall_finetuned+lm_generic", "XLM-R base FT + GPT-2 rescoring (generic)", "fine-tuned + ext. LM"),
    ("punctall_finetuned+lm_domain", "XLM-R base FT + GPT-2 rescoring (domain)", "fine-tuned + ext. LM"),
]
GROUP_COL = {"baseline": "#9aa5b1", "pre-trained": "#e4572e", "fine-tuned": "#2a6fdb", "fine-tuned + ext. LM": "#1b998b"}

data = {s: load_jsonl(P("data", "processed", f"{s}.jsonl")) for s in ("val", "test")}
refs = {s: [d["labels"] for d in data[s]] for s in data}
hyps, raw_hyps = {}, {}
for tag, name, grp in SYSTEMS:
    if tag == "rule_end_period":
        hyps[tag] = {s: [["O"] * (len(r) - 1) + ["PERIOD"] for r in refs[s]] for s in refs}
        continue
    f = P("results", "preds", f"{tag}.pkl")
    if os.path.exists(f):
        d = pickle.load(open(f, "rb"))

        hyps[tag] = {s: [end_of_input([LABELS[i] for i in p.argmax(1)], p) for p in d[s]] for s in ("val", "test")}
        raw_hyps[tag] = {s: [[LABELS[i] for i in p.argmax(1)] for p in d[s]] for s in ("val", "test")}
systems = [s for s in SYSTEMS if s[0] in hyps]

tables = {}
for split in ("val", "test"):
    rows = []
    for tag, name, grp in systems:
        sc, _ = evaluate(refs[split], hyps[tag][split])
        rows.append(flat_row(name, sc) | {"tag": tag, "group": grp})
    tables[split] = pd.DataFrame(rows)
    tables[split].to_csv(P("results", "tables", f"{split}_results.csv"), index=False)

t = tables["test"]
cols = ["model", "group", "COMMA_F1", "PERIOD_F1", "QUESTION_F1", "micro_P", "micro_R", "micro_F1", "macro_F1", "SER"]
md = t[cols].copy()
for c in cols[2:]:
    md[c] = md[c].map(lambda x: f"{x * 100:.1f}")
with open(P("results", "tables", "test_results.md"), "w") as f:
    f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for _, r in md.iterrows():
        f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
print(t[cols].round(4).to_string())

rows = []
for tag, name, grp in systems:
    if tag in raw_hyps:
        rows.append({"model": name, **{f"{sp}_raw": evaluate(refs[sp], raw_hyps[tag][sp])[0]["micro"]["f1"] for sp in ("val", "test")},
                     **{f"{sp}_constrained": evaluate(refs[sp], hyps[tag][sp])[0]["micro"]["f1"] for sp in ("val", "test")},
                     "val_docs_missing_final_mark": sum(h[-1] in ("O", "COMMA") for h in raw_hyps[tag]["val"])})
pd.DataFrame(rows).to_csv(P("results", "tables", "end_constraint_effect.csv"), index=False)

detail = t[["model"] + [f"{l}_{m}" for l in PUNCT_LABELS for m in ("P", "R", "F1")]]
detail.to_csv(P("results", "tables", "test_per_class_PRF.csv"), index=False)

rng = np.random.default_rng(0)
B = 1000
n = len(refs["test"])
cms = {tag: [confusion([refs["test"][i]], [hyps[tag]["test"][i]]) for i in range(n)] for tag, _, _ in systems}
idx = rng.integers(0, n, size=(B, n))
boot = {}
for tag, _, _ in systems:
    arr = np.stack(cms[tag])
    boot[tag] = np.array([scores_from_confusion(arr[b].sum(0))["micro"]["f1"] for b in idx])
sig = {"ci95_micro_f1": {tag: [float(np.percentile(boot[tag], 2.5)), float(np.percentile(boot[tag], 97.5))]
                         for tag in boot}}
pairs = [("punctall_finetuned", "punctall_zeroshot"), ("distilbert_finetuned", "punctall_zeroshot"),
         ("distilbert_finetuned", "fullstop_zeroshot"), ("punctall_finetuned", "fullstop_zeroshot"),
         ("distilbert_finetuned+lm_domain", "distilbert_finetuned"),
         ("distilbert_finetuned+lm_generic", "distilbert_finetuned"),
         ("punctall_finetuned+lm_domain", "punctall_finetuned"),
         ("distilbert_finetuned+lm_domain", "distilbert_finetuned+lm_generic"),
         ("punctall_finetuned+lm_domain", "punctall_zeroshot"),
         ("punctall_finetuned+lm_domain", "fullstop_zeroshot"),
         ("punctall_finetuned+lm_domain", "punctall_finetuned+lm_generic"),
         ("fullstop_zeroshot", "punctall_zeroshot")]
sig["paired"] = []
for a, b in pairs:
    if a in boot and b in boot:
        diff = boot[a] - boot[b]
        sig["paired"].append({"A": a, "B": b, "mean_diff": float(diff.mean()),
                              "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                              "p_A_not_better": float((diff <= 0).mean())})
json.dump(sig, open(P("results", "tables", "significance.json"), "w"), indent=2)

fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.8 / len(systems)
for i, (tag, name, grp) in enumerate(systems):
    r = t[t.tag == tag].iloc[0]
    vals = [r["COMMA_F1"], r["PERIOD_F1"], r["QUESTION_F1"], r["micro_F1"]]
    ax.bar(np.arange(4) + i * w - 0.4 + w / 2, vals, w, label=name, color=plt.cm.tab10(i), edgecolor="white")
ax.set_xticks(range(4)); ax.set_xticklabels(["COMMA F1", "PERIOD F1", "QUESTION F1", "micro F1"])
ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.1))
ax.set_title("Test-set F1 per punctuation class")
plt.tight_layout(); plt.savefig(P("results", "figures", "res_f1_by_class.png")); plt.close()

fig, ax = plt.subplots(figsize=(8, 0.45 * len(systems) + 1))
names = [n for _, n, _ in systems][::-1]
vals = [t[t.tag == tg].micro_F1.iloc[0] for tg, _, _ in systems][::-1]
lo = [sig["ci95_micro_f1"][tg][0] for tg, _, _ in systems][::-1]
hi = [sig["ci95_micro_f1"][tg][1] for tg, _, _ in systems][::-1]
cols_ = [GROUP_COL[g] for _, _, g in systems][::-1]
ax.barh(names, vals, color=cols_, xerr=[np.array(vals) - lo, np.array(hi) - vals], capsize=3)
for i, v in enumerate(vals): ax.text(v + 0.01, i, f"{v * 100:.1f}", va="center", fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("micro-F1 (COMMA+PERIOD+QUESTION), 95% bootstrap CI")
ax.set_title("Pre-trained vs fine-tuned vs fine-tuned + external LM (test)")
plt.tight_layout(); plt.savefig(P("results", "figures", "res_micro_f1.png")); plt.close()


def best_of(group_prefix):
    cand = t[t.group.str.startswith(group_prefix)]
    return cand.sort_values("micro_F1").iloc[-1].tag if len(cand) else None


pick = [x for x in (best_of("pre-trained"), best_of("fine-tuned"), best_of("fine-tuned + ext")) if x]
pick = list(dict.fromkeys(pick))
fig, axes = plt.subplots(1, len(pick), figsize=(4.4 * len(pick), 3.9))
for ax, tag in zip(np.atleast_1d(axes), pick):
    cm = confusion(refs["test"], hyps[tag]["test"]).astype(float)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j] * 100:.0f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(LABELS, fontsize=8); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("reference")
    ax.set_title(dict((a, b) for a, b, _ in SYSTEMS)[tag], fontsize=8)
plt.tight_layout(); plt.savefig(P("results", "figures", "res_confusion.png")); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.2))
for tag, col in (("distilbert_finetuned", "#2a6fdb"), ("punctall_finetuned", "#e4572e")):
    f = P("results", "logs", f"{tag}_loss.json")
    if os.path.exists(f):
        h = json.load(open(f))
        y = pd.Series([x["loss"] for x in h]).rolling(25, min_periods=1).mean()
        ax.plot([x["step"] for x in h], y, color=col, label=tag)
ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss (smoothed)"); ax.set_yscale("log"); ax.legend()
ax.set_title("Fine-tuning loss")
plt.tight_layout(); plt.savefig(P("results", "figures", "res_training_loss.png")); plt.close()

fig, ax = plt.subplots(figsize=(9, 0.45 * len(systems) + 1))
errs = []
for tag, name, _ in systems:
    cm = confusion(refs["test"], hyps[tag]["test"])
    pi = [LABELS.index(l) for l in PUNCT_LABELS]
    errs.append((name, sum(cm[0, j] for j in pi), sum(cm[i, 0] for i in pi),
                 sum(cm[i, j] for i in pi for j in pi if i != j)))
e = pd.DataFrame(errs, columns=["model", "insertions", "deletions", "substitutions"]).iloc[::-1]
left = np.zeros(len(e))
for c, col in zip(["deletions", "insertions", "substitutions"], ["#e4572e", "#f4a259", "#5b8e7d"]):
    ax.barh(e.model, e[c], left=left, color=col, label=c); left += e[c].values
ax.set_xlabel("number of errors on test"); ax.legend(fontsize=8); ax.set_title("What kind of mistakes?")
plt.tight_layout(); plt.savefig(P("results", "figures", "res_error_types.png")); plt.close()
e.to_csv(P("results", "tables", "error_types.csv"), index=False)


def sent_len_of_word(labels):
    out, cur = [], []
    for l in labels:
        cur.append(l)
        if l in ("PERIOD", "QUESTION"):
            out += [len(cur)] * len(cur); cur = []
    out += [len(cur)] * len(cur)
    return out


buckets = [(1, 10), (11, 20), (21, 30), (31, 45), (46, 1000)]
fig, ax = plt.subplots(figsize=(7, 3.2))
for tag in pick:
    f1s = []
    for lo_, hi_ in buckets:
        R, H = [], []
        for r, h in zip(refs["test"], hyps[tag]["test"]):
            sl = sent_len_of_word(r)
            R.append([x for x, s in zip(r, sl) if lo_ <= s <= hi_]); H.append([x for x, s in zip(h, sl) if lo_ <= s <= hi_])
        f1s.append(evaluate(R, H)[0]["micro"]["f1"])
    ax.plot(range(len(buckets)), f1s, marker="o", label=dict((a, b) for a, b, _ in SYSTEMS)[tag])
ax.set_xticks(range(len(buckets))); ax.set_xticklabels([f"{a}-{b}" if b < 1000 else f"{a}+" for a, b in buckets])
ax.set_xlabel("reference sentence length (words)"); ax.set_ylabel("micro-F1"); ax.legend(fontsize=7); ax.set_ylim(0, 1)
ax.set_title("Performance vs sentence length (test)")
plt.tight_layout(); plt.savefig(P("results", "figures", "res_by_sentence_length.png")); plt.close()

for tagger in ("distilbert_finetuned", "punctall_finetuned"):
    f = P("results", "tables", f"external_lm_{tagger}.json")
    if not os.path.exists(f):
        continue
    g = json.load(open(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, name in zip(axes, ("generic", "domain")):
        df = pd.DataFrame(g["grid"][name]["all"]).pivot(index="lambda", columns="beta", values="val_f1")
        im = ax.imshow(df.values * 100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(j, i, f"{df.values[i, j] * 100:.1f}", ha="center", va="center", fontsize=6, color="w")
        ax.set_xlabel("beta (punctuation bonus)"); ax.set_ylabel("lambda (LM weight)")
        ax.set_title(f"{name} GPT-2: val micro-F1 (%)", fontsize=9)
    plt.tight_layout(); plt.savefig(P("results", "figures", f"res_lm_grid_{tagger}.png")); plt.close()

best_ft = best_of("fine-tuned")
zs = best_of("pre-trained")
lines = ["# Error analysis examples (test set)\n",
         f"Legend: REF = gold, ZS = {zs} (zero-shot), FT = {best_ft} (best fine-tuned system)\n"]
rng2 = np.random.default_rng(3)
cand = [i for i, r in enumerate(refs["test"]) if 25 <= len(r) <= 70]
for i in rng2.choice(cand, size=min(8, len(cand)), replace=False):
    w = data["test"][i]["words"]
    lines += [f"\n### Test response {i}\n",
              f"- **INPUT:** {' '.join(w)}",
              f"- **REF:** {render(w, refs['test'][i], True)}",
              f"- **ZS:** {render(w, hyps[zs]['test'][i], True)}",
              f"- **FT:** {render(w, hyps[best_ft]['test'][i], True)}"]
open(P("results", "tables", "error_examples.md"), "w", encoding="utf-8").write("\n".join(lines))

missed = {l: collections.Counter() for l in PUNCT_LABELS}
for d, r, h in zip(data["test"], refs["test"], hyps[best_ft]["test"]):
    for w_, a, b in zip(d["words"], r, h):
        if a != "O" and a != b:
            missed[a][w_] += 1
json.dump({l: c.most_common(15) for l, c in missed.items()},
          open(P("results", "tables", "most_missed_words.json"), "w"), indent=1)
print(json.dumps(sig, indent=1))
