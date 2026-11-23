import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from punct.data import load_jsonl, load_raw
from punct.labels import LABELS, PUNCT_LABELS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(ROOT, "results", "tables")
COL = {"O": "#9aa5b1", "COMMA": "#2a6fdb", "PERIOD": "#1b998b", "QUESTION": "#e4572e"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})

train = load_jsonl(os.path.join(ROOT, "data", "processed", "train.jsonl"))
splits = {s: load_jsonl(os.path.join(ROOT, "data", "processed", f"{s}.jsonl")) for s in ("train", "val", "test")}
raw = load_raw(os.path.join(ROOT, "data", "raw", "train.csv"))
summary = {}

summary["raw_columns"] = list(raw.columns)
summary["raw_rows"] = len(raw)
summary["raw_missing"] = raw.isna().sum().to_dict()
summary["contexts_unique"] = int(raw.Context.nunique())
summary["responses_unique"] = int(raw.Response.nunique())
summary["responses_per_context"] = raw.groupby("Context").Response.nunique().describe().round(2).to_dict()

cnt = {s: collections.Counter(l for d in docs for l in d["labels"]) for s, docs in splits.items()}
summary["label_counts"] = {s: dict(c) for s, c in cnt.items()}
summary["label_share_train"] = {l: cnt["train"][l] / sum(cnt["train"].values()) for l in LABELS}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
c = cnt["train"]
ax[0].bar(LABELS, [c[l] for l in LABELS], color=[COL[l] for l in LABELS])
for i, l in enumerate(LABELS):
    ax[0].text(i, c[l], f"{c[l]:,}\n({100 * c[l] / sum(c.values()):.1f}%)", ha="center", va="bottom", fontsize=8)
ax[0].set_title("Label distribution (train, all words)"); ax[0].set_ylabel("words")
ax[0].set_ylim(0, c["O"] * 1.18)
ax[1].bar(PUNCT_LABELS, [c[l] for l in PUNCT_LABELS], color=[COL[l] for l in PUNCT_LABELS])
for i, l in enumerate(PUNCT_LABELS):
    ax[1].text(i, c[l], f"{c[l]:,}", ha="center", va="bottom", fontsize=8)
ax[1].set_title("Punctuation classes only (O removed)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_label_distribution.png")); plt.close()

share = pd.DataFrame({s: {l: cnt[s][l] / sum(cnt[s].values()) * 100 for l in PUNCT_LABELS} for s in splits})
summary["punct_share_per_split_pct"] = share.round(2).to_dict()

n_words = np.array([len(d["words"]) for d in train])
sent_lens = []
for d in train:
    k = 0
    for l in d["labels"]:
        k += 1
        if l in ("PERIOD", "QUESTION"):
            sent_lens.append(k); k = 0
sent_lens = np.array(sent_lens)
comma_per_sent = []
for d in train:
    k = 0
    for l in d["labels"]:
        if l == "COMMA": k += 1
        if l in ("PERIOD", "QUESTION"):
            comma_per_sent.append(k); k = 0
summary["words_per_response"] = pd.Series(n_words).describe().round(1).to_dict()
summary["words_per_sentence"] = pd.Series(sent_lens).describe().round(1).to_dict()
summary["commas_per_sentence_mean"] = float(np.mean(comma_per_sent))
summary["vocab_size_train"] = len({w for d in train for w in d["words"]})

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(n_words, bins=50, color="#2a6fdb"); ax[0].set_title("Words per response")
ax[0].axvline(np.median(n_words), color="k", ls="--", lw=1); ax[0].text(np.median(n_words), ax[0].get_ylim()[1] * .9, f" median {np.median(n_words):.0f}")
ax[1].hist(np.clip(sent_lens, 0, 80), bins=40, color="#1b998b"); ax[1].set_title("Words per sentence (clipped at 80)")
ax[1].axvline(np.median(sent_lens), color="k", ls="--", lw=1); ax[1].text(np.median(sent_lens), ax[1].get_ylim()[1] * .9, f" median {np.median(sent_lens):.0f}")
vals, freqs = np.unique(np.clip(comma_per_sent, 0, 8), return_counts=True)
ax[2].bar(vals, freqs, color="#2a6fdb"); ax[2].set_title("Commas per sentence (8 = 8+)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_lengths.png")); plt.close()

before = {l: collections.Counter() for l in PUNCT_LABELS}
after = {l: collections.Counter() for l in PUNCT_LABELS}
qfirst = collections.Counter()
for d in train:
    w, lab = d["words"], d["labels"]
    start = 0
    for i, l in enumerate(lab):
        if l in before:
            before[l][w[i]] += 1
            if i + 1 < len(w):
                after[l][w[i + 1]] += 1
        if l in ("PERIOD", "QUESTION"):
            if l == "QUESTION":
                qfirst[w[start]] += 1
            start = i + 1
summary["top_word_before"] = {l: before[l].most_common(10) for l in before}
summary["top_word_after"] = {l: after[l].most_common(10) for l in after}
summary["question_first_word"] = qfirst.most_common(12)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, (title, counter, col) in zip(ax, [
        ("Word AFTER a comma", after["COMMA"], COL["COMMA"]),
        ("Word BEFORE a period", before["PERIOD"], COL["PERIOD"]),
        ("FIRST word of a question", qfirst, COL["QUESTION"])]):
    items = counter.most_common(12)[::-1]
    a.barh([k for k, _ in items], [v for _, v in items], color=col); a.set_title(title)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_lexical_cues.png")); plt.close()

pos = {l: [] for l in PUNCT_LABELS}
for d in train:
    n = len(d["labels"])
    for i, l in enumerate(d["labels"][:-1]):
        if l in pos:
            pos[l].append(i / max(1, n - 1))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for l in PUNCT_LABELS:
    ax.hist(pos[l], bins=20, density=True, histtype="step", lw=2, color=COL[l], label=l)
ax.set_xlabel("relative position in response (0 = start, 1 = end)"); ax.set_ylabel("density")
ax.set_title("Where punctuation occurs inside a response"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_position.png")); plt.close()

raw_resp = raw.Response.dropna().drop_duplicates()
issues = {
    "missing_space_after_sentence_end": int(raw_resp.str.count(r"[a-z][.?!][A-Z]").sum()),
    "non_breaking_spaces": int(raw_resp.str.count(" ").sum()),
    "urls": int(raw_resp.str.count(r"https?://|www\.").sum()),
    "curly_quotes_apostrophes": int(raw_resp.str.count("[‘’“”]").sum()),
    "responses_with_newlines": int(raw_resp.str.contains("\n").sum()),
    "spanish_responses": summary.get("non_english", None),
}
rep = json.load(open(os.path.join(TAB, "dataset_report.json")))
issues["spanish_responses"] = rep["non_english"]
summary["raw_issues"] = issues
fig, ax = plt.subplots(figsize=(7, 3))
k = list(issues); v = [issues[x] for x in k]
ax.barh(k[::-1], v[::-1], color="#e4572e")
for i, x in enumerate(v[::-1]): ax.text(x, i, f" {x:,}", va="center", fontsize=8)
ax.set_title("Problems found in the raw Response column"); ax.set_xscale("log")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_raw_issues.png")); plt.close()

funnel = [("raw rows", rep["raw_rows"]), ("non-missing", rep["raw_rows"] - rep["missing_response"]),
          ("unique responses", rep["unique_responses"]),
          ("English", rep["unique_responses"] - rep["non_english"]),
          ("unique after cleaning", rep["final_responses"])]
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh([f[0] for f in funnel][::-1], [f[1] for f in funnel][::-1], color="#2a6fdb")
for i, (_, x) in enumerate(funnel[::-1]): ax.text(x, i, f" {x:,}", va="center", fontsize=8)
ax.set_title("Cleaning funnel (rows kept after each step)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_cleaning_funnel.png")); plt.close()

with open(os.path.join(TAB, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps({k: summary[k] for k in ("label_share_train", "words_per_response", "words_per_sentence",
                                          "question_first_word", "raw_issues", "vocab_size_train")}, indent=1, default=str))
