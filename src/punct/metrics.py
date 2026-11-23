from __future__ import annotations

import numpy as np

from .labels import LABEL2ID, LABELS, PUNCT_LABELS


def confusion(refs, hyps):
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for r_doc, h_doc in zip(refs, hyps):
        for r, h in zip(r_doc, h_doc):
            cm[LABEL2ID[r], LABEL2ID[h]] += 1
    return cm


def scores_from_confusion(cm):
    out = {}
    tp_all = fp_all = fn_all = 0
    for lab in PUNCT_LABELS:
        i = LABEL2ID[lab]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[lab] = {"precision": p, "recall": r, "f1": f, "support": int(cm[i, :].sum())}
        tp_all += tp; fp_all += fp; fn_all += fn

    p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    out["micro"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "support": int(sum(out[l]["support"] for l in PUNCT_LABELS))}
    out["macro"] = {k: float(np.mean([out[l][k] for l in PUNCT_LABELS]))
                    for k in ("precision", "recall", "f1")}
    o = LABEL2ID["O"]
    punct = [LABEL2ID[l] for l in PUNCT_LABELS]
    subs = sum(cm[i, j] for i in punct for j in punct if i != j)
    dels = sum(cm[i, o] for i in punct)
    ins = sum(cm[o, j] for j in punct)
    ref = sum(cm[i, :].sum() for i in punct)
    out["SER"] = float((subs + dels + ins) / ref) if ref else 0.0
    out["accuracy_all_words"] = float(np.trace(cm) / cm.sum())
    return out


def evaluate(refs, hyps):
    cm = confusion(refs, hyps)
    return scores_from_confusion(cm), cm


def flat_row(name, s):
    row = {"model": name}
    for lab in PUNCT_LABELS:
        row[f"{lab}_P"] = s[lab]["precision"]
        row[f"{lab}_R"] = s[lab]["recall"]
        row[f"{lab}_F1"] = s[lab]["f1"]
    row["micro_P"] = s["micro"]["precision"]
    row["micro_R"] = s["micro"]["recall"]
    row["micro_F1"] = s["micro"]["f1"]
    row["macro_F1"] = s["macro"]["f1"]
    row["SER"] = s["SER"]
    return row
