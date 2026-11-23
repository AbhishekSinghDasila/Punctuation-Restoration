from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .labels import CHAR2LABEL, ID2LABEL, LABEL2ID, LABELS, end_of_input
from .paths import resolve


def fold_label(name: str) -> str:
    n = str(name).strip()
    up = n.upper()
    if up in LABEL2ID:
        return up
    if up in ("0", "O", "LABEL_0", ""):
        return "O"
    for c in n:
        if c in CHAR2LABEL:
            return CHAR2LABEL[c]
    return "O"


class Tagger:
    def __init__(self, name_or_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name_or_path = resolve(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=True) \
            if "roberta" in name_or_path.lower() and "xlm" not in name_or_path.lower() \
            else AutoTokenizer.from_pretrained(name_or_path)
        cfg_probe = AutoModelForTokenClassification.from_pretrained
        try:
            self.model = cfg_probe(name_or_path)
            native = {int(i): l for i, l in self.model.config.id2label.items()}

            if all(l.startswith("LABEL_") for l in native.values()):
                raise ValueError("no punctuation head")
        except Exception:
            self.model = cfg_probe(name_or_path, num_labels=len(LABELS),
                                   id2label=ID2LABEL, label2id=LABEL2ID,
                                   ignore_mismatched_sizes=True)
            native = dict(ID2LABEL)
        self.native_id2label = native

        self.fold = np.array([LABEL2ID[fold_label(native[j])] for j in range(len(native))])

        self.target_id = {}
        for ours in LABELS:
            exact = [j for j, n in native.items() if fold_label(n) == ours]
            pref = {"O": ("0", "O"), "COMMA": (",", "COMMA"), "PERIOD": (".", "PERIOD"),
                    "QUESTION": ("?", "QUESTION")}[ours]
            best = [j for j in exact if native[j].strip().upper() in pref]
            self.target_id[ours] = (best or exact)[0]
        self.model.to(self.device)

    def encode(self, words, labels=None, max_length=256):
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                             max_length=max_length)
        word_ids = enc.word_ids()
        if labels is not None:
            lab, prev = [], None
            for wid in word_ids:
                if wid is None or wid == prev:
                    lab.append(-100)
                else:
                    lab.append(self.target_id[labels[wid]])
                prev = wid
            enc["labels"] = lab
        return enc, word_ids

    def train(self, samples, out_dir, epochs=3, lr=5e-5, batch_size=16, max_length=256,
              warmup=0.1, weight_decay=0.01, log_every=50, seed=42, log_fn=print):
        torch.manual_seed(seed)
        ds = _EncodedDataset([self.encode(s["words"], s["labels"], max_length)[0] for s in samples])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self._collate)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total = epochs * len(dl)
        sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
        history, step, t0 = [], 0, time.time()
        self.model.train()
        for ep in range(epochs):
            run = 0.0
            for b in dl:
                b = {k: v.to(self.device) for k, v in b.items()}
                loss = self.model(**b).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                run += loss.item()
                history.append({"step": step, "epoch": ep + step / len(dl) - ep, "loss": loss.item()})
                if step % log_every == 0:
                    log_fn(f"  step {step}/{total}  loss {run / log_every:.4f}  "
                           f"{(time.time() - t0) / step:.2f}s/step")
                    run = 0.0
        self.model.eval()
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        return history

    def _collate(self, batch):
        m = max(len(x["input_ids"]) for x in batch)
        pad = self.tokenizer.pad_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in batch:
            n = len(x["input_ids"])
            out["input_ids"].append(x["input_ids"] + [pad] * (m - n))
            out["attention_mask"].append([1] * n + [0] * (m - n))
            if "labels" in x:
                out["labels"].append(x["labels"] + [-100] * (m - n))
        if not out["labels"]:
            del out["labels"]
        return {k: torch.tensor(v) for k, v in out.items()}

    @torch.no_grad()
    def predict_proba(self, docs_words, window=100, stride=50, batch_size=32, max_length=256):
        self.model.eval()
        jobs = []
        for di, words in enumerate(docs_words):
            starts = list(range(0, max(1, len(words) - window + stride), stride)) or [0]
            for s in starts:
                jobs.append((di, s, words[s:s + window]))
        probs = [np.zeros((len(w), len(LABELS)), dtype=np.float32) for w in docs_words]
        best_margin = [np.full(len(w), -1.0) for w in docs_words]
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            encs = [self.encode(w, None, max_length) for _, _, w in chunk]
            batch = self._collate([dict(e) for e, _ in encs])
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "token_type_ids"}
            logits = self.model(**batch).logits.float()
            p_native = torch.softmax(logits, -1).cpu().numpy()
            for (di, s, w), (enc, wids), p in zip(chunk, encs, p_native):
                folded = np.zeros((p.shape[0], len(LABELS)), dtype=np.float32)
                np.add.at(folded.T, self.fold, p.T)
                prev = None
                for t, wid in enumerate(wids):
                    if wid is None or wid == prev:
                        prev = wid
                        continue
                    prev = wid
                    g = s + wid

                    left = wid if s > 0 else 10 ** 6
                    right = len(w) - 1 - wid if s + len(w) < len(docs_words[di]) else 10 ** 6
                    margin = min(left, right)
                    if margin > best_margin[di][g]:
                        best_margin[di][g] = margin
                        probs[di][g] = folded[t]

        for p in probs:
            p[p.sum(1) == 0] = np.eye(len(LABELS))[0]
        return probs

    def predict(self, docs_words, end_constraint=True, **kw):
        out = []
        for p in self.predict_proba(docs_words, **kw):
            lab = [LABELS[i] for i in p.argmax(1)]
            out.append(end_of_input(lab, p) if end_constraint else lab)
        return out


class _EncodedDataset(Dataset):
    def __init__(self, encs):
        self.encs = [{k: list(v) for k, v in e.items() if k in ("input_ids", "labels")} for e in encs]

    def __len__(self):
        return len(self.encs)

    def __getitem__(self, i):
        return self.encs[i]
