from __future__ import annotations

import heapq
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .labels import LABELS, end_of_input, render
from .paths import resolve


def nbest(logp: np.ndarray, k: int = 8, min_prob: float = 0.05):
    beams = [(0.0, [])]
    for row in logp:
        cand = [i for i in np.argsort(-row) if row[i] >= math.log(min_prob)] or [int(row.argmax())]
        nxt = [(s + row[i], seq + [i]) for s, seq in beams for i in cand]
        beams = heapq.nlargest(k, nxt, key=lambda x: x[0])
    return beams


class ExternalLM:
    def __init__(self, name_or_path="distilgpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name_or_path = resolve(name_or_path)
        self.tok =AutoTokenizer.from_pretrained(name_or_path)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path).to(self.device).eval()

    def finetune(self, texts, out_dir, epochs=2, lr=5e-5, block=128, batch_size=8, log_fn=print):
        ids = self.tok("\n".join(texts) + self.tok.eos_token)["input_ids"]
        blocks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]
        dl = DataLoader(_Blocks(blocks), batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total = epochs * len(dl)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        self.model.train()
        step, hist = 0, []
        for ep in range(epochs):
            for x in dl:
                x = x.to(self.device)
                loss = self.model(input_ids=x, labels=x).loss
                loss.backward(); opt.step(); sched.step(); opt.zero_grad()
                step += 1
                hist.append({"step": step, "loss": loss.item()})
                if step % 50 == 0:
                    log_fn(f"  LM step {step}/{total} loss {np.mean([h['loss'] for h in hist[-50:]]):.3f}")
        self.model.eval()
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir); self.tok.save_pretrained(out_dir)
        return hist

    @torch.no_grad()
    def perplexity(self, texts, block=128):
        ids = self.tok("\n".join(texts))["input_ids"]
        nll, n = 0.0, 0
        for i in range(0, len(ids) - 1, block):
            x = torch.tensor([ids[i:i + block + 1]], device=self.device)
            if x.shape[1] < 2:
                break
            logits = self.model(input_ids=x[:, :-1]).logits
            lp = torch.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            nll -= lp.sum().item(); n += lp.numel()
        return math.exp(nll / n)

    @torch.no_grad()
    def score(self, prefixes, continuations, batch_size=32, prefix_tokens=48):
        all_seqs, all_starts = [], []
        for p, c in zip(prefixes, continuations):
            pid = self.tok(p)["input_ids"][-prefix_tokens:] if p else []
            cid = self.tok((" " if p else "") + c)["input_ids"]
            pid = [self.tok.eos_token_id] + pid
            all_seqs.append(pid + cid); all_starts.append(len(pid))
        order = sorted(range(len(all_seqs)), key=lambda j: len(all_seqs[j]))
        out = [0.0] * len(all_seqs)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            seqs = [all_seqs[j] for j in idx]
            m = max(map(len, seqs))
            x = torch.full((len(seqs), m), self.tok.eos_token_id, dtype=torch.long)
            att = torch.zeros_like(x)
            for j, s in enumerate(seqs):
                x[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            x, att = x.to(self.device), att.to(self.device)
            logits = self.model(input_ids=x, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            for j, s, g in zip(range(len(seqs)), seqs, idx):
                out[g] = lp[j, all_starts[g] - 1:len(s) - 1].sum().item()
        return out


class _Blocks(Dataset):
    def __init__(self, b): self.b = b
    def __len__(self): return len(self.b)
    def __getitem__(self, i): return torch.tensor(self.b[i])


def build_nbest_cache(docs_words, docs_probs, lm: ExternalLM, seg_len=32, k=8, log_fn=None):
    cache = []
    for di, (words, probs) in enumerate(zip(docs_words, docs_probs)):
        logp = np.log(np.clip(probs, 1e-9, 1.0))
        greedy = [LABELS[i] for i in probs.argmax(1)]
        segs = []
        for s in range(0, len(words), seg_len):
            w = words[s:s + seg_len]
            hyps = nbest(logp[s:s + seg_len], k=k)
            prev = render(words[max(0, s - seg_len):s], greedy[max(0, s - seg_len):s])
            segs.append({"start": s, "prefix": prev,
                         "hyps": [{"labels": [LABELS[i] for i in seq], "tag": sc,
                                   "text": render(w, [LABELS[i] for i in seq])} for sc, seq in hyps]})
        cache.append(segs)

    flat = [(seg["prefix"], h["text"]) for segs in cache for seg in segs for h in seg["hyps"]]
    scores = lm.score([p for p, _ in flat], [c for _, c in flat])
    it = iter(scores)
    for segs in cache:
        for seg in segs:
            for h in seg["hyps"]:
                h["lm"] = next(it)
                h["n_punct"] = sum(l != "O" for l in h["labels"])
    return cache


def decode_from_cache(cache, lam=0.3, beta=0.0, end_constraint=True):
    score = lambda h: h["tag"] + lam * h["lm"] + beta * h["n_punct"]
    out = []
    for segs in cache:
        labels = []
        for i, seg in enumerate(segs):
            hyps = seg["hyps"]
            if end_constraint and i == len(segs) - 1:
                hyps = [h for h in hyps if h["labels"][-1] in ("PERIOD", "QUESTION")] or hyps
            labels.extend(max(hyps, key=score)["labels"])
        out.append(end_of_input(labels) if end_constraint else labels)
    return out
