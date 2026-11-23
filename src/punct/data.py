from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from .labels import CHAR2LABEL, PRIORITY, render


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


_TRANSLATE = {
    " ": " ", "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": " - ", "—": " - ", "…": ". ", "·": "\n", "•": "\n",
    "\r": "\n", "\t": " ",
}
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov)(/\S*)?)", re.I)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
BASE64_RE = re.compile(r"(data:image/\S+|[A-Za-z0-9+/=]{40,})")
HTML_RE = re.compile(r"<[^>]+>")
EMOTICON_RE = re.compile(r"(?<!\w)[:;=][-']?[)(DPp/\\|](?!\w)")

GLUED_END_RE = re.compile(r"([a-z0-9)\"'])([.?!]+)([A-Z])")
GLUED_COMMA_RE = re.compile(r"([a-zA-Z])([,;:])([a-zA-Z])")
REPEAT_PUNCT_RE = re.compile(r"([.?!,])[.?!,]+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    for a, b in _TRANSLATE.items():
        text = text.replace(a, b)
    text = HTML_RE.sub(" ", text)
    text = BASE64_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = EMOTICON_RE.sub(" ", text)
    text = GLUED_END_RE.sub(r"\1\2 \3", text)
    text = GLUED_COMMA_RE.sub(r"\1\2 \3", text)

    text = re.sub(r"[?!]*\?[?!]*", "?", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)

    lines = [l.strip(" -*#>") for l in text.split("\n")]
    lines = [l for l in lines if l]
    lines = [l if re.search(r"[.?!:,;]['\")\]]*$", l) else l + "." for l in lines]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SPANISH_HINTS = {"que", "de", "el", "la", "los", "las", "por", "una", "es", "mi", "con", "para", "pero", "muy"}


def is_english(text: str) -> bool:
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    if not words:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127) / max(1, len(text))
    spanish = sum(w in SPANISH_HINTS for w in words) / len(words)
    return non_ascii < 0.01 and spanish < 0.08


ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "jr", "sr", "vs", "etc", "eg", "ie", "am", "pm", "approx", "dept", "inc"}
WORD_CHARS_RE = re.compile(r"[^a-z0-9']")


def _normalise_word(tok: str) -> list[str]:
    tok = tok.lower().replace("&", " and ").replace("%", " percent ")
    parts = re.split(r"[-/_+]+", tok)
    out = []
    for p in parts:
        for q in p.split():
            q = WORD_CHARS_RE.sub("", q).strip("'")
            if q:
                out.append(q)
    return out


def to_words_labels(text: str) -> tuple[list[str], list[str]]:
    words, labels = [], []
    for raw in text.split():

        m = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", raw)
        core, trail = m.group(2), m.group(3)

        if not core:
            lab = _label_from_chars(trail + m.group(1))
            if words and PRIORITY[lab] > PRIORITY[labels[-1]]:
                labels[-1] = lab
            continue
        pieces = _normalise_word(core)
        if not pieces:
            continue
        lab = _label_from_chars(trail)

        bare = core.lower().replace(".", "")
        if lab == "PERIOD" and (bare in ABBREVIATIONS or re.fullmatch(r"([a-z]\.){2,}", core.lower() + ".")):
            lab = "O"
        words.extend(pieces)
        labels.extend(["O"] * (len(pieces) - 1) + [lab])
    if labels and labels[-1] in ("O", "COMMA"):
        labels[-1] = "PERIOD"
    return words, labels


def _label_from_chars(chars: str) -> str:
    best = "O"
    for c in chars:
        lab = CHAR2LABEL.get(c, "O")
        if PRIORITY[lab] > PRIORITY[best]:
            best = lab
    return best


@dataclass
class BuildReport:
    raw_rows: int = 0
    missing_response: int = 0
    duplicate_rows: int = 0
    unique_responses: int = 0
    non_english: int = 0
    too_short: int = 0
    duplicates_after_cleaning: int = 0
    final_responses: int = 0
    split_sizes: dict = field(default_factory=dict)


def build_documents(csv_path: str, min_words: int = 5, seed: int = 42,
                    split=(0.8, 0.1, 0.1)) -> tuple[dict, BuildReport]:

    rep = BuildReport()
    df = load_raw(csv_path)
    rep.raw_rows = len(df)
    rep.missing_response = int(df["Response"].isna().sum())
    rep.duplicate_rows = int(df.duplicated().sum())
    resp = df["Response"].dropna()
    resp = resp[resp.str.strip() != ""].drop_duplicates()
    rep.unique_responses = len(resp)

    docs, seen = [], set()
    for idx, raw in resp.items():
        clean = clean_text(raw)
        if not is_english(clean):
            rep.non_english += 1
            continue
        words, labels = to_words_labels(clean)
        if len(words) < min_words:
            rep.too_short += 1
            continue
        key = " ".join(words)
        if key in seen:
            rep.duplicates_after_cleaning += 1
            continue
        seen.add(key)
        docs.append({
            "id": int(idx), "raw": raw, "clean": clean, "words": words, "labels": labels,
            "input_text": " ".join(words), "target_text": render(words, labels),
        })
    rep.final_responses = len(docs)

    rng = random.Random(seed)
    rng.shuffle(docs)
    n = len(docs)
    n_tr, n_va = int(split[0] * n), int(split[1] * n)
    splits = {"train": docs[:n_tr], "val": docs[n_tr:n_tr + n_va], "test": docs[n_tr + n_va:]}
    rep.split_sizes = {k: len(v) for k, v in splits.items()}
    return splits, rep


def make_windows(docs: list[dict], min_len: int = 24, max_len: int = 96, passes: int = 2,
                 seed: int = 0) -> list[dict]:

    rng = random.Random(seed)
    samples = []
    for p in range(passes):
        order = list(range(len(docs)))
        rng.shuffle(order)
        words = [w for i in order for w in docs[i]["words"]]
        labels = [l for i in order for l in docs[i]["labels"]]
        pos = rng.randint(0, max_len // 2) if p else 0
        while pos < len(words):
            L = rng.randint(min_len, max_len)
            samples.append({"words": words[pos:pos + L], "labels": labels[pos:pos + L]})
            pos += L
    return samples


def save_jsonl(items: list[dict], path: str, keys=None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            if keys:
                it = {k: it[k] for k in keys}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]
