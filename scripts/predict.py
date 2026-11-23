import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from punct.data import _normalise_word
from punct.labels import LABELS, end_of_input, render
from punct.tagger import Tagger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def to_words(text):
    words = []
    for tok in text.split():
        words += _normalise_word(re.sub(r"[^\w'\-/]", "", tok))
    return words


def punctuate(text, tagger, lm=None, lam=0.2, beta=1.0):
    words = to_words(text)
    probs = tagger.predict_proba([words])
    if lm is None:
        labels = end_of_input([LABELS[i] for i in probs[0].argmax(1)], probs[0])
    else:
        from punct.lm_rescore import build_nbest_cache, decode_from_cache
        labels = decode_from_cache(build_nbest_cache([words], probs, lm), lam, beta)[0]
    return render(words, labels, capitalize=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "punctall_finetuned"))
    ap.add_argument("--lm", default=None, help="path/id of a GPT-2 model for rescoring (optional)")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    tagger = Tagger(a.model)
    lm = None
    if a.lm:
        from punct.lm_rescore import ExternalLM
        lm = ExternalLM(a.lm)
    print(punctuate(" ".join(a.text), tagger, lm, a.lam, a.beta))
