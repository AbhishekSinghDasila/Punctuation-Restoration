LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
PUNCT_LABELS = ["COMMA", "PERIOD", "QUESTION"]

CHAR2LABEL = {
    ",": "COMMA", ";": "COMMA", ":": "COMMA", "-": "COMMA",
    ".": "PERIOD", "!": "PERIOD",
    "?": "QUESTION",
}
LABEL2CHAR = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}

PRIORITY = {"QUESTION": 3, "PERIOD": 2, "COMMA": 1, "O": 0}


def end_of_input(labels, probs=None):
    labels = list(labels)
    if labels and labels[-1] in ("O", "COMMA"):
        q = probs is not None and probs[-1][LABEL2ID["QUESTION"]] > probs[-1][LABEL2ID["PERIOD"]]
        labels[-1] = "QUESTION" if q else "PERIOD"
    return labels


def render(words, labels, capitalize=False):
    out, start = [], True
    for w, l in zip(words, labels):
        if capitalize:
            if start or w == "i" or w.startswith("i'"):
                w = w[:1].upper() + w[1:]
        out.append(w + LABEL2CHAR[l])
        start = l in ("PERIOD", "QUESTION")
    return " ".join(out)
