import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    local = os.path.join(ROOT, "models", "hf", name_or_path.replace("/", "__"))
    return local if os.path.exists(os.path.join(local, "config.json")) else name_or_path
