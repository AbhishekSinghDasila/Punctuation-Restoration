import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEEDED = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "vocab.txt", "vocab.json", "merges.txt", "sentencepiece.bpe.model", "generation_config.json")
CHUNK = 8 * 1024 * 1024
proxy = urllib.request.ProxyHandler()
opener = urllib.request.build_opener(proxy)


def get(url, rng=None, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-download"} |
                                         ({"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}))
            with opener.open(req, timeout=60) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1:
                raise IOError("short read")
            return data
        except Exception as e:
            time.sleep(2 + 2 * t)
    raise RuntimeError(f"failed {url} {rng}")


def download(repo, workers=24):
    out = os.path.join(ROOT, "models", "hf", repo.replace("/", "__"))
    os.makedirs(out, exist_ok=True)
    tree = json.loads(get(f"https://huggingface.co/api/models/{repo}/tree/main"))
    files = {f["path"]: f["size"] for f in tree if f["type"] == "file"}
    weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    for name in [f for f in NEEDED if f in files] + [weight]:
        dst = os.path.join(out, name)
        size = files[name]
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        t0 = time.time()
        if size < CHUNK:
            data = get(url)
            open(dst, "wb").write(data)
        else:
            ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.truncate(size)
            done = 0
            with cf.ThreadPoolExecutor(workers) as ex, open(tmp, "r+b") as f:
                futs = {ex.submit(get, url, r): r for r in ranges}
                for fu in cf.as_completed(futs):
                    r = futs[fu]
                    f.seek(r[0]); f.write(fu.result())
                    done += r[1] - r[0] + 1
                    print(f"\r  {repo}/{name}: {done / 1e6:.0f}/{size / 1e6:.0f} MB "
                          f"({done / 1e6 / (time.time() - t0):.1f} MB/s)", end="", flush=True)
            os.replace(tmp, dst)
            print()
        print(f"  {repo}/{name} {size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        print("->", download(repo))
