"""Config, I/O and text normalization shared by all stages."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "configs/pilot.json", profile="full"):
    c = json.loads(Path(path).read_text())
    p = c["profiles"][profile]
    names = p.get("models", [m["name"] for m in c["models"]])
    c["models"] = [m for m in c["models"] if m["name"] in names]
    c["profile"] = dict(p, name=profile)
    return c


def model_spec(c, name):
    for m in c["models"] + [c["simulator"]]:
        if m["name"] == name:
            return m
    raise KeyError(name)


def relation(c, a, b):
    if a == b:
        return "none"
    fam = {m["name"]: m["family"] for m in c["models"]}
    return "same" if fam[a] == fam[b] else "cross"


def select_seeds(c, seeds):
    """Profile seeds: the first n train and n test seeds of the frozen file."""
    p = c["profile"]
    train = [s for s in seeds if s["split"] == "train"][: p["train_seeds"]]
    test = [s for s in seeds if s["split"] == "test"][: p["test_seeds"]]
    if len(train) != p["train_seeds"] or len(test) != p["test_seeds"]:
        raise ValueError("Seed file too small for profile")
    return train + test


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def stable_int(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


SPECIAL = re.compile(r"<\|[a-z_]+\|>|<(?:start|end)_of_turn>|<eos>|<bos>|</?s>")
THINK = re.compile(r"<think>.*?</think>", re.S)


def normalize(text):
    """Strip leftover special tokens / thinking blocks and unify whitespace."""
    t = THINK.sub("", text)
    t = t.replace("<think>", "").replace("</think>", "")
    t = SPECIAL.sub("", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def dialogues(rows):
    """Group assistant-turn rows into {(seed_id, branch): [rows sorted by turn]}."""
    out = {}
    for r in rows:
        out.setdefault((r["seed_id"], r["branch"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["turn"])
    return out


def to_messages(turns):
    msgs = []
    for r in turns:
        msgs.append({"role": "user", "content": r["user_text"]})
        msgs.append({"role": "assistant", "content": r["text_norm"]})
    return msgs
