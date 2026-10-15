#!/usr/bin/env python3
"""Select 200 English first-turn requests from WildChat multi-turn conversations.

Filters: English, >=2 user turns, not toxic/redacted/moderation-flagged,
self-contained text (no uploads/links/long code), not jailbreak or
model-identity prompts, deduplicated by normalized prefix. Any IDs listed in
configs/seed_exclusions.txt (manual review rejections) are dropped and the
pool is re-sampled deterministically.
"""
import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ATTACH = re.compile(r"\b(attach\w*|upload\w*|screenshot|(this|the|my|that) (image|picture|photo|file|pdf|document|video|link|spreadsheet|excel|csv)|see (above|below)|following (code|article|passage|text|paragraph|data|table|essay))\b", re.I)
JAILBREAK = re.compile(r"\b(DAN|jailbreak|ignore (all |any )?(previous|prior)|developer mode|you are now|pretend (to be|you)|roleplay|role-play|role play|uncensored|nsfw)\b", re.I)
IDENTITY = re.compile(r"\b(chat ?gpt|gpt-?\d|openai|are you (an? )?(ai|bot|human)|which model|what model|who (made|created) you|your name)\b", re.I)
GENERATOR = re.compile(r"midjourney|stable diffusion|prompt generator|as a prompt|\bimage prompt", re.I)
CODEY = re.compile(r"```|[{};=<>\[\]$\\]")


def keep(conv_row):
    if conv_row["language"] != "English" or conv_row["toxic"] or conv_row["redacted"]:
        return None
    turns = conv_row["conversation"]
    users = [t for t in turns if t["role"] == "user"]
    if len(users) < 2:
        return None
    if any(m["flagged"] for m in conv_row["openai_moderation"]):
        return None
    tox = conv_row["detoxify_moderation"][0]
    if (tox["toxicity"] or 1) > 0.05 or (tox["sexual_explicit"] or 1) > 0.05:
        return None
    text = users[0]["content"].strip()
    if not (40 <= len(text) <= 1200) or len(text.split()) < 8:
        return None
    if sum(ord(ch) < 128 for ch in text) / len(text) < 0.98:
        return None
    if "http" in text or "www." in text or ATTACH.search(text) or JAILBREAK.search(text) \
            or IDENTITY.search(text) or GENERATOR.search(text):
        return None
    if len(CODEY.findall(text)) / len(text) > 0.01 or text.count("\n") > 12:
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "data/seeds.jsonl"))
    ap.add_argument("--n_train", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--rng", type=int, default=20261013)
    args = ap.parse_args()
    excl_path = ROOT / "configs/seed_exclusions.txt"
    excluded = set(excl_path.read_text().split()) if excl_path.exists() else set()

    pool, seen = [], set()
    cols = ["conversation_hash", "conversation", "language", "toxic", "redacted",
            "openai_moderation", "detoxify_moderation"]
    for f in args.parquet:
        for r in pq.read_table(f, columns=cols).to_pylist():
            text = keep(r)
            if text is None:
                continue
            key = re.sub(r"\W+", " ", text.lower())[:80]
            if key in seen:
                continue
            seen.add(key)
            pool.append({"source_id": r["conversation_hash"], "user_text": text})
    pool.sort(key=lambda s: s["source_id"])
    random.Random(args.rng).shuffle(pool)
    pool = [s for s in pool if s["source_id"] not in excluded]
    n = args.n_train + args.n_test
    chosen = pool[:n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, s in enumerate(chosen):
            split = "train" if i < args.n_train else "test"
            f.write(json.dumps({"seed_id": f"s{i:04d}", "split": split, **s}, ensure_ascii=False) + "\n")
    print(f"pool={len(pool)} excluded={len(excluded)} wrote={len(chosen)} -> {args.out}")


if __name__ == "__main__":
    main()
