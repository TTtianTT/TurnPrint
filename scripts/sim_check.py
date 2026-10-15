#!/usr/bin/env python3
"""Regenerate simulator follow-ups for existing dialogues and flag role slips.

Usage (inside a 1-GPU Slurm step): python scripts/sim_check.py RUN_DIR
Writes RUN_DIR/sim_check.jsonl and prints a summary plus examples.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turnprint.common import dialogues, load_config, read_jsonl  # noqa: E402
from turnprint.generate import clean_user, simulator_messages  # noqa: E402

SLIP = re.compile(r"(happy to help|I can help|let me know if|here (are|is) (a|some|the)|could you (please )?(provide|tell me more about your)|"
                  r"to help me (better )?understand|let's take a look|great question|^\s*(sure|certainly|absolutely)[,!])", re.I)
LEAK = re.compile(r"(cut off|got cut|continue (where|from)|qwen|gemma|llama|chatgpt|as an ai)", re.I)


def main():
    run = Path(sys.argv[1])
    c = load_config(profile="smoke")
    rows = [r for f in sorted((run / "gen").glob("clean_*.jsonl")) for r in read_jsonl(f)]
    jobs = []
    for (sid, br), turns in sorted(dialogues(rows).items()):
        for t in range(2, len(turns) + 1):
            jobs.append((sid, br, t, simulator_messages(c, turns[: t - 1])))
    from vllm import LLM, SamplingParams
    llm = LLM(c["simulator"]["path"], dtype="bfloat16", max_model_len=c["generation"]["max_model_len"],
              gpu_memory_utilization=0.85, seed=0, attention_backend="FLASH_ATTN")
    g = c["generation"]
    sp = SamplingParams(temperature=g["temperature"], top_p=1.0, max_tokens=g["user_max_tokens"], seed=0)
    outs = llm.chat([j[3] for j in jobs], sp, use_tqdm=False)
    res = []
    for (sid, br, t, _), o in zip(jobs, outs):
        text = clean_user(o.outputs[0].text)
        res.append({"seed_id": sid, "model": br, "turn": t, "text": text, "slip": bool(SLIP.search(text)),
                    "leak": bool(LEAK.search(text)), "words": len(text.split())})
    with open(run / "sim_check.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in res)
    n = len(res)
    print(f"n={n} slip={sum(r['slip'] for r in res)} leak={sum(r['leak'] for r in res)} "
          f"empty={sum(not r['text'] for r in res)} mean_words={sum(r['words'] for r in res) / n:.1f}")
    for m in sorted({r["model"] for r in res}):
        rs = [r for r in res if r["model"] == m]
        print(f"  {m}: slip={sum(r['slip'] for r in rs)}/{len(rs)} mean_words={sum(r['words'] for r in rs) / len(rs):.1f}")
    print("--- flagged")
    for r in res:
        if r["slip"] or r["leak"]:
            print(f"[{r['model']} {r['seed_id']} t{r['turn']}] {r['text'][:250]}")
    print("--- sample")
    for r in res[::5]:
        print(f"[{r['model']} {r['seed_id']} t{r['turn']}] {r['text'][:250]}")


if __name__ == "__main__":
    main()
