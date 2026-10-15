"""D2: local likelihood of every assistant turn under one candidate model.

For each dialogue and turn t, the candidate renders the history up to user
turn t with ITS OWN chat template (generation prompt, thinking disabled), and
we read the log-probabilities of the turn's normalized text appended to that
prompt. This is exactly the context the candidate would have generated from,
so Qwen3's "empty think block only on the current turn" rendering is handled
without special cases. End-of-turn tokens are not scored.

Output: <run>/d2/<model>.jsonl with one row per (seed_id, branch, turn):
n_tok, sum_lp, mean_lp, n_bytes.
"""
import argparse
import json
import time
from pathlib import Path

from .common import ROOT, dialogues, load_config, model_spec, read_jsonl, to_messages, write_json


def build_requests(c, tok, run):
    rows = []
    for f in sorted((Path(run) / "gen").glob("*.jsonl")):
        rows += read_jsonl(f)
    reqs, skipped = [], 0
    for (seed_id, branch), turns in sorted(dialogues(rows).items()):
        msgs = to_messages(turns)
        for i, r in enumerate(turns):
            if r["copied"]:  # identical to the source clean dialogue's turn; analysis reuses it
                continue
            prompt = tok.apply_chat_template(msgs[: 2 * i + 1], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["text_norm"], add_special_tokens=False)["input_ids"]
            if not c_ids or len(p_ids) + len(c_ids) > c["generation"]["max_model_len"]:
                skipped += 1
                continue
            reqs.append({"key": (seed_id, branch, r["turn"]), "ids": p_ids + c_ids, "start": len(p_ids),
                         "n_bytes": len(r["text_norm"].encode())})
    return reqs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()
    c = load_config(profile=args.profile)
    spec = model_spec(c, args.model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(spec["path"])
    reqs, skipped = build_requests(c, tok, args.run)
    out_path = Path(args.run) / "d2" / f"{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(r["seed_id"], r["branch"], r["turn"]) for r in read_jsonl(out_path)}
    reqs = [q for q in reqs if q["key"] not in done]
    print(f"[d2:{args.model}] {len(reqs)} turns to score, {len(done)} done, {skipped} skipped", flush=True)

    # Gemma-3's prefix-LM attention needs TRITON_ATTN; FlashInfer needs nvcc (absent on nodes)
    kw = ({"limit_mm_per_prompt": {"image": 0}, "attention_backend": "TRITON_ATTN"} if spec["family"] == "gemma3"
          else {"attention_backend": "FLASH_ATTN"})
    llm = LLM(spec["path"], dtype="bfloat16", max_model_len=c["generation"]["max_model_len"],
              gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False, **kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
    t0 = time.time()
    n_prompt_tokens = 0
    with open(out_path, "a") as out:
        for k in range(0, len(reqs), args.chunk):
            chunk = reqs[k: k + args.chunk]
            res = llm.generate([{"prompt_token_ids": q["ids"]} for q in chunk], sp, use_tqdm=False)
            for q, o in zip(chunk, res):
                lps = [o.prompt_logprobs[i][q["ids"][i]].logprob for i in range(q["start"], len(q["ids"]))]
                n_prompt_tokens += len(q["ids"])
                s = float(sum(lps))
                out.write(json.dumps({"seed_id": q["key"][0], "branch": q["key"][1], "turn": q["key"][2],
                                      "n_tok": len(lps), "sum_lp": s, "mean_lp": s / len(lps),
                                      "n_bytes": q["n_bytes"]}) + "\n")
            out.flush()
            print(f"  {min(k + args.chunk, len(reqs))}/{len(reqs)} {time.time() - t0:.0f}s", flush=True)
    write_json(Path(args.run) / "throughput" / f"d2_{args.model}.json",
               {"turns": len(reqs), "skipped_too_long": skipped, "seconds": round(time.time() - t0, 1),
                "prompt_tokens": n_prompt_tokens})


if __name__ == "__main__":
    main()
