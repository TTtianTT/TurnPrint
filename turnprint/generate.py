"""Dialogue generation against two vLLM OpenAI-compatible servers.

phase=clean : model M answers all 8 turns for every (train+test) seed.
phase=branch: model M (=B) takes over test-seed dialogues of every other
              model A after turn tau. Turns 1..tau (and user turn tau+1) are
              copied from A's clean dialogue, whose history up to tau is
              identical, so the first post-handoff user message is a valid
              simulator draw for this branch and is shared across all B.

Histories are sent as role-tagged messages, so each server re-renders them
with its own chat template. Rows are appended per finished dialogue; rerunning
skips finished dialogues.
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

from .common import (ROOT, load_config, normalize, read_jsonl, relation,
                     select_seeds, stable_int, write_json, dialogues)


class Gen:
    def __init__(self, c, cand_url, sim_url, model, script=None):
        self.script = script  # replay mode: {seed_id: [(user_text, finish_reason) for turns 1..T]}
        self.c, self.g, self.T = c, c["generation"], c["turns"]
        self.cand_url, self.sim_url, self.model = cand_url, sim_url, model
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0),
                                        limits=httpx.Limits(max_connections=1024))
        self.sem = asyncio.Semaphore(self.g["concurrency"])
        self.stats = {"assistant_turns": 0, "assistant_tokens": 0, "user_turns": 0,
                      "user_tokens": 0, "empty_retries": 0}

    async def chat(self, url, served, messages, max_tokens, seed):
        g = self.g
        body = {"model": served, "messages": messages, "max_tokens": max_tokens,
                "temperature": g["temperature"], "top_p": g["top_p"], "top_k": g["top_k"],
                "min_p": g["min_p"], "repetition_penalty": g["repetition_penalty"],
                "seed": seed, "skip_special_tokens": True,
                "chat_template_kwargs": {"enable_thinking": g["enable_thinking"]}}
        for attempt in range(6):
            try:
                r = await self.client.post(url + "/v1/chat/completions", json=body)
                if r.status_code == 400:
                    raise RuntimeError(f"400 from {served}: {r.text[:500]}")
                r.raise_for_status()
                d = r.json()
                ch = d["choices"][0]
                return ch["message"]["content"] or "", ch["finish_reason"], d["usage"]["completion_tokens"]
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 5:
                    raise
                await asyncio.sleep(5 * (attempt + 1))

    async def user_turn(self, history, seed_id, branch, t):
        msgs = simulator_messages(self.c, history)
        for attempt in range(4):
            raw, fin, n = await self.chat(self.sim_url, "simulator", msgs, self.g["user_max_tokens"],
                                          stable_int(seed_id, branch, t, "user", attempt))
            text = clean_user(raw)
            self.stats["user_turns"] += 1
            self.stats["user_tokens"] += n
            if text:
                return text, fin
            self.stats["empty_retries"] += 1
        raise RuntimeError(f"Simulator produced empty output for {seed_id} {branch} t={t}")

    async def assistant_turn(self, history, user_text, seed_id, branch, t):
        msgs = [m for r in history for m in (
            {"role": "user", "content": r["user_text"]}, {"role": "assistant", "content": r["text_norm"]})]
        msgs.append({"role": "user", "content": user_text})
        for attempt in range(4):
            raw, fin, n = await self.chat(self.cand_url, self.model, msgs, self.g["max_tokens"],
                                          stable_int(seed_id, branch, t, "assistant", attempt))
            self.stats["assistant_turns"] += 1
            self.stats["assistant_tokens"] += n
            if normalize(raw):
                return raw, fin, n
            self.stats["empty_retries"] += 1
        return raw, fin, n

    async def dialogue(self, seed, branch, meta, prefix, first_user, out):
        async with self.sem:
            rows = list(prefix)
            for t in range(len(prefix) + 1, self.T + 1):
                if self.script is not None:
                    user, ufin = self.script[seed["seed_id"]][t - 1]
                elif first_user is not None and t == len(prefix) + 1:
                    user, ufin = first_user
                else:
                    user, ufin = await self.user_turn(rows, seed["seed_id"], branch, t)
                raw, fin, n = await self.assistant_turn(rows, user, seed["seed_id"], branch, t)
                rows.append({"seed_id": seed["seed_id"], "split": seed["split"], "branch": branch,
                             "turn": t, "author": self.model, **meta, "user_text": user,
                             "user_finish_reason": ufin, "text_raw": raw, "text_norm": normalize(raw),
                             "finish_reason": fin, "n_tokens": n, "copied": False})
            out.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            out.flush()


def simulator_messages(c, history):
    """The simulator reads the dialogue as a labeled transcript (more robust for an
    8B model than role-flipping, which made it slip into the assistant role)."""
    transcript = "\n\n".join(f"USER: {r['user_text']}\n\nAI: {r['text_norm']}" for r in history)
    return [{"role": "system", "content": c["simulator_system_prompt"]},
            {"role": "user", "content": f"Conversation so far:\n\n{transcript}\n\n---\nWrite the USER's next message."}]


def replay_messages(c, first, followups):
    """Replay mode: the simulator never sees assistant text, only the goal and its own
    earlier messages, so user turns carry no information about the assistant."""
    k = len(followups) + 2
    prev = "".join(f"\n({i + 2}) {u}" for i, u in enumerate(followups))
    body = (f"Your first message to the assistant was:\n\n{first}\n\n"
            + (f"Your follow-up messages so far:{prev}\n\n" if followups else "")
            + f"Write follow-up message number {k} of {c['turns']}.")
    return [{"role": "system", "content": c["replay_system_prompt"]}, {"role": "user", "content": body}]


async def make_scripts(c, seeds, sim_url, path):
    """One fixed user script per seed (turn 1 = seed request), shared by every model."""
    gen = Gen(c, None, sim_url, "simulator")
    done = {r["seed_id"] for r in read_jsonl(path)} if path.exists() else set()

    async def one(s, out):
        async with gen.sem:
            turns = [(s["user_text"], "seed")]
            for t in range(2, c["turns"] + 1):
                msgs = replay_messages(c, s["user_text"], [u for u, _ in turns[1:]])
                for attempt in range(4):
                    raw, fin, _ = await gen.chat(sim_url, "simulator", msgs, c["generation"]["user_max_tokens"],
                                                 stable_int(s["seed_id"], "script", t, attempt))
                    text = clean_user(re.sub(r"^\(?\d+\)\s*", "", raw.strip()))
                    if text:
                        break
                else:
                    raise RuntimeError(f"empty script turn {s['seed_id']} t={t}")
                turns.append((text, fin))
            out.write(json.dumps({"seed_id": s["seed_id"], "turns": turns}, ensure_ascii=False) + "\n")
            out.flush()

    with open(path, "a") as out:
        await asyncio.gather(*[one(s, out) for s in seeds if s["seed_id"] not in done])
    await gen.client.aclose()


LEAD = re.compile(r'^(?:\*\*)?(?:user|me|human|you)\s*:\s*(?:\*\*)?', re.I)


def clean_user(raw):
    t = normalize(raw)
    t = LEAD.sub("", t).strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t


def done_keys(path, T):
    if not path.exists():
        return set()
    by = dialogues(read_jsonl(path))
    return {k for k, v in by.items() if len(v) == T}


async def main_async(args):
    c = load_config(args.config, args.profile)
    run = Path(args.run)
    seeds = select_seeds(c, read_jsonl(args.seeds))
    write_json(run / "config.json", c)
    names = [m["name"] for m in c["models"]]
    if args.phase != "script" and args.model not in names:
        raise SystemExit(f"{args.model} not in profile models {names}")
    replay = c["profile"].get("user_mode") == "replay"
    script_path = run / "user_scripts.jsonl"
    if args.phase == "script":
        if not replay:
            raise SystemExit("script phase is only used with user_mode=replay")
        await make_scripts(c, seeds, args.sim_url, script_path)
        print(f"[script] {len(read_jsonl(script_path))} user scripts", flush=True)
        return
    script = None
    if replay:
        script = {r["seed_id"]: [tuple(u) for u in r["turns"]] for r in read_jsonl(script_path)}
    gen = Gen(c, args.cand_url, args.sim_url, args.model, script)
    out_path = run / "gen" / f"{args.phase}_{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(out_path, c["turns"])
    tasks, t0 = [], time.time()
    with open(out_path, "a") as out:
        if args.phase == "clean":
            meta = {"prefix_model": args.model, "tau": c["turns"], "relation": "none"}
            for s in seeds:
                if (s["seed_id"], args.model) not in finished:
                    tasks.append(gen.dialogue(s, args.model, meta, [], (s["user_text"], "seed"), out))
        else:
            B = args.model
            for A in names:
                if A == B:
                    continue
                clean = dialogues(read_jsonl(run / "gen" / f"clean_{A}.jsonl"))
                for s in seeds:
                    if s["split"] != "test":
                        continue
                    src = clean[(s["seed_id"], A)]
                    assert len(src) == c["turns"], (A, s["seed_id"])
                    for tau in c["profile"]["taus"]:
                        branch = f"{A}>{B}@{tau}"
                        if (s["seed_id"], branch) in finished:
                            continue
                        meta = {"prefix_model": A, "tau": tau, "relation": relation(c, A, B)}
                        prefix = [dict(r, branch=branch, copied=True, **meta) for r in src[:tau]]
                        nxt = src[tau]
                        tasks.append(gen.dialogue(s, branch, meta, prefix,
                                                  (nxt["user_text"], nxt["user_finish_reason"]), out))
        print(f"[{args.phase}:{args.model}] {len(tasks)} dialogues to generate "
              f"({len(finished)} already done)", flush=True)
        await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    stats = dict(gen.stats, dialogues=len(tasks), seconds=round(elapsed, 1),
                 assistant_tok_per_s=round(gen.stats["assistant_tokens"] / max(elapsed, 1e-9), 1))
    write_json(run / "throughput" / f"{args.phase}_{args.model}.json", stats)
    print(json.dumps(stats), flush=True)
    await gen.client.aclose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    ap.add_argument("--profile", default="full")
    ap.add_argument("--seeds", default=str(ROOT / "data/seeds.jsonl"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", choices=["script", "clean", "branch"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cand-url", default=None)
    ap.add_argument("--sim-url", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
