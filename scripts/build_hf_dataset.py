"""Package runs/ into a Hugging Face dataset folder for manual review.

Layout written to OUT:
  conversations/<run>.parquet  one row per dialogue (messages list), D2 scores merged per turn
  turns/<run>.parquet          one row per assistant turn (original gen records, flat)
  d2/<run>.parquet             per-turn D2 likelihoods, all scorers
  seeds/seeds.parquet          frozen WildChat seeds
  user_scripts/<run>.parquet   replayed user scripts (full_replay only)
  raw/                         byte-identical copy of data/ and runs/ (jsonl, analysis, logs, configs)
"""
import glob, os, shutil, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1]
RUNS = ["full_replay", "full", "smoke"]


def load_gen(run):
    dfs = []
    for f in sorted(glob.glob(f"{ROOT}/runs/{run}/gen/*.jsonl")):
        cond, model = os.path.basename(f)[:-6].split("_", 1)
        dfs.append(pd.read_json(f, lines=True, dtype=False).assign(condition=cond, file_model=model))
    return pd.concat(dfs, ignore_index=True)


def load_d2(run):
    dfs = [pd.read_json(f, lines=True).assign(scorer=os.path.basename(f)[:-6])
           for f in sorted(glob.glob(f"{ROOT}/runs/{run}/d2/*.jsonl"))]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def build_conversations(gen, d2):
    keys = ["seed_id", "branch", "turn"]
    if d2 is not None:
        wide = d2.pivot_table(index=keys, columns="scorer", values="mean_lp").reset_index()
        scorers = [c for c in wide.columns if c not in keys]
        gen = gen.merge(wide.rename(columns={s: "d2_" + s.replace("-", "_").replace(".", "_") for s in scorers}), on=keys, how="left")
    else:
        scorers = []
    rows = []
    for (cond, seed, branch), g in gen.sort_values("turn").groupby(["condition", "seed_id", "branch"], sort=True):
        msgs = []
        for r in g.itertuples(index=False):
            vals = {s: getattr(r, "d2_" + s.replace("-", "_").replace(".", "_")) for s in scorers}
            d2s = {s: None if pd.isna(v) else float(v) for s, v in vals.items()}
            msgs.append(dict(role="user", turn=r.turn, author="simulator", content=r.user_text,
                             finish_reason=r.user_finish_reason, copied=bool(r.copied),
                             n_tokens=None, content_norm=None, d2_mean_lp=None))
            msgs.append(dict(role="assistant", turn=r.turn, author=r.author, content=r.text_raw,
                             finish_reason=r.finish_reason, copied=bool(r.copied),
                             n_tokens=int(r.n_tokens), content_norm=r.text_norm,
                             d2_mean_lp=d2s or None))
        first = g.iloc[0]
        cont = first.file_model
        is_branch = cond == "branch"
        rows.append(dict(
            conv_id=f"{seed}|{branch}", seed_id=seed, split=first.split, condition=cond,
            branch=branch, model_A=first.prefix_model, model_B=cont if is_branch else None,
            tau=int(first.tau) if is_branch else None, relation=first.relation,
            n_turns=len(g), n_truncated=int((g.finish_reason == "length").sum()),
            authors=" ".join(g.author), messages=msgs))
    df = pd.DataFrame(rows)
    df["tau"] = df.tau.astype("Int64")
    return df


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; remove it first")
    for sub in ["conversations", "turns", "d2", "seeds", "user_scripts"]:
        os.makedirs(f"{OUT}/{sub}")
    pd.read_json(f"{ROOT}/data/seeds.jsonl", lines=True, dtype=False).to_parquet(f"{OUT}/seeds/seeds.parquet", index=False)
    for run in RUNS:
        gen, d2 = load_gen(run), load_d2(run)
        conv = build_conversations(gen, d2)
        conv.to_parquet(f"{OUT}/conversations/{run}.parquet", index=False, row_group_size=100)
        gen.drop(columns="file_model").to_parquet(f"{OUT}/turns/{run}.parquet", index=False)
        if d2 is not None:
            d2.to_parquet(f"{OUT}/d2/{run}.parquet", index=False)
        us = f"{ROOT}/runs/{run}/user_scripts.jsonl"
        if os.path.exists(us):
            s = pd.read_json(us, lines=True, dtype=False)
            s["turns"] = s.turns.map(lambda ts: [dict(text=t, finish_reason=fr) for t, fr in ts])
            s.to_parquet(f"{OUT}/user_scripts/{run}.parquet", index=False)
        print(run, "convs", len(conv), "turns", len(gen), "d2", 0 if d2 is None else len(d2))
    shutil.copytree(f"{ROOT}/runs", f"{OUT}/raw/runs")
    shutil.copytree(f"{ROOT}/data", f"{OUT}/raw/data")
    shutil.copy(f"{ROOT}/configs/pilot.json", f"{OUT}/raw/pilot.json")


if __name__ == "__main__":
    main()
