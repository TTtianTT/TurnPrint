"""D1 / D2 / D3 detectors and analyses A1-A4 (CPU only).

All confidence intervals are 95% percentile intervals from a seed-cluster
bootstrap: test seeds are resampled with replacement and everything that came
from a sampled seed (all pairs, taus and turns) moves together.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .common import ROOT, load_config, read_jsonl, write_json

PUNCT = list(".,;:!?-—–()\"'*#`/") + ["\n", "**", "...", "…"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


# ---------------------------------------------------------------- D1 features
def style_features(texts):
    out = []
    for t in texts:
        n = max(len(t), 1)
        words = re.findall(r"[A-Za-z']+", t)
        wl = np.array([len(w) for w in words]) if words else np.array([0])
        hist = np.bincount(np.clip(wl, 0, 12), minlength=13)[1:] / max(len(words), 1)
        lines = t.split("\n")
        f = [np.log1p(len(t)), np.log1p(len(words)), np.log1p(len(lines)), wl.mean(),
             sum(ch.isupper() for ch in t) / n, sum(ch.isdigit() for ch in t) / n,
             len(EMOJI.findall(t)) / n,
             sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", l)) for l in lines) / len(lines),
             sum(l.lstrip().startswith("#") for l in lines) / len(lines),
             np.mean([len(s.split()) for s in re.split(r"[.!?]+\s", t) if s.strip()] or [0])]
        f += [t.count(p) / n for p in PUNCT]
        out.append(np.concatenate([f, hist]))
    return np.array(out)


class D1:
    """TF-IDF word/char n-grams + stylometric features -> multinomial LR."""

    def __init__(self, C_grid, rng):
        self.C_grid, self.rng = C_grid, rng

    def _fit_vec(self, texts):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000,
                                    sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, max_features=50000,
                                    sublinear_tf=True, lowercase=False)
        self.scaler = StandardScaler()
        self.word.fit(texts), self.char.fit(texts), self.scaler.fit(style_features(texts))

    def X(self, texts):
        s = np.clip(self.scaler.transform(style_features(texts)), -5, 5)
        return sparse.hstack([self.word.transform(texts), self.char.transform(texts),
                              sparse.csr_matrix(s)]).tocsr()

    def fit(self, texts, y, groups):
        self._fit_vec(texts)
        X = self.X(texts)
        y = np.asarray(y)
        cv = {}
        for C in self.C_grid:  # C chosen by seed-grouped CV on TRAIN seeds only
            accs = []
            for tr, va in GroupKFold(min(5, len(set(groups)))).split(X, y, groups):
                m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
                accs.append((m.predict(X[va]) == y[va]).mean())
            cv[C] = float(np.mean(accs))
        self.C = max(cv, key=cv.get)
        self.cv = cv
        self.clf = LogisticRegression(C=self.C, max_iter=3000).fit(X, y)
        self.classes = list(self.clf.classes_)
        return self

    def log_proba(self, texts):
        return self.clf.predict_log_proba(self.X(texts))

    def top_features(self, k=25):
        names = np.concatenate([self.word.get_feature_names_out(), ["c:" + x for x in self.char.get_feature_names_out()],
                                ["style:" + str(i) for i in range(self.scaler.n_features_in_)]])
        return {c: [names[j] for j in np.argsort(-self.clf.coef_[i])[:k]] for i, c in enumerate(self.classes)}


# ---------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, seeds, B, rng):
        self.seeds = sorted(seeds)
        self.idx = {s: i for i, s in enumerate(self.seeds)}
        self.W = rng.multinomial(len(self.seeds), np.ones(len(self.seeds)) / len(self.seeds), size=B)

    def mean(self, df, col):
        """Mean of df[col] (rows pooled) with seed-cluster CI."""
        df = df.dropna(subset=[col])
        if df.empty:
            return {"mean": None, "lo": None, "hi": None, "n": 0}
        g = df.groupby("seed_id")[col].agg(["sum", "count"])
        s = np.zeros(len(self.seeds)); n = np.zeros(len(self.seeds))
        ix = [self.idx[k] for k in g.index]
        s[ix], n[ix] = g["sum"].values, g["count"].values
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (self.W @ s) / (self.W @ n)
        bs = bs[np.isfinite(bs)]
        return {"mean": float(s.sum() / n.sum()), "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5)), "n": int(n.sum())}

    def stat(self, df, fn):
        """Arbitrary statistic fn(df) with seed-cluster CI (slower)."""
        by = {s: d for s, d in df.groupby("seed_id")}
        point = fn(df)
        vals = []
        for w in self.W[:1000]:
            parts = [by[s] for s, k in zip(self.seeds, w) for _ in range(k) if s in by]
            try:
                vals.append(fn(pd.concat(parts)))
            except ValueError:
                pass
        return {"mean": float(point), "lo": float(np.percentile(vals, 2.5)),
                "hi": float(np.percentile(vals, 97.5)), "n": len(df)}


# ---------------------------------------------------------------- D3
def d3(S):
    """S: T x M evidence. Returns (tau_hat, A_hat, B_hat, detection score).
    Best two-segment split with A != B (tau in 1..T-1) minus best one-segment."""
    T, M = S.shape
    C = np.vstack([np.zeros(M), np.cumsum(S, 0)])
    one = C[T].max()
    best = (-np.inf, None, None, None)
    for tau in range(1, T):
        two = C[tau][:, None] + (C[T] - C[tau])[None, :]
        np.fill_diagonal(two, -np.inf)
        a, b = np.unravel_index(np.argmax(two), two.shape)
        if two[a, b] > best[0]:
            best = (two[a, b], tau, a, b)
    return best[1], best[2], best[3], best[0] - one


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--exclude-truncated", action="store_true")
    ap.add_argument("--strip-boilerplate", action="store_true")
    ap.add_argument("--no-d2", action="store_true", help="D1-only preview (e.g. while D2 scoring runs)")
    args = ap.parse_args()
    c = load_config(profile=args.profile)
    run = Path(args.run)
    sub = "analysis" + ("_d1only" if args.no_d2 else "") + ("_excl_trunc" if args.exclude_truncated else "") + ("_strip" if args.strip_boilerplate else "")
    out = run / sub
    out.mkdir(parents=True, exist_ok=True)
    A_cfg = c["analysis"]
    rng = np.random.default_rng(A_cfg["rng_seed"])
    models = [m["name"] for m in c["models"]]
    M, T = len(models), c["turns"]
    mi = {m: i for i, m in enumerate(models)}

    rows = []
    for f in sorted((run / "gen").glob("*.jsonl")):
        rows += read_jsonl(f)
    df = pd.DataFrame(rows)
    df = df[df.author.isin(models) & df.prefix_model.isin(models)].reset_index(drop=True)
    df["text"] = df.text_norm
    if args.strip_boilerplate:
        for pat in A_cfg["boilerplate_patterns"]:
            df["text"] = df.text.str.replace(pat, "", regex=True, flags=re.M).str.strip()
    df["trunc"] = df.finish_reason.eq("length")
    clean = df[df.relation == "none"]
    train = clean[clean.split == "train"].reset_index(drop=True)
    test_clean = clean[clean.split == "test"]
    test_seeds = sorted(df[df.split == "test"].seed_id.unique())
    boot = Boot(test_seeds, A_cfg["bootstrap"], rng)
    metrics = {"profile": c["profile"], "models": models, "n_rows": len(df)}

    # ---- data audit: truncation and boilerplate
    metrics["truncation_rate"] = {m: float(df[(df.author == m) & ~df.copied].trunc.mean()) for m in models}
    metrics["mean_tokens"] = {m: float(df[(df.author == m) & ~df.copied].n_tokens.mean()) for m in models}
    metrics["think_tag_rows"] = int(df.text_raw.str.contains("<think>|</think>").sum())
    bp = {}
    for m in models:
        tx = train[train.author == m].text
        first = Counter(t.split("\n")[0][:60] for t in tx).most_common(5)
        last = Counter(t.strip().split("\n")[-1][:60] for t in tx).most_common(5)
        bp[m] = {"first_lines": first, "last_lines": last}
    write_json(out / "boilerplate_audit.json", bp)

    # ---- D1
    d1 = D1(A_cfg["d1_C_grid"], rng).fit(list(train.text), list(train.author), list(train.seed_id))
    lp = d1.log_proba(list(df.text))
    order = [d1.classes.index(m) for m in models]
    lp = lp[:, order]
    for m in models:
        df["d1_" + m] = np.exp(lp[:, mi[m]])
    metrics["d1"] = {"C": d1.C, "cv_acc_train": d1.cv}
    write_json(out / "d1_top_features.json", d1.top_features())
    df["d1_pred"] = np.array(models)[lp.argmax(1)]
    df["d1_correct"] = (df.d1_pred == df.author).astype(float)
    tc = df[(df.relation == "none") & (df.split == "test")]
    metrics["d1_clean_acc"] = boot.mean(tc, "d1_correct")
    metrics["d1_clean_acc_by_turn"] = {int(t): float(g.d1_correct.mean()) for t, g in tc.groupby("turn")}
    metrics["d1_clean_confusion"] = pd.crosstab(tc.author, tc.d1_pred).to_dict()
    metrics["chance"] = 1 / M
    df[["seed_id", "split", "branch", "turn", "author", "prefix_model", "tau", "relation", "copied", "trunc"]
       + ["d1_" + m for m in models]].to_csv(out / "d1_turn_probs.csv", index=False)

    # ---- D2 (likelihood) evidence
    d2 = {}
    for m in models:
        p = run / "d2" / f"{m}.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                d2[(m, r["seed_id"], r["branch"], r["turn"])] = r
    have_d2 = len(d2) > 0 and not args.no_d2

    def d2_lookup(m, r, field):
        # copied prefix turns were scored once, on the source clean dialogue
        br = r.prefix_model if r.copied else r.branch
        v = d2.get((m, r.seed_id, br, r.turn))
        return np.nan if v is None else v[field]

    if have_d2:
        for m in models:
            df["d2mean_" + m] = [d2_lookup(m, r, "mean_lp") for r in df.itertuples()]
            df["d2sum_" + m] = [d2_lookup(m, r, "sum_lp") for r in df.itertuples()]
        df["n_bytes"] = df.text_norm.str.encode("utf-8").str.len()
        # D2-cal: tiny multinomial LR over the M likelihood features, fit on train-clean turns
        feats = ["d2mean_" + m for m in models] + ["d2sum_" + m for m in models]
        X = df[feats].copy()
        for m in models:
            X["d2sum_" + m] = X["d2sum_" + m] / df.n_bytes.clip(lower=1)
        ok = X.notna().all(axis=1)
        trmask = ok & (df.relation == "none") & (df.split == "train")
        sc = StandardScaler().fit(X[trmask])
        cal = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[trmask]), df.author[trmask])
        cl = np.full((len(df), M), np.nan)
        cl[ok.values] = cal.predict_log_proba(sc.transform(X[ok]))[:, [list(cal.classes_).index(m) for m in models]]
        for m in models:
            df["d2cal_" + m] = cl[:, mi[m]]
        tc = df[(df.relation == "none") & (df.split == "test")]
        for name in ["d2mean", "d2sum", "d2cal"]:
            sub_ = tc.dropna(subset=[f"{name}_{m}" for m in models])
            corr = (np.array(models)[sub_[[f"{name}_{m}" for m in models]].values.argmax(1)] == sub_.author).astype(float)
            metrics[f"{name}_clean_acc"] = boot.mean(sub_.assign(c=corr.values), "c")
        metrics["d2_missing_turns"] = int(df["d2mean_" + models[0]].isna().sum())

    # ---- index helpers
    key = df.set_index(["seed_id", "branch", "turn"])
    test_br = df[(df.split == "test") & (df.relation != "none")]

    # ---- A1: contamination index CI(k), k = tau
    a1 = []
    for (s, br), g in test_br.groupby(["seed_id", "branch"]):
        g = g.set_index("turn")
        A, B, tau = g.prefix_model.iloc[0], g.author.iloc[-1], int(g.tau.iloc[0])
        x, ref = g.loc[tau + 1], key.loc[(s, B, tau + 1)]
        if args.exclude_truncated and (x.trunc or ref.trunc):
            continue
        rec = {"seed_id": s, "A": A, "B": B, "tau": tau, "relation": g.relation.iloc[0],
               "pA_branch": x["d1_" + A], "pA_clean": ref["d1_" + A], "ci": x["d1_" + A] - ref["d1_" + A]}
        if have_d2:  # exploratory likelihood analogue: shift of (l_A - l_B) in nats/token
            rec["ci_d2"] = (x["d2mean_" + A] - x["d2mean_" + B]) - (ref["d2mean_" + A] - ref["d2mean_" + B])
        a1.append(rec)
    a1 = pd.DataFrame(a1)
    a1.to_csv(out / "a1_pairs.csv", index=False)
    metrics["A1"] = {rel: {"pooled": boot.mean(a1[a1.relation == rel], "ci"),
                           **{f"k={t}": boot.mean(a1[(a1.relation == rel) & (a1.tau == t)], "ci")
                              for t in c["profile"]["taus"]}} for rel in ["same", "cross"]}
    metrics["A1_by_pair"] = {f"{a}>{b}": boot.mean(g, "ci") for (a, b), g in a1.groupby(["A", "B"])}
    if have_d2:
        metrics["A1_D2"] = {rel: {"pooled": boot.mean(a1[a1.relation == rel], "ci_d2"),
                                  **{f"k={t}": boot.mean(a1[(a1.relation == rel) & (a1.tau == t)], "ci_d2")
                                     for t in c["profile"]["taus"]}} for rel in ["same", "cross"]}

    # ---- A2: transient P(B) after handoff vs B's clean dialogue at the same absolute turn
    a2 = []
    for (s, br), g in test_br.groupby(["seed_id", "branch"]):
        g = g.set_index("turn")
        B, tau = g.author.iloc[-1], int(g.tau.iloc[0])
        for t in range(tau + 1, T + 1):
            x, ref = g.loc[t], key.loc[(s, B, t)]
            if args.exclude_truncated and (x.trunc or ref.trunc):
                continue
            rec = {"seed_id": s, "B": B, "tau": tau, "j": t - tau, "relation": g.relation.iloc[0],
                   "diff": x["d1_" + B] - ref["d1_" + B]}
            if have_d2:  # margin of l_B over the best other candidate, branch minus clean
                marg = lambda r: r["d2mean_" + B] - max(r["d2mean_" + m] for m in models if m != B)
                rec["diff_d2"] = marg(x) - marg(ref)
            a2.append(rec)
    a2 = pd.DataFrame(a2)
    a2.to_csv(out / "a2_turns.csv", index=False)
    metrics["A2"] = {}
    for rel in ["same", "cross", "all"]:
        d = a2 if rel == "all" else a2[a2.relation == rel]
        for name, col in [("A2", "diff"), ("A2_D2", "diff_d2")]:
            if col in d:
                metrics.setdefault(name, {})[rel] = {
                    f"tau={t}": {f"j={j}": boot.mean(d[(d.tau == t) & (d.j == j)], col)
                                 for j in range(1, T - t + 1)} for t in c["profile"]["taus"]}

    # ---- A3: D3 localization and detection
    evid = {"D1": "d1"}
    if have_d2:
        evid.update({"D2": "d2mean", "D2sum": "d2sum", "D2cal": "d2cal"})
    test_all = df[df.split == "test"]
    seg = []
    for (s, br), g in test_all.groupby(["seed_id", "branch"]):
        g = g.sort_values("turn")
        if len(g) != T:
            continue
        if args.exclude_truncated and g.trunc.any():
            continue
        rel = g.relation.iloc[0]
        A, B, tau = g.prefix_model.iloc[0], g.author.iloc[-1], int(g.tau.iloc[0])
        rec = {"seed_id": s, "branch": br, "relation": rel, "A": A, "B": B, "tau": tau}
        for ename, col in evid.items():
            cols = [f"{col}_{m}" for m in models]
            S = g[cols].values.astype(float)
            if ename == "D1":
                S = np.log(np.clip(S, 1e-12, 1))
            if np.isnan(S).any():
                continue
            th, ah, bh, det = d3(S)
            rec.update({f"{ename}_tau": th, f"{ename}_A": models[ah], f"{ename}_B": models[bh], f"{ename}_det": det})
            if rel != "none":
                rec[f"{ename}_exact"] = float(th == tau)
                rec[f"{ename}_pm1"] = float(abs(th - tau) <= 1)
                rec[f"{ename}_pair"] = float(models[ah] == A and models[bh] == B)
                rec[f"{ename}_joint"] = float(th == tau and models[ah] == A and models[bh] == B)
        seg.append(rec)
    seg = pd.DataFrame(seg)
    seg.to_csv(out / "segments.csv", index=False)
    table = {}
    for ename in evid:
        if f"{ename}_det" not in seg:
            continue
        for rel in ["same", "cross", "all"]:
            sw = seg[(seg.relation != "none") if rel == "all" else (seg.relation == rel)].dropna(subset=[f"{ename}_det"])
            cl = seg[seg.relation == "none"].dropna(subset=[f"{ename}_det"])
            both = pd.concat([sw.assign(y=1), cl.assign(y=0)])
            table[f"{ename}/{rel}"] = {
                "exact": boot.mean(sw, f"{ename}_exact"), "pm1": boot.mean(sw, f"{ename}_pm1"),
                "pair": boot.mean(sw, f"{ename}_pair"), "joint": boot.mean(sw, f"{ename}_joint"),
                "auroc": boot.stat(both, lambda d: roc_auc_score(d.y, d[f"{ename}_det"])),
                "exact_by_tau": {int(t): float(gg[f"{ename}_exact"].mean()) for t, gg in sw.groupby("tau")}}
    metrics["A3"] = table
    metrics["A3_random"] = {"exact": 1 / (T - 1), "pm1": 3 / (T - 1)}
    if have_d2 and "D2_exact" in seg:
        sw = seg[seg.relation != "none"].dropna(subset=["D1_exact", "D2_exact"])
        metrics["A3_D2_minus_D1_exact"] = {rel: boot.mean(sw[sw.relation == rel].assign(d=lambda d: d.D2_exact - d.D1_exact), "d")
                                           for rel in ["same", "cross"]}

    # ---- A4: shortcut check, user turns only -> which assistant they respond to
    def user_set(split):
        d = clean[(clean.split == split) & (clean.turn >= 2)]
        return list(d.user_text), list(d.author), list(d.seed_id)
    ux, uy, ug = user_set("train")
    u = D1(A_cfg["d1_C_grid"], rng).fit(ux, uy, ug)
    tx, ty, tg = user_set("test")
    ucorr = (np.array(u.classes)[u.log_proba(tx).argmax(1)] == np.array(ty)).astype(float)
    metrics["A4_user_only_acc"] = boot.mean(pd.DataFrame({"seed_id": tg, "c": ucorr}), "c")
    write_json(out / "a4_user_top_features.json", u.top_features())

    # ---- gates
    thr = c["gates"]["user_only_max_accuracy"] if M == 4 else c["gates"]["user_only_max_accuracy_3models"]
    gates = {"user_only_shortcut_ok": metrics["A4_user_only_acc"]["mean"] <= thr, "user_only_threshold": thr,
             "clean_d1_above_chance": metrics["d1_clean_acc"]["lo"] > 1 / M,
             "A1_cross_pooled_ci_excludes_0": (metrics["A1"]["cross"]["pooled"]["lo"] or 0) > 0
                                               or (metrics["A1"]["cross"]["pooled"]["hi"] or 0) < 0,
             "A1_same_pooled_ci_excludes_0": (metrics["A1"]["same"]["pooled"]["lo"] or 0) > 0
                                              or (metrics["A1"]["same"]["pooled"]["hi"] or 0) < 0}
    target = c["gates"]["localization_exact_min_factor"] / (T - 1)
    for k, v in table.items():
        gates[f"loc_exact>=2x_random[{k}]"] = v["exact"]["mean"] >= target
    metrics["gates"] = gates
    write_json(out / "metrics.json", metrics)
    figures(metrics, c, out)
    report(metrics, c, out)
    print(json.dumps(gates, indent=1))


# ---------------------------------------------------------------- figures / report
COL = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax):
    ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    ax.axhline(0, color="#8a897f", lw=1)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#8a897f")
    ax.tick_params(colors="#55544c")


def figures(metrics, c, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    taus = c["profile"]["taus"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
    for i, (rel, lab) in enumerate([("same", "same family"), ("cross", "cross family")]):
        pts = [metrics["A1"][rel][f"k={t}"] for t in taus]
        x = np.array(taus) + (i - 0.5) * 0.08
        y = np.array([p["mean"] for p in pts])
        err = np.array([[p["mean"] - p["lo"], p["hi"] - p["mean"]] for p in pts]).T
        ax.errorbar(x, y, yerr=err, color=COL[i], lw=2, marker="o", ms=6, capsize=3, label=lab)
    _style(ax)
    ax.set_xticks(taus)
    ax.set_xlabel("handoff after turn k (= τ)")
    ax.set_ylabel("CI(k) = ΔP_D1(A) on B's first turn")
    ax.set_title("Fig 1. Contamination index (95% seed-bootstrap CI)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_contamination.png")
    plt.close(fig)

    if "omit_a2" in c["profile"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=160, sharey=True)
    for ax, rel in zip(axes, ["same", "cross", "all"]):
        for i, t in enumerate(taus):
            d = metrics["A2"][rel][f"tau={t}"]
            js = [int(k[2:]) for k in d]
            y = np.array([d[k]["mean"] for k in d], float)
            lo = np.array([d[k]["lo"] for k in d], float)
            hi = np.array([d[k]["hi"] for k in d], float)
            ax.plot(js, y, color=COL[i], lw=2, marker="o", ms=4, label=f"τ={t}")
            ax.fill_between(js, lo, hi, color=COL[i], alpha=0.15, lw=0)
        _style(ax)
        ax.set_title({"same": "same family", "cross": "cross family", "all": "all pairs"}[rel], fontsize=9)
        ax.set_xlabel("j (turns since handoff)")
    axes[0].set_ylabel("P_D1(B) branch − clean")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 2. Handoff transient (95% seed-bootstrap CI)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_transient.png")
    plt.close(fig)


def _f(p):
    return "n/a" if p is None or p.get("mean") is None else f"{p['mean']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]"


def report(m, c, out):
    L = [f"# TurnPrint early pilot — {m['profile']['name']}", "",
         f"Models: {', '.join(m['models'])}. Chance = {m['chance']:.3f}. 95% seed-cluster bootstrap CIs.", "",
         "## Preconditions",
         f"- A4 user-only author accuracy: {_f(m['A4_user_only_acc'])} (gate ≤ {m['gates']['user_only_threshold']})",
         f"- D1 clean per-turn accuracy: {_f(m['d1_clean_acc'])} (C={m['d1']['C']})",
         f"- D1 clean accuracy by turn: " + ", ".join(f"t{k}={v:.2f}" for k, v in m['d1_clean_acc_by_turn'].items())]
    for k in ["d2mean", "d2sum", "d2cal"]:
        if f"{k}_clean_acc" in m:
            L.append(f"- {k} clean per-turn accuracy (argmax): {_f(m[f'{k}_clean_acc'])}")
    L += ["", "## Data audit",
          "- truncation rate: " + ", ".join(f"{k}={v:.3f}" for k, v in m["truncation_rate"].items()),
          "- mean tokens/turn: " + ", ".join(f"{k}={v:.0f}" for k, v in m["mean_tokens"].items()),
          f"- rows with think tags in raw output: {m['think_tag_rows']}", "",
          "## A1 contamination index CI(k)", "| relation | pooled | " + " | ".join(f"k={t}" for t in m["profile"]["taus"]) + " |",
          "|---|---|" + "---|" * len(m["profile"]["taus"])]
    for rel in ["same", "cross"]:
        L.append(f"| {rel} | {_f(m['A1'][rel]['pooled'])} | " + " | ".join(_f(m['A1'][rel][f'k={t}']) for t in m["profile"]["taus"]) + " |")
    if "A1_D2" in m:
        L += ["", "Exploratory D2 analogue: shift of (ℓ_A − ℓ_B) on B's first turn, nats/token", "| relation | pooled | "
              + " | ".join(f"k={t}" for t in m["profile"]["taus"]) + " |", "|---|---|" + "---|" * len(m["profile"]["taus"])]
        for rel in ["same", "cross"]:
            L.append(f"| {rel} | {_f(m['A1_D2'][rel]['pooled'])} | " + " | ".join(_f(m['A1_D2'][rel][f'k={t}']) for t in m["profile"]["taus"]) + " |")
    for name, lab in [("A2", "P_D1(B) branch − clean"), ("A2_D2", "D2 margin of B, branch − clean (nats/token)")]:
        if name in m:
            L += ["", f"## {name} transient: {lab} (all pairs)"]
            for t in m["profile"]["taus"]:
                L.append(f"- τ={t}: " + ", ".join(f"{k}: {_f(v)}" for k, v in m[name]["all"][f"tau={t}"].items()))
    L += ["", "## A3 localization / detection (Table 1)",
          f"Random: exact {m['A3_random']['exact']:.3f}, ±1 {m['A3_random']['pm1']:.3f}", "",
          "| evidence/relation | τ exact | τ ±1 | (A,B) correct | joint | detection AUROC |", "|---|---|---|---|---|---|"]
    for k, v in m["A3"].items():
        L.append(f"| {k} | {_f(v['exact'])} | {_f(v['pm1'])} | {_f(v['pair'])} | {_f(v['joint'])} | {_f(v['auroc'])} |")
    if "A3_D2_minus_D1_exact" in m:
        L += ["", "D2 − D1 exact-hit (paired): " + ", ".join(f"{r}: {_f(v)}" for r, v in m["A3_D2_minus_D1_exact"].items())]
    L += ["", "## Gates", *[f"- {k}: {v}" for k, v in m["gates"].items()], ""]
    (out / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
