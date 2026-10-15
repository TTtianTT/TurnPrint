"""CPU end-to-end check of the analysis on a synthetic run.

Each model writes text with a planted signature word, and D2 likelihoods
favour the true author, so D1/D2 + D3 must localize every switch exactly and
the user-only shortcut check must sit near chance (user text carries no
author signal). Run: python -m unittest tests.test_pipeline
"""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from turnprint.analyze import d3
from turnprint.common import ROOT, load_config, relation

WORDS = "alpha beta gamma delta river stone cloud light paper green quick slow table chair music".split()


def text(rng, model):
    body = " ".join(rng.choice(WORDS) for _ in range(40))
    return f"{body} sig{model.replace('-', '').replace('.', '')} {body[:30]}."


def synth(run, c, n_train=10, n_test=6):
    rng = random.Random(0)
    models = [m["name"] for m in c["models"]]
    T = c["turns"]
    rows = {m: [] for m in models}
    seeds = [(f"s{i:03d}", "train" if i < n_train else "test") for i in range(n_train + n_test)]
    clean = {}
    for m in models:
        for sid, split in seeds:
            d = []
            for t in range(1, T + 1):
                d.append({"seed_id": sid, "split": split, "branch": m, "turn": t, "author": m, "prefix_model": m,
                          "tau": T, "relation": "none", "user_text": " ".join(rng.choice(WORDS) for _ in range(12)),
                          "user_finish_reason": "stop", "text_raw": "", "text_norm": text(rng, m),
                          "finish_reason": "stop", "n_tokens": 50, "copied": False})
            clean[(sid, m)] = d
            rows[m] += d
    branch = {m: [] for m in models}
    for B in models:
        for A in models:
            if A == B:
                continue
            for sid, split in seeds:
                if split != "test":
                    continue
                for tau in c["profile"]["taus"]:
                    br = f"{A}>{B}@{tau}"
                    meta = {"prefix_model": A, "tau": tau, "relation": relation(c, A, B)}
                    for r in clean[(sid, A)][:tau]:
                        branch[B].append(dict(r, branch=br, copied=True, **meta))
                    for t in range(tau + 1, T + 1):
                        branch[B].append({**clean[(sid, B)][t - 1], "branch": br, **meta,
                                          "text_norm": text(rng, B), "copied": False})
    (run / "gen").mkdir(parents=True)
    (run / "d2").mkdir()
    for m in models:
        for name, rs in [(f"clean_{m}", rows[m]), (f"branch_{m}", branch[m])]:
            (run / "gen" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    allrows = [r for m in models for r in rows[m] + branch[m] if not r["copied"]]
    for m in models:  # scorer m prefers its own text
        out = [{"seed_id": r["seed_id"], "branch": r["branch"], "turn": r["turn"], "n_tok": 50,
                "mean_lp": (-1.0 if r["author"] == m else -2.0) + rng.gauss(0, 0.1),
                "sum_lp": (-50.0 if r["author"] == m else -100.0), "n_bytes": 300} for r in allrows]
        (run / "d2" / f"{m}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))


class TestD3(unittest.TestCase):
    def test_exact_switch(self):
        S = np.full((8, 4), -3.0)
        S[:3, 1] = 0
        S[3:, 2] = 0
        tau, a, b, det = d3(S)
        self.assertEqual((tau, a, b), (3, 1, 2))
        self.assertGreater(det, 0)

    def test_no_switch_scores_lower(self):
        S = np.full((8, 4), -3.0)
        S[:, 0] = 0
        self.assertLess(d3(S)[3], 0)


class TestPipeline(unittest.TestCase):
    def test_synthetic_run(self):
        c = load_config(profile="smoke")
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            synth(run, c)
            # profile smoke has 2+2 seeds; analysis must not depend on profile seed counts
            subprocess.run([sys.executable, "-m", "turnprint.analyze", "--run", str(run), "--profile", "smoke"],
                           check=True, cwd=ROOT)
            m = json.loads((run / "analysis/metrics.json").read_text())
            self.assertGreater(m["d1_clean_acc"]["mean"], 0.95)
            for k in ["D1/all", "D2/all", "D2cal/all"]:
                self.assertEqual(m["A3"][k]["exact"]["mean"], 1.0, k)
                self.assertEqual(m["A3"][k]["joint"]["mean"], 1.0, k)
                self.assertGreater(m["A3"][k]["auroc"]["mean"], 0.95, k)
            self.assertLess(m["A4_user_only_acc"]["mean"], 0.45)
            self.assertTrue((run / "analysis/fig1_contamination.png").exists())
            self.assertTrue((run / "analysis/fig2_transient.png").exists())


if __name__ == "__main__":
    unittest.main()
