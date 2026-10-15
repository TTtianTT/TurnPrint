# TurnPrint — early pilot (1–2 days)

Question: when model B takes over model A's conversation history, does the
handoff leave a signal strong enough to justify the one-month pilot?
Four candidates (Qwen3-8B/32B, Gemma-3-12B/27B-IT), user simulator
Llama-3.1-8B-Instruct, all local BF16, no system prompts, thinking disabled.

## Layout

```
configs/pilot.json          models, profiles (full / compressed / smoke), sampling, gates
configs/seed_exclusions.txt WildChat conversation hashes rejected in manual review
data/seeds.jsonl            frozen 200 seeds (150 train / 50 test)
scripts/prepare_seeds.py    WildChat -> seeds (filters + deterministic sample)
scripts/submit.sh           chain generation -> D2 scoring -> analysis (afterok); <= 2 GPUs total
scripts/sim_check.py        regenerate simulator follow-ups and flag role slips (1 GPU)
slurm/generate.sbatch       2 GPUs: simulator vLLM server on GPU1, candidate servers on GPU0
slurm/score.sbatch          2 GPUs: D2 scoring as two parallel 1-GPU srun steps
slurm/analyze.sbatch        CPU only (4RTXq): analysis, with and without truncated turns
turnprint/generate.py       async dialogue driver (clean + branch phases)
turnprint/score.py          D2 per-turn likelihoods (vLLM prompt_logprobs)
turnprint/analyze.py        D1, D3, A1–A4, figures, report
tests/test_pipeline.py      synthetic end-to-end check of the analysis (CPU)
```

Environments: generation/scoring use `/dataset1/zailong/envs/peft-sft-lab`
(vLLM 0.25.1); analysis uses `/dataset1/zailong/envs/turnprint` (sklearn etc.).
Slurm needs `SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf` on this host
(`submit.sh` sets it).

## Running

```bash
scripts/submit.sh smoke smoke all                  # 2+2 seeds, mechanical check
scripts/submit.sh full  full  all                  # reactive simulator (fails the A4 shortcut gate)
scripts/submit.sh full_replay full_replay all      # replayed user turns (primary)
/dataset1/zailong/envs/turnprint/bin/python -m unittest tests.test_pipeline
```

Outputs in `runs/<name>/`: `gen/{clean,branch}_<model>.jsonl` (one row per
assistant turn, format from the plan plus `user_finish_reason` and `copied`),
`d2/<model>.jsonl`, `throughput/`, `server_logs/`, and `analysis/`
(`report.md`, `metrics.json`, `fig1_contamination.png`, `fig2_transient.png`,
`segments.csv`, `a1_pairs.csv`, `a2_turns.csv`, D1 per-turn probabilities,
top features, boilerplate audit). `analysis_excl_trunc/` repeats everything
without truncated turns.

## Design decisions not fixed by the plan

- **Deviation: assistant cap 2048 tokens, not 512.** In the smoke test 100%
  of Qwen3-8B and 94% of Qwen3-32B turns hit 512 tokens (Qwen3 without
  thinking writes long markdown answers). At 512, "ends mid-sentence" becomes
  a trivial author cue, simulator follow-ups drift to "you got cut off", and
  the plan's remedy (drop truncated turns) would drop nearly all Qwen data.
  Context limit raised to 32k accordingly. Truncated turns stay flagged and
  `analysis_excl_trunc/` repeats every result without them.
- **Attention backends.** FLASH_ATTN for Qwen/Llama, TRITON_ATTN for Gemma-3
  (its prefix-LM attention is unsupported by FLASH_ATTN); FlashInfer is
  avoided because it JIT-compiles with nvcc, which compute nodes lack.
- **Generation.** Sampling T=0.7, top-p 1, top-k off, 2048 assistant tokens,
  256 user tokens; per-request seeds derived from (seed, branch, turn, role).
  Histories are passed as role messages, so every model re-renders with its
  own template. Empty outputs are re-sampled (counted in throughput stats).
- **Branches.** Turns 1..τ and user turn τ+1 are copied from A's clean
  dialogue (the history up to τ is identical, so that user turn is a valid
  simulator draw for the branch; all B then face the same first prompt).
  In replay mode all user turns come from the shared script.
- **Simulator, reactive mode (`full`).** Llama reads the dialogue as a
  labeled USER/AI transcript (role-flipping made it slip into the assistant
  role) and writes a short, plain follow-up. This run FAILED the A4 shortcut
  gate (user-only accuracy 0.36 > 0.30): follow-ups track each model's
  topical drift and answer its clarifying questions.
- **Simulator, replay mode (`full_replay`, primary).** The plan's prescribed
  fix: before any candidate runs, Llama writes one 8-turn user script per
  seed seeing only the goal and its own earlier messages (never assistant
  text); every model and branch answers the same user turns, so A4 is at
  chance by construction and no author signal can travel via user turns.
- **D1.** Word 1–2-gram + char 2–4-gram TF-IDF + 37 stylometric features,
  multinomial LR; C picked by seed-grouped 5-fold CV on train seeds only.
- **D2.** Each turn is scored with exactly the prompt the candidate would
  have generated from (own template, generation header, thinking disabled),
  so Qwen3's current-turn-only empty think block needs no special handling.
  Primary D2 = mean token log-prob (as in the plan). Also reported:
  D2sum (total log-likelihood — tokenizer-independent, the Bayes-optimal
  version of raw likelihood) and D2cal (LR on the likelihood features, fit on
  train seeds) as exploratory. Also exploratory: likelihood analogues of
  A1/A2 (`A1_D2`, `A2_D2`). They mix generator drift with the scorer's own
  in-context adaptation to the (different) histories, so they are not clean
  contamination measures.
- **D3.** τ searched over all 7 internal boundaries with A≠B (forced
  switch). Detection score = best A≠B two-segment score − best one-segment
  score; AUROC is switched (by relation) vs all clean test dialogues.
  Random baselines: exact 1/7, ±1 3/7.
- **CIs.** Seed-cluster bootstrap (2000 reps; 1000 for AUROC).
- **A4.** User turn t (t≥2) → the model that produced assistant turn t−1,
  same D1 pipeline; trained on train seeds, tested on test seeds.
- **Seeds.** WildChat-1M shards 0–1: English, ≥2 user turns, not
  toxic/redacted/flagged, 40–1200 chars, no links/attachments/code blocks,
  no jailbreak, identity or image-prompt-generator requests, deduped; 21
  rejected in manual review and replaced deterministically.
