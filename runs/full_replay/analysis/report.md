# TurnPrint early pilot — full_replay

Models: qwen3-8b, qwen3-32b, gemma3-12b, gemma3-27b. Chance = 0.250. 95% seed-cluster bootstrap CIs.

## Preconditions
- A4 user-only author accuracy: 0.250 [0.250, 0.250] (gate ≤ 0.3)
- D1 clean per-turn accuracy: 0.794 [0.759, 0.826] (C=10.0)
- D1 clean accuracy by turn: t1=0.59, t2=0.76, t3=0.81, t4=0.82, t5=0.80, t6=0.84, t7=0.88, t8=0.86
- d2mean clean per-turn accuracy (argmax): 1.000 [1.000, 1.000]
- d2sum clean per-turn accuracy (argmax): 1.000 [1.000, 1.000]
- d2cal clean per-turn accuracy (argmax): 1.000 [1.000, 1.000]

## Data audit
- truncation rate: qwen3-8b=0.055, qwen3-32b=0.059, gemma3-12b=0.008, gemma3-27b=0.007
- mean tokens/turn: qwen3-8b=1272, qwen3-32b=1330, gemma3-12b=1023, gemma3-27b=1014
- rows with think tags in raw output: 0

## A1 contamination index CI(k)
| relation | pooled | k=2 | k=4 | k=6 |
|---|---|---|---|---|
| same | 0.129 [0.097, 0.163] | 0.079 [0.038, 0.123] | 0.121 [0.075, 0.164] | 0.188 [0.147, 0.234] |
| cross | 0.110 [0.090, 0.131] | 0.047 [0.026, 0.068] | 0.115 [0.086, 0.147] | 0.167 [0.138, 0.198] |

Exploratory D2 analogue: shift of (ℓ_A − ℓ_B) on B's first turn, nats/token
| relation | pooled | k=2 | k=4 | k=6 |
|---|---|---|---|---|
| same | -0.007 [-0.012, -0.001] | 0.001 [-0.008, 0.012] | -0.007 [-0.015, 0.002] | -0.014 [-0.023, -0.004] |
| cross | -0.086 [-0.097, -0.074] | -0.073 [-0.092, -0.051] | -0.081 [-0.098, -0.065] | -0.103 [-0.120, -0.085] |

## A2 transient: P_D1(B) branch − clean (all pairs)
- τ=2: j=1: -0.070 [-0.109, -0.033], j=2: -0.046 [-0.080, -0.012], j=3: -0.031 [-0.067, 0.003], j=4: -0.016 [-0.054, 0.023], j=5: -0.040 [-0.069, -0.012], j=6: -0.059 [-0.089, -0.031]
- τ=4: j=1: -0.160 [-0.198, -0.122], j=2: -0.112 [-0.152, -0.074], j=3: -0.130 [-0.159, -0.102], j=4: -0.101 [-0.132, -0.069]
- τ=6: j=1: -0.242 [-0.277, -0.208], j=2: -0.200 [-0.234, -0.167]

## A2_D2 transient: D2 margin of B, branch − clean (nats/token) (all pairs)
- τ=2: j=1: -0.012 [-0.022, -0.004], j=2: -0.009 [-0.015, -0.003], j=3: -0.008 [-0.015, -0.001], j=4: -0.010 [-0.018, -0.002], j=5: 0.000 [-0.009, 0.009], j=6: -0.007 [-0.013, -0.001]
- τ=4: j=1: -0.011 [-0.019, -0.003], j=2: -0.010 [-0.017, -0.003], j=3: -0.005 [-0.014, 0.003], j=4: -0.005 [-0.012, 0.001]
- τ=6: j=1: 0.000 [-0.009, 0.009], j=2: -0.001 [-0.007, 0.004]

## A3 localization / detection (Table 1)
Random: exact 0.143, ±1 0.429

| evidence/relation | τ exact | τ ±1 | (A,B) correct | joint | detection AUROC |
|---|---|---|---|---|---|
| D1/same | 0.375 [0.325, 0.427] | 0.658 [0.610, 0.705] | 0.788 [0.732, 0.838] | 0.365 [0.312, 0.420] | 0.716 [0.671, 0.762] |
| D1/cross | 0.803 [0.770, 0.837] | 0.902 [0.878, 0.924] | 0.627 [0.573, 0.678] | 0.566 [0.511, 0.618] | 0.943 [0.925, 0.959] |
| D1/all | 0.661 [0.631, 0.689] | 0.821 [0.797, 0.843] | 0.681 [0.634, 0.726] | 0.499 [0.451, 0.546] | 0.867 [0.843, 0.891] |
| D2/same | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2/cross | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2/all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2sum/same | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2sum/cross | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2sum/all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2cal/same | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2cal/cross | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| D2cal/all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

D2 − D1 exact-hit (paired): same: 0.625 [0.573, 0.675], cross: 0.197 [0.163, 0.230]

## Gates
- user_only_shortcut_ok: True
- user_only_threshold: 0.3
- clean_d1_above_chance: True
- A1_cross_pooled_ci_excludes_0: True
- A1_same_pooled_ci_excludes_0: True
- loc_exact>=2x_random[D1/same]: True
- loc_exact>=2x_random[D1/cross]: True
- loc_exact>=2x_random[D1/all]: True
- loc_exact>=2x_random[D2/same]: True
- loc_exact>=2x_random[D2/cross]: True
- loc_exact>=2x_random[D2/all]: True
- loc_exact>=2x_random[D2sum/same]: True
- loc_exact>=2x_random[D2sum/cross]: True
- loc_exact>=2x_random[D2sum/all]: True
- loc_exact>=2x_random[D2cal/same]: True
- loc_exact>=2x_random[D2cal/cross]: True
- loc_exact>=2x_random[D2cal/all]: True
