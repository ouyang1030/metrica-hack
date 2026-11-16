# Experiments

One entry per hypothesis: date, hypothesis, change, off-game spellF1 / macroF1 (clean and
degraded) with intervals, BAR-ATH diagnostics, decision.

**Decision rule.** A change is accepted when it improves mean spellF1 on the degraded off-game
harness beyond its bootstrap interval, without hurting macroF1 materially, and passes the
BAR-ATH diagnostics (summaries of our own predictions). If two variants are within noise, the
simpler one stays. The public 5-minute window never decides between variants.

**Commits.** Commit hashes cited below belong to the development history, which was not
published: the public repository starts from a single commit of the final state, and code an
entry says was removed is not in it.

## 2026-10-27 — v0: scaffold, baseline reproduction

- **Hypothesis.** The parsed cache plus a vectorised nearest-player rule reproduces the four
  published baselines byte for byte, and the one-command pipeline emits `baseline_smooth50`.
- **Change.** `mtah/` (ATD parser, cache, firewall, vectorised baseline, submission helpers),
  `scripts/make_submission.py`, `scripts/reproduce_baselines.py`, clean-room script, tests.
- **Off-game harness.** Not built yet (v1).
- **Synthetic bundle (random positions, 199,319 frames).** Vectorised codes and all four files
  are byte-identical to the official `starter/baseline.py` run on the same files, edge cases
  included (ties, NaN distances). Two clean-room runs give byte-identical files. Full pipeline
  including cache build: about 2 s, 0.65 GB peak RSS.
- **BAR-ATH.** Pending: the data bundle was not on the machine. Expected on first run: the four
  "identical to the committed file" checks pass, and `make_submission` writes a file
  byte-identical to `evaluation/results/baseline_smooth50.csv`.
- **Decision.** Accept as v0 once the BAR-ATH checks pass.

## 2026-10-27 — v1 step 1: off-game harness and noise simulator

- **Hypothesis.** IDSSE / DFL tracking, degraded to look like the ATD, reproduces the way the
  published baselines behave on BAR-ATH, so it can rank methods.
- **Change.** `harness/idsse.py` (7 matches, 14 halves, labels from `BallStatus` /
  `BallPossession`), `harness/degrade.py` (default parameters from DATA.md numbers only: 41% ball
  missing on in-play frames, 20.5 tracked objects, 46 slots), `harness/score.py`.
- **Off-game harness**, mean over 14 halves, 5–95% bootstrap interval, degraded = mean of seeds
  0–2:

  | method | clean spellF1 | clean macroF1 | degraded spellF1 | degraded macroF1 |
  |---|---|---|---|---|
  | baseline_raw | 0.098 [0.093, 0.105] | 0.363 | 0.017 [0.014, 0.019] | 0.500 |
  | baseline_smooth25 | 0.297 [0.282, 0.314] | 0.385 | 0.210 [0.202, 0.218] | 0.461 |
  | baseline_smooth50 | 0.309 [0.294, 0.325] | 0.391 | 0.254 [0.244, 0.265] | 0.424 |
  | baseline_smooth125 | 0.219 [0.206, 0.231] | 0.387 | 0.207 [0.191, 0.222] | 0.385 |

- **Reading.** Degraded spellF1 ranks the baselines like the published full-match scores
  (smooth50 best, raw far behind; published 0.049 / 0.200 / 0.243 / 0.224). macroF1 does not
  match (published 0.64–0.68): the simulated dropouts are short (median gap 0.12 s), so the
  baselines almost never say D, while the real ATD presumably loses the ball for long stretches
  around dead balls. The published scores are a comparison point only, never a calibration
  target (COMPLIANCE_LOG.md). Calibration waits for ATD aggregates under STRICT.
- **Decision.** Accept the harness as the decision tool for v1; calibrate the simulator once the
  data bundle is in place.

## 2026-10-27 — v1 step 2: evidence + HMM decoder (accepted)

- **Hypothesis.** Collective player speed and where the ball was last seen separate dead from
  alive; the closer team to the ball separates H from A; an HMM over {H, A, D} turns that into
  piecewise-constant spells and beats smoothing.
- **Exploration (off-game, degraded seed 0, 4 matches).** P(dead) is 0.89–1.00 when the median
  player speed is below 1 m/s and under 0.05 above 2 m/s; a ball seen beyond a line is dead
  99.6% of the time; the nearest player's team owns the ball 92.5% of the time within 1 m,
  ~60% beyond 3 m.
- **Change.** `mtah/features.py` (ATD fields only: slot positions and speeds, ball x/y),
  `mtah/emissions.py` (logistic P(dead | x): speed, speed², ball seen, last seen out, log time
  since seen; logistic P(home | alive, x) on exp(-d_home/1.5 m) - exp(-d_away/1.5 m)),
  `mtah/decoder.py` (Viterbi; transitions from off-game label counts; evidence tempered by
  kappa), `harness/train_v1.py`, `models/v1.json`.
- **Off-game harness**, leave one match out (fit on degraded seed 0 of six matches, score the
  seventh clean and on seeds 0–2), 14 halves:

  | method | clean spellF1 | clean macroF1 | degraded spellF1 | degraded macroF1 |
  |---|---|---|---|---|
  | baseline_smooth50 | 0.309 [0.294, 0.325] | 0.391 | 0.254 [0.244, 0.265] | 0.424 |
  | v1 kappa=0.2 | 0.387 [0.377, 0.399] | 0.782 | 0.318 [0.303, 0.333] | 0.763 |
  | **v1 kappa=0.4** | **0.410 [0.400, 0.420]** | 0.776 | **0.334 [0.324, 0.345]** | 0.755 |
  | v1 kappa=0.7 | 0.377 [0.367, 0.386] | 0.769 | 0.327 [0.315, 0.340] | 0.746 |
  | v1 kappa=1.0 | 0.336 | 0.762 | 0.304 | 0.737 |

  Paired difference to smooth50, degraded: spellF1 **+0.080 [+0.070, +0.091]**, macroF1
  +0.331 [+0.316, +0.345]. kappa was picked on the same 14 halves (one parameter; neighbours
  within 0.01).
- **Where it still loses** (3 matches, seed 1): alive/dead F1 0.85–0.93 and the number of D
  segments is about right, but it predicts 130–150 possession spells per half where the labels
  have 175–275, with H/A accuracy 0.72–0.87. Team switches are too few and too late.
- **BAR-ATH.** Not run (no data). Synthetic full-length bundle: valid, two runs byte-identical.
- **Decision.** Accept; `mtah/pipeline.py` is now v1 (kappa 0.4).

## 2026-10-27 — separate temperatures for dead and team evidence (rejected)

- **Hypothesis.** Play stops rarely but possession changes often, so team evidence should be
  trusted more (kappa_team > kappa_dead) to recover the missing switches.
- **Result** (same protocol): best pair 0.3 / 0.4 at +0.084 [+0.075, +0.094] vs +0.080 for one
  kappa — within noise; kappa_team 0.8 no better, 1.6 clearly worse (+0.023). Trusting the
  nearest-player signal more adds wrong switches as fast as right ones.
- **Decision.** Keep one temperature (the code keeps the factorised form). The fix has to be
  better possession evidence, not a different weight on the current one.

## 2026-10-27 — where v1 loses: short possessions

- **Diagnosis** (all 14 halves, degraded seed 1, model fitted on all matches — for understanding
  only). Recall at IoU 0.5 by ground-truth spell length: < 1 s 0.02 (12.5% of spells), 1–2 s
  0.05 (17.5%), 2–4 s 0.09 (22.9%), 4–8 s 0.36 (20.3%), 8–16 s 0.57 (15.2%), > 16 s 0.74–0.80
  (11.6%). Half the DFL spells are shorter than 4 s and v1 finds almost none of them; it predicts
  1,979 spells where the labels have 3,061. DFL's notion of possession may be finer-grained than
  Metrica's; the harness compares methods, it does not say how many spells BAR-ATH has.

## 2026-10-27 — simulator: ball positional noise

- **Change.** `degrade.py` adds 0.3 m ball jitter (correlation 0.2 s), a guess until calibrated:
  without it, ball velocities on the harness would be exact. The harness break between halves is
  now 15 min (was 1 min), so that windows of a few minutes never straddle two halves.
- **Effect.** v1 (near, kappa 0.4) degraded spellF1 0.334 -> 0.336; smooth50 0.254 -> 0.249.
  New reference numbers below are on this simulator.

## 2026-10-27 — control evidence: ball and player moving together (rejected)

- **Hypothesis.** A nearby player whose velocity matches the ball's controls it; that sharpens
  team evidence enough to catch quick turnovers.
- **Change.** `features.py`: ball and slot velocities by central differences (±2 frames), relative
  speed to the nearest slot of each team; team term `control` = closeness × exp(-rel / 2 m/s).
- **Result.** Weight 2.0 next to closeness 5.0, but within 1.5 m the nearest team owns the ball
  83–95% of the time whatever the relative speed, and with 0.3 m ball jitter ±2 m/s is normal
  even when dribbling. Paired difference to v1: spellF1 +0.000 [-0.001, +0.001].
- **Decision.** Reject. The features stay in `features.py` (cheap), unused by the model.

## 2026-10-27 — team movement while the ball is lost (accepted)

- **Hypothesis.** The whole block moves towards the goal of the team without the ball; that
  says who has it on the 41% of frames without a ball.
- **Exploration.** Sign of the mean player velocity towards the away goal (direction of attack
  from where the teams sit over ±2 min): right 64% of alive frames, 68% when the ball is lost,
  72.5% when the movement is clear. Spread (home more stretched along the pitch): 60%. Relative
  height of the teams: no signal (47%).
- **Change.** `features.py`: `flow`, `spread`; the team model now trains on every alive frame.
- **Off-game harness** (leave one match out, degraded seeds 0–2, 14 halves):

  | variant (kappa 0.4) | clean spellF1 | degraded spellF1 | degraded macroF1 |
  |---|---|---|---|
  | baseline_smooth50 | 0.309 | 0.249 [0.236, 0.263] | 0.422 |
  | near (v1) | 0.407 | 0.336 [0.324, 0.349] | 0.761 |
  | **near+flow** | 0.411 | **0.358 [0.347, 0.370]** | 0.771 |
  | near+flow+spread | 0.404 | 0.345 [0.333, 0.359] | 0.772 |

  near+flow minus near: spellF1 **+0.022 [+0.009, +0.035]**, macroF1 +0.011 [+0.003, +0.018].
  Adding spread: +0.009 [-0.004, +0.022] over near, i.e. worse than flow alone.
- **Decision.** Accept near+flow; reject spread.

## 2026-10-27 — duration chains and the temperature (chains rejected, kappa 0.2)

- **Hypothesis.** Possessions have a typical length, not a geometric one; k sub-states per state
  (minimum k frames, duration peaking above zero) cut implausible spells.
- **Result** (near+flow, same protocol): chain 2 at kappa 0.4 0.372, chain 3 at 0.55 0.372, vs
  chain 1 at 0.25 0.371 — paired +0.002 [+0.001, +0.003] and +0.001. The chain and the
  temperature trade off against each other; neither adds anything. Extending the temperature
  grid for chain 1: 0.1 0.347, 0.15 0.364, **0.2 0.371**, 0.25 0.371 (clean 0.420 / 0.427).
- **Decision.** Reject chains (within noise, keep the simpler decoder); kappa 0.2.

## 2026-10-27 — boundary refinement (rejected)

- **Diagnosis** (matched spells, IoU >= 0.3, degraded seed 1): starts unbiased (median 0
  frames), ends late — median +20 frames before a stoppage, +22 before a turnover.
- **Hypothesis.** Moving each switch back to its physical onset (the new team's first located
  touch; the ball first seen beyond a line), at most 2 s and never past the previous switch,
  removes that lag.
- **Result.** Degraded spellF1 0.374 vs 0.371, paired +0.003 [+0.001, +0.006]; clean 0.415 vs
  0.420.
- **Decision.** Reject (within noise, clean worse). `mtah/refine.py` stays, switched off.

## v1 as it stands (2026-10-27)

near+flow, one temperature 0.2, plain HMM, no refinement: degraded spellF1 **0.371
[0.359, 0.384]**, macroF1 0.782; clean 0.420 / 0.793. Against baseline_smooth50 on the same
copies: spellF1 **+0.122 [+0.108, +0.138]**, macroF1 +0.360 [+0.343, +0.377]. Not run on BAR-ATH
(no data). Main open problem: possessions shorter than ~4 s.

## 2026-10-27 — v2: one gradient-boosted frame classifier (accepted)

- **Hypothesis.** v1 sees two numbers per frame. A classifier over a wide table of
  permutation-invariant tracking features, with context before *and after* each frame (the
  pipeline is offline), gives evidence sharp enough for the same HMM to find the short
  possessions v1 misses (v1 found 5.6% of those under 4 s).
- **Change.** `features.matrix`: 66 columns — collective and per-team speed, ball located / time
  since and until located / last and next location beyond a line, ball speed, stillness and
  place (distance to the touchline, the goal line, each team's target goal), ball velocity
  towards each team's goal, per team: nearest distance, closeness in 0.5 / 1 / 2 s centred
  windows and 2 s before / after, players within 2 / 5 / 10 m, depth, push towards goal (same
  windows), length, width. Every team column comes as a home / away pair measured towards the
  goal that team attacks, so swapping the teams permutes the columns (tested). Classifier:
  scikit-learn `HistGradientBoostingClassifier`, 100 iterations, 31 leaves, at least 100 frames
  per leaf, learning rate 0.1, no early stopping — defaults apart from the leaf size, not tuned.
  Trained on every 5th in-play frame of degraded seed 0 of the training matches plus the same
  frames with the teams swapped. P(H, A, D | frame) feeds the v1 HMM (`decode_probs`). The
  trees are exported to JSON (811 kB) and evaluated in numpy (`mtah/trees.py`; equal to
  scikit-learn to 1e-12, tested), so the pipeline needs no scikit-learn.
- **Stress copy.** New in the protocol: one more copy of each held-out match, much noisier than
  the simulator's defaults (ball missing 55%, no extra dropout off the pitch, players missing
  15%, team errors 5%, swaps 20% within 2 m, jitter 0.4 m / ball 0.6 m). Never trained on. It
  asks whether a gain rests on the particular noise the classifier saw in training.
- **Result** (leave one match out, 14 halves, spellF1 [5-95%] / macroF1):

  | | clean | degraded | stress |
  |---|---|---|---|
  | baseline_smooth50 | 0.309 / 0.391 | 0.249 / 0.422 | 0.162 / 0.331 |
  | v1 (kappa 0.2) | 0.420 / 0.793 | 0.371 [0.359, 0.384] / 0.782 | 0.296 / 0.729 |
  | v2, kappa 0.2 | 0.593 / 0.920 | 0.578 / 0.909 | 0.485 / 0.845 |
  | **v2, kappa 0.3** | 0.611 / 0.922 | **0.589 [0.569, 0.609]** / 0.910 | **0.486** / 0.842 |
  | v2, kappa 0.6 | 0.632 / 0.924 | 0.595 / 0.910 | 0.464 / 0.837 |
  | v2, kappa 1.0 | 0.636 / 0.925 | 0.591 / 0.910 | 0.436 / 0.834 |

  v2 (kappa 0.3) minus v1, paired: degraded spellF1 **+0.218 [+0.201, +0.235]**, macroF1 +0.128
  [+0.118, +0.137]; stress +0.190 [+0.170, +0.209] / +0.113. Minus smooth50: degraded +0.340
  [+0.319, +0.363].
  Possessions found (IoU 0.5, degraded, 3 seeds): under 4 s 5.6% (v1) -> 23.1%; 4 s and longer
  55.7% -> 78.6%. Predicted / true possessions 0.57 -> 0.65: still too few switches.
- **Temperature.** The degraded score is flat from 0.3 to 1.0 (0.589-0.595) while the stress score
  falls (0.486 -> 0.436): sharper evidence pays only when the noise is the noise trained on. The
  ATD's noise level is not known yet, so kappa is chosen on the mean of degraded and stress
  spellF1 -> 0.3. Re-choose after calibration.
- **Too good to be true?** Checked before accepting. No input is derived from the labels
  (`harness/idsse.py` takes only positions and speeds; possession and status go to the labels).
  Folds are whole matches. The gain holds on the stress copy. Permutation importance (log loss
  0.18-0.21 on a held-out degraded copy, three folds; increase when a group is shuffled): ball
  motion and place +0.53-0.54, players' speed +0.35-0.43, team depth / push / shape
  +0.16-0.19, closeness windows +0.12-0.15, ball located / out +0.03-0.04, closeness of this
  frame alone ~0.01 (the windows carry it). Nothing points at an artefact; the gain looks like
  context plus non-linearity over a v1 that saw two numbers. What the harness cannot test:
  whether the ATD's noise and Metrica's label semantics are close enough to the simulator and
  DFL's for the gain to carry over.
- **Decision.** Accept v2 at kappa 0.3 (`models/v2.json`, refitted on all seven matches); the
  pipeline now runs v2 (7 s and 1.2 GB on a full-length synthetic bundle, two runs
  byte-identical). Cross-validation reruns give identical numbers.

## 2026-10-27 — diagnosis: where v2 loses possessions (no change)

Folds 0, 2, 4, 6 held out, degraded seed 1, kappa 0.3; 1,704 true possessions.

| true length | share | found | missed outright | merged into a longer spell | split | shifted |
|---|---|---|---|---|---|---|
| < 2 s | 30% | 0.08 | 0.59 | 0.26 | 0.00 | 0.07 |
| 2-4 s | 22% | 0.38 | 0.26 | 0.23 | 0.00 | 0.12 |
| 4-10 s | 26% | 0.69 | 0.04 | 0.21 | 0.00 | 0.06 |
| >= 10 s | 22% | 0.94 | 0.00 | 0.05 | 0.01 | 0.00 |

- v2 under-segments and never over-segments. Matched spells: start median 0 frames (q10 -34,
  q90 +21), end median +4 (q10 -25, q90 +52).
- What separates the two halves of a merge: true segments lying between two spells of one team,
  reproduced by the decoder — the other team's spell: under 1 s 33% (131), 1-2 s 45% (163),
  2-4 s 68% (225), 4-10 s 95%; a stoppage: only 22 such stoppages, almost all found. All
  stoppages: 94-100% found from 2 s up; none shorter than 1 s exists.
- Oracle boundaries (the true segments, each labelled by v2's mean probabilities): spellF1 0.696
  against 0.593 decoded on the same folds. The headroom is in the segmentation, not the labels.
- Reading: the misses are short spells of the other team inside one team's possession. Under
  Metrica's written rule (a touch without control does not change possession) many of these
  would not be spells at all, so chasing them on DFL labels risks flicker on BAR-ATH. Plan (the
  v2 plan, in git history): step 0 first.

## 2026-10-28 — plan step 0: Metrica's possession rule as a second label set (harness change; training on it rejected)

- **Hypothesis.** Some of DFL's sub-second spells of the other team are touches without control,
  which Metrica's ground truth does not count (CHALLENGE.md). Scoring only on DFL's
  labels would reward chasing them.
- **Change.** `harness/semantics.py`: a spell of one team shorter than a control time, between
  two spells of the other team, goes to the other team (left to right; spells next to a stoppage
  kept; tested). Every prediction is now scored against three label sets: DFL (as is), M1
  (control time 1 s) and M2 (2 s, a check). On the seven matches M1 removes 12% of the
  possessions (3,061 -> 2,705; under 1 s 13% -> 6%; median 3.7 -> 4.4 s) but only 0.2% of the
  frames; M2 removes 26%. The temperature is now chosen on the mean of degraded and stress
  spellF1 over DFL and M1. Organiser question 6 asks which rule the ground truth follows.
- **Result** (spellF1, degraded [5-95%]; stress):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | v1 | 0.371 / 0.296 | 0.405 / 0.325 | 0.461 / 0.362 |
  | v2, kappa 0.2 | 0.578 / 0.485 | 0.629 / 0.524 | **0.690** / 0.572 |
  | **v2, kappa 0.3** | 0.589 [0.569, 0.609] / 0.486 | 0.638 [0.625, 0.651] / 0.521 | 0.686 / 0.556 |
  | v2, kappa 0.6 | 0.595 / 0.464 | 0.638 / 0.492 | 0.667 / 0.508 |
  | v2, kappa 1.0 | 0.591 / 0.436 | 0.630 / 0.458 | 0.645 / 0.465 |
  | v2 trained on M1, kappa 0.3 | 0.586 / 0.484 | 0.635 / 0.521 | 0.685 / 0.560 |

  Trained on M1 minus trained on DFL (kappa 0.3 each): DFL -0.003 [-0.007, +0.000], M1 -0.003
  [-0.007, +0.000], M2 -0.001 [-0.004, +0.002]; stress within +-0.004. v2 minus v1 holds on
  every label set (+0.218 / +0.233 / +0.225).
- **Reading.** The label rule moves the scores by 0.05-0.10 but not the ranking of v1 and v2.
  The temperature is where it matters: the stricter the rule, the more a high kappa (more
  switches) costs; under M2 the best is 0.2 and kappa 1.0 loses 0.045. Chasing DFL's
  sub-second spells by sharpening the evidence would be a bet on DFL's semantics. Predicted /
  true possessions at kappa 0.3: 0.65 (DFL), 0.74 (M1), 0.88 (M2).
- **Decision.** Keep training on DFL labels (equal within noise; no change). Keep kappa 0.3 (best
  on the new criterion too). Every later step is judged on DFL and M1 and must not lose on M2.
  `models/v2.json` unchanged.

## 2026-10-28 — plan step 1: more and randomised training noise (three default copies accepted; randomised noise deferred)

- **Hypothesis** (domain randomisation, Tobin 2017). Training on one default-noise copy fits the
  classifier to one guess of the ATD's noise; noise drawn over a range should hold up better
  when the real noise differs.
- **Change.** `degrade.drawn()`: every simulator parameter that differs between the defaults and
  the stress copy is drawn uniformly between the two. Training copies per match (each gives every
  5th in-play frame, the n-th copy from the n-th frame on): one default copy (as before), three
  default copies (seeds 0-2), one or three randomised copies (seeds 10-12). Note: the stress copy
  now lies at the edge of the randomised training range, so for those variants it is no longer
  noise never trained on.
- **Result** (spellF1 degraded / stress; each variant at its best kappa, chosen as in step 0):

  | training copies | kappa | DFL | M1 | M2 |
  |---|---|---|---|---|
  | one default (before) | 0.3 | 0.589 / 0.486 | 0.638 / 0.521 | 0.686 / 0.556 |
  | **three default** | 0.3 | **0.595** / 0.492 | **0.644** / 0.528 | **0.692** / 0.566 |
  | one randomised | 0.6 | 0.567 / 0.538 | 0.611 / 0.576 | 0.656 / 0.609 |
  | three randomised | 0.45 | 0.584 / 0.544 | 0.630 / 0.585 | 0.676 / 0.621 |

  Minus one default copy, paired, degraded: three default +0.006 [+0.002, +0.010] (DFL),
  +0.006 [+0.001, +0.010] (M1), +0.006 [+0.001, +0.011] (M2), stress +0.006 to +0.011, macroF1
  +0.001 to +0.003. Three randomised: degraded -0.005 [-0.008, -0.003] / -0.008 / -0.010,
  stress +0.058 / +0.064 / +0.065, macroF1 -0.005 degraded, +0.044 stress.
- **Reading.** More copies of the same noise help a little, on every label set. Randomised noise
  buys a lot of robustness to heavy noise at a small cost under the default noise: which one
  wins on BAR-ATH depends on how noisy the ATD really is, which only the calibration can say.
- **Decision.** Accept three default copies (beyond the interval on all three label sets). Keep
  randomised-only training for the re-choice after calibration. Next: default + randomised
  copies together, run with step 2.

## 2026-10-28 — plan step 2: switch-aware decoding (change detector accepted; switch factor rejected) and mixed training noise (deferred)

Reference: v2 trained on three default copies (step 1), constant-transition HMM, kappa 0.3.

- **Mixed noise** (three default + three randomised copies per match; best kappa 0.45). Minus
  the reference, degraded: DFL -0.000 [-0.007, +0.007], M1 -0.002 [-0.009, +0.006], M2 -0.007
  [-0.013, -0.000]; stress +0.046 / +0.051 / +0.053, macroF1 stress +0.038. Equal under the
  default noise (slightly worse on M2), much better under heavy noise; the stress copy is inside
  its training range. **Deferred** to the calibration with the randomised-only variant: if the ATD
  is noisier than the simulator's defaults, this is the one to take.
- **2a. Switch factor** (possession switches H <-> A made 2x or 0.5x as likely per frame, kappa
  re-chosen): every difference within +-0.003. A constant cost per switch mostly duplicates what
  kappa does (kappa scales the evidence against all transition costs at once). **Rejected**,
  code removed.
- **2b. Change detector, input-output HMM** (Bengio & Frasconi; PathCRF). A second classifier
  on the same 66 features and copies gives P(no change, possession switch, stop, restart within
  +-0.2 s | frame) (`harness/train_v2.py: change_kinds, fit_first`). Its P(kind) / 11 is that
  kind's per-frame hazard, blended with the HMM's constant one in log space by gamma
  (`decoder.input_transitions`; gamma 0 is the plain HMM). Result, spellF1 degraded / stress:

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | constant HMM, kappa 0.3 | 0.595 / 0.492 | 0.644 / 0.528 | 0.692 / 0.566 |
  | detector, gamma 0.5, kappa 0.3 | 0.598 / 0.493 | 0.646 / 0.528 | 0.691 / 0.565 |
  | detector, gamma 1, kappa 0.3 | 0.604 / 0.496 | 0.651 / 0.530 | 0.693 / 0.563 |
  | **detector, gamma 1, kappa 0.2** (chosen) | 0.598 / 0.501 | 0.647 / 0.540 | 0.699 / 0.583 |
  | detector, gamma 1, kappa 0.45 | 0.610 / 0.486 | 0.655 / 0.517 | 0.684 / 0.538 |

  Chosen minus reference, paired, degraded: DFL +0.002 [+0.000, +0.005], M1 +0.003 [+0.001,
  +0.006], M2 +0.006 [+0.004, +0.008]; stress +0.009 [+0.003, +0.014] / +0.012 / +0.017; macroF1
  +0.002 degraded, +0.004 stress. Clean: 0.618 -> 0.625 (0.636 at kappa 0.3). gamma 1 beats 0.5
  everywhere.
- **Reading.** A real but small gain, on every label set and noise level; the decoder now knows
  *where* a switch is plausible, not only how often. It does not fix the merges by itself:
  predicted / true possessions stay at 0.63-0.71.
- **Decision.** Accept the change detector, gamma 1, kappa 0.2. Next run tries gamma 1.5 too.
- **Run note.** This run took 4.6 h instead of ~25 min: a second training job ran alongside it
  (two OpenMP jobs on the same cores), and the machine was short of memory (swapping) and slept.
  Now one job at a time, under `caffeinate`; randomised copies are only built when a variant
  uses them, and second-stage tables are built one at a time.

## 2026-10-28 — plan step 3: stacked second stage over ±8 s (accepted)

Reference: three default copies, change detector (step 2), now also tried at gamma 1.5.

- **Hypothesis** (stacked sequential learning, Cohen & Carvalho 2005; auto-context, Tu 2008;
  MS-TCN). The first stage sees +-2 s; a second stage that sees how the first stage's
  probabilities and change scores move over up to +-8 s can tell a real switch from a blip and
  find the spells the decoder merges.
- **Change.** `features.context` / `second_stage`: the 66 features plus, for P(H), P(A), P(D) and
  the detector's P(switch), P(stop), P(restart), the value and its means over 0.5, 1, 2, 4 and
  8 s before and after the frame and after minus before (96 columns). The second classifier
  (same settings) learns from out-of-fold first-stage outputs: for each training match the first
  stage is refitted on the other training matches (`fit_second`). The detector still drives the
  transitions. Cost: seven first-stage fits per fold instead of one (6 min per fold).
- **Result** (spellF1, degraded [5-95%] / stress):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | constant HMM, kappa 0.3 | 0.595 / 0.492 | 0.644 / 0.528 | 0.692 / 0.566 |
  | detector, kappa 0.2, gamma 1 (step 2) | 0.598 / 0.501 | 0.647 / 0.540 | 0.699 / 0.583 |
  | detector, kappa 0.3, gamma 1.5 | 0.609 / 0.498 | 0.654 / 0.532 | 0.689 / 0.559 |
  | **stacked, kappa 0.3, gamma 1.5** | **0.621** [0.602, 0.640] / 0.526 | **0.670** [0.657, 0.681] / 0.564 | **0.704** / 0.593 |
  | stacked, kappa 0.2, gamma 1 | 0.608 / 0.517 | 0.660 / 0.556 | 0.713 / 0.604 |

  Stacked minus detector alone (both kappa 0.3, gamma 1.5), paired, degraded: DFL +0.012
  [+0.008, +0.015], M1 +0.015 [+0.011, +0.019], M2 +0.015 [+0.012, +0.019]; stress +0.029 /
  +0.032 / +0.034; macroF1 +0.006 degraded, +0.010 stress. Clean 0.647 -> 0.656. Short
  possessions found (DFL, degraded): 27.2% -> 27.7%; 4 s and longer 79.8% -> 80.9%.
- **Reading.** The longer view helps everywhere, most under heavy noise, where +-2 s says least.
  Gamma 1.5 over 1 helps on DFL and M1, costs a little on M2 -- the label-semantics trade-off of
  step 0 again. Since the start of the plan: DFL 0.589 -> 0.621, M1 0.638 -> 0.670, M2 0.686 ->
  0.704, stress 0.486 -> 0.526.
- **Decision.** Accept the second stage, kappa 0.3, gamma 1.5.

## 2026-10-28 — plan step 4: touch and duel features (accepted)

Reference: the stacked model of step 3 (kappa 0.3, gamma 1.5), refitted in the same run with the
new columns left out; it reproduces step 3's numbers exactly.

- **Hypothesis** (Vidal-Codina et al. 2022; Link & Hoernig 2017). Control and a deflection both
  change the ball's velocity next to a player; who touches it, how often, and whether an opponent
  is as close tell one from the other -- which is Metrica's rule.
- **Change** (`features.matrix`, 19 new columns, 85 in all): `impulse`, how much the ball's
  velocity changes from the 0.2 s before a frame to the 0.2 s after; per team, that impulse
  where the team's nearest player is within 1.5 m of the ball and no opponent is nearer, in
  windows of 0.5, 1 and 2 s before and after; `duel`, both teams within 1.5 m (0.5 and 1 s
  windows); per team, how long it has been and will stay nearest to where the ball was last
  seen. 1.5 m rather than the papers' 1 m because the simulator's ball jitters by 0.3 m.
- **Result** (spellF1, degraded [5-95%] / stress; clean in brackets):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | stacked, no touch features | 0.621 [0.602, 0.640] / 0.526 (0.656) | 0.670 / 0.564 (0.701) | 0.704 / 0.593 (0.737) |
  | **stacked, touch features** | **0.635** [0.619, 0.652] / 0.526 (0.707) | **0.683** / 0.564 (0.751) | **0.714** / 0.587 (0.762) |

  Paired, degraded: DFL +0.014 [+0.008, +0.020], M1 +0.013 [+0.007, +0.020], M2 +0.010
  [+0.005, +0.016]; macroF1 +0.003 [+0.002, +0.004]. Stress: -0.000 [-0.006, +0.005], -0.000,
  -0.006 [-0.013, +0.001]. Clean +0.051, macroF1 0.932 -> 0.944. Short possessions found (DFL,
  degraded) 27.7% -> 29.5%; 4 s and longer 80.9% -> 82.5%.
- **Reading.** The gain grows with how precisely the ball is located: large on clean data,
  +0.01 under the default noise, nothing under the stress copy's 0.6 m ball jitter and 55%
  dropout. So on BAR-ATH it depends on how good the ATD's ball positions are, and the simulator
  does not model false ball detections yet. No leakage: the columns come from positions only.
- **Decision.** Accept (beyond the interval on all three label sets, no loss under stress).
  Re-check after the calibration, with false detections added to the simulator.
- **Run note.** The two stacked models per fold took 11.5 min per fold when the machine was
  awake and had memory; the first two folds took hours (sleep, swapping).

## v2 as it stands (2026-10-28, after plan steps 0-4)

Three default-noise training copies per match; first-stage classifier (85 features) and change
detector; second stage over +-8 s of both, trained on out-of-fold outputs; input-output HMM,
kappa 0.3, gamma 1.5. `models/v2.json` refitted on all seven matches (2.7 MB). Leave one match
out, spellF1 degraded [5-95%] / stress / clean:

| | DFL | M1 | M2 |
|---|---|---|---|
| baseline_smooth50 | 0.249 / 0.162 / 0.309 | 0.261 / 0.166 / 0.329 | 0.264 / 0.161 / 0.343 |
| v1 | 0.371 / 0.296 / 0.420 | 0.405 / 0.325 / 0.460 | 0.461 / 0.362 / 0.513 |
| v2 as accepted on 10-27 | 0.589 / 0.486 / 0.611 | 0.638 / 0.521 / 0.661 | 0.686 / 0.556 / 0.712 |
| **v2 now** | **0.635** [0.619, 0.652] / 0.526 / 0.707 | **0.683** / 0.564 / 0.751 | **0.714** / 0.587 / 0.762 |

macroF1 degraded 0.910 -> 0.923. Steps, each paired against the previous (DFL, degraded): three
copies +0.006, change detector +0.002 (+0.009 stress), second stage +0.012, touch features
+0.014. Pipeline on a full-length synthetic bundle: 19.5 s, 1.6 GB, validator passes, two runs
byte-identical. Still open: the merges (predicted / true possessions 0.69 on DFL); the choice
between default and randomised training noise, which waits for the calibration.

## 2026-10-29 — diagnosis of v2 after plan steps 0-4 (no change)

Same protocol as the diagnosis of 10-27 (folds 0, 2, 4, 6, degraded seed 1), current model
(stacked, touch features, change detector, kappa 0.3, gamma 1.5). In brackets: 10-27.

| true length (DFL) | n | found | missed outright | merged into a longer spell | split | shifted |
|---|---|---|---|---|---|---|
| < 2 s | 519 | 0.15 (0.08) | 0.53 (0.59) | 0.23 (0.26) | 0.00 | 0.08 |
| 2-4 s | 375 | 0.47 (0.38) | 0.23 (0.26) | 0.21 (0.23) | 0.00 | 0.09 |
| 4-10 s | 442 | 0.73 (0.69) | 0.04 (0.04) | 0.17 (0.21) | 0.00 | 0.07 |
| >= 10 s | 368 | 0.95 (0.94) | 0.00 | 0.04 (0.05) | 0.01 | 0.00 |

- Matched spells: start median -1 frame (q10 -30, q90 +18; was -34, +21), end +1 (q10 -21, q90
  +39; was +4, -25, +52). Predicted / true possessions 0.68 (0.65); on M1 0.78.
- Decoded 0.646, oracle boundaries 0.739 (0.593 / 0.696): both up ~0.05, the gap unchanged.
  On M1: 0.701 / 0.800.
- **The decoder is no longer what loses the short spells.** For the other team's segments
  between two spells of one team, the share the decoder reproduces equals the share where the
  evidence (second-stage probabilities averaged over the segment) favours the true team: under
  1 s 0.40 / 0.39 (n 131), 1-2 s 0.52 / 0.54 (163), 2-4 s 0.73 / 0.70 (225), 4-10 s 0.95 / 0.92.
  Where the evidence points the right way, the decoder finds it; where it does not, no decoder
  can.
- **Learning curve** (first stage + change detector, no second stage, trained on 2 / 4 / 6
  matches): DFL 0.614 / 0.620 / 0.628, M1 0.662 / 0.673 / 0.679 -- about +0.008 per doubling.
  More matches of the same kind would help this model little.
- **Reading for plan step 5.** What is missing is information in the evidence about short
  spells -- who receives or intercepts the ball -- not decoding and not data volume for the
  trees. A neural sequence model over the raw player set could in principle find that
  information, but it needs far more than seven matches (the trees gain little from more), CPU
  training on this machine, and an export to numpy. Decision: not now; first targeted features
  for receptions and interceptions (where a ball flight ends and with whom), which test the same
  hypothesis for a fraction of the cost.

## 2026-10-29 — simulator: false ball detections; the touch features under them (kept)

- **Simulator change.** `degrade.false_detections`: in runs of 0.5 s on average, the reported
  ball is something else -- half the runs a player (the "ball" sits at his feet and follows
  him), the rest a fixed spot on the pitch or up to 3 m beyond its lines. `Params.ball_false`
  is the share of in-play frames; 0 by default (DATA.md says nothing about it, every earlier
  copy stays bit-identical; tested), to be set by the calibration (v2 plan).
- **Experiment.** The step-4 comparison (stacked, with vs without the touch features) rerun with
  false detections in every training and scored copy: 5% of in-play frames in the default
  copies, 15% in the stress copy (guesses).
- **Result** (spellF1, degraded / stress; both variants at kappa 0.3, gamma 1.5):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | no touch features | 0.602 / 0.506 | 0.652 / 0.546 | 0.697 / 0.577 |
  | touch features | 0.613 / 0.505 | 0.661 / 0.544 | 0.702 / 0.573 |
  | gain with false detections | +0.011 / -0.001 | +0.009 / -0.002 | +0.005 / -0.004 |
  | gain without them (step 4) | +0.014 / -0.000 | +0.013 / -0.000 | +0.010 / -0.006 |

  At each variant's chosen setting (touch: kappa 0.45, gamma 1), paired, degraded: DFL +0.015
  [+0.012, +0.019], M1 +0.013 [+0.010, +0.017], M2 +0.005 [-0.002, +0.012]; stress +0.002,
  -0.003, -0.010 [-0.018, -0.002]. Clean +0.053. False detections cost both variants about
  0.02 (no touch 0.621 -> 0.602, touch 0.635 -> 0.613 on DFL); v1 0.371 -> 0.350.
- **Reading.** About two thirds of the touch features' gain survives 5% false detections; on
  M2 it halves and is no longer certain; under heavy noise (stress, 15%) it is gone and slightly
  negative on M2. The trees learn to discount touches that a false ball fakes, but only so far.
- **Decision.** Keep the touch features. Re-check once the calibration gives the ATD's
  false-detection rate; above ~15% they would no longer pay. `models/v2.json` unchanged
  (trained without false detections, as the default simulator).

## 2026-10-29 — the data bundle on the machine: checks, diagnostics, the pre-kickoff stretch

- **Checks** (all behind `mtah/safe_run.py`, all PASS): the parser rebuilds the four published
  baselines byte for byte, against the committed files and against a fresh run of
  `starter/baseline.py`; two runs of the one command give byte-identical files; the official
  validator accepts the file; Smart Tagging's kickoff windows contain the published kickoffs.
- **Public window, alarm only** (COMPLIANCE_LOG.md): v2 spellF1 0.563, macroF1 0.847;
  `baseline_smooth50` 0.326 / 0.621, as published. No sign of misalignment or a label swap.
- **Unsupervised diagnostics** (`scripts/diagnose.py`, our predictions only): 292 possessions,
  duration p10/p50/p90 1.9 / 6.7 / 32.2 s, 1.4% under 1 s, 44.1% dead. Agreement with Smart
  Tagging team tags: 0.72 in attacking phases (the tagged team), 0.86 in defending phases (the
  other team). Stability with the ball removed on about 10% more frames: 1.1% of graded frames
  change, possessions x0.98, spellF1 of the perturbed run against the original 0.97.
- **Bug found:** of the graded stretch before the second-half kickoff -- dead in the ground
  truth (DATA.md) -- only 31% was predicted dead; the harness cannot show this (its halves start
  at kickoff). **Fix** (v2.1, `pipeline.before_kickoff`): graded frames before a half's Smart
  Tagging "Kick Off (Start)" window are dead; inside the window the decoder places the restart.
  After: 97.5% of the stretch dead, 284 possessions (8 spurious ones gone), the other
  diagnostics unchanged. No frame number of ours in the code; not scored on the public window.

## 2026-10-29 — calibrating the simulator to the ATD (STRICT; partly done)

- **Method** (`harness/calibrate.py`, `scripts/calibrate_noise.py`). Eleven noise statistics
  that need no labels, computed alike on the ATD's graded frames and on degraded DFL matches:
  ball missing share; mean gap; share of gaps of 1-2 frames (flicker); share of the missing time
  in gaps of 5 s or more (long losses); crowd within 3 m of the ball on missing vs seen frames
  (the ball put on the line between its sightings); ball speed before a gap vs overall; share of
  frame-to-frame moves over 1.5 m (jumps: false detections); ball and player jitter (robust
  spread around a centred mean); tracked players per frame; mean tracklet length. The ATD's go
  to `cache/aggregates/atd_noise.json`, unseen.
- **Simulator changes** (default behaviour as before in distribution; realisations differ from
  earlier entries, which are therefore not bit-reproducible with today's code): dropout noise is
  a mix of rough (exponential kernel, flickers at the threshold) and smooth (Gaussian kernel)
  noise, `ball_rough`; extra dropout while the ball is dead, `ball_weight_dead`; each part of the
  ball's noise draws from its own stream and false detections come from a fixed pool (common
  random numbers); the missing share is exact after false detections. Why: the first fit showed
  the old model could not reach the ATD's gap statistics -- the exponential-kernel noise
  crosses its threshold in bursts, which caps the mean gap at about 2 s whatever its
  correlation time -- and that fitting all ball knobs at once ended in poor local minima.
- **Fit.** Player knobs one at a time by bisection; the missing share directly; the five
  dropout knobs (smooth/rough mix, correlation time, dead, crowd and speed weights) jointly by
  bounded least squares; false detections and ball jitter by bisection on their own statistics.
- **Recovery check on DFL** (copies made with known knobs, fitted from another seed): all
  eleven statistics reproduced within 0.3%; person_missing, gap_mean_s, jitter, ball_false and
  ball_jitter recovered; the dropout knobs are not individually identifiable (another
  combination gives the same statistics), which is all the classifier sees.
- **On the ATD:** tracked players, tracklet length, player jitter, missing share, jumps and
  ball jitter reproduced within 10% (player jitter at the bottom of its range); the five dropout
  statistics (mean gap, flicker, long losses, crowd and speed lifts) are not, with no knob at a
  bound: the ATD's combination lies outside what the dropout model can make. The fitted knobs
  are at least as close to it as the defaults (least squares from the defaults).
- **Decision.** Use the partial calibration for now; ask the team, who may look at
  `cache/aggregates/atd_noise.json` beside `fitted_noise.json`, how the dropout differs, in
  generic terms. A cross-validation under this noise is running (v2; trained on mixed noise;
  without the touch features).

## 2026-10-29 — v2 under the partly calibrated noise (no change)

- **Setup.** Leave one match out as in plan steps 0-4, but every training and scored copy now
  comes from `degrade.calibrated()` (three copies per match) and the stress copy from
  `degrade.stressed()` of it. Variants: v2 as it stands; v2 trained on mixed copies; v2 without
  the touch features. Decoder settings chosen as before (mean over degraded and stress): v2
  kappa 0.3, gamma 1.5, unchanged.
- **Result** (spellF1 degraded [5-95%] / stress / clean):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | baseline_smooth50 | 0.169 / 0.103 / 0.309 | 0.174 / 0.106 / 0.329 | 0.166 / 0.098 / 0.343 |
  | v1 | 0.287 / 0.256 / 0.382 | 0.317 / 0.284 / 0.420 | 0.362 / 0.323 / 0.479 |
  | v2 | 0.677 [0.659, 0.694] / 0.573 / 0.706 | 0.716 [0.703, 0.728] / 0.611 / 0.738 | 0.714 [0.700, 0.728] / 0.627 / 0.713 |
  | v2, default noise (10-28) | 0.635 / 0.526 / 0.707 | 0.683 / 0.564 / 0.751 | 0.714 / 0.587 / 0.762 |

  macroF1 degraded 0.927. Predicted / true possessions on DFL 0.75 (0.69 under default noise).
  Paired against v2, degraded / stress:
  - without the touch features: DFL -0.029 [-0.037, -0.021] / +0.004, M1 -0.026
    [-0.034, -0.018] / +0.007, M2 -0.009 [-0.018, +0.000] / +0.005 (stress intervals contain 0);
  - trained on mixed copies: DFL -0.003 [-0.007, +0.002] / +0.015 [+0.009, +0.021], M1 -0.004 /
    +0.017, M2 -0.005 / +0.011 [+0.001, +0.020]; macroF1 +0.002 / +0.010.
- **Reading.** The calibrated copies are harder than the default ones for smooth50 and v1 and
  easier for v2 trained on them; a different test noise, so the rise over the 10-28 numbers is
  not a gain of the method. Which part of the noise does it is not shown under STRICT. v1 and v2
  fit parts of themselves on the training copies, hence their clean scores move too (v2 on M2
  clean -0.049: trained on noisier copies, it does a little worse on clean ones). The touch
  features now pay about twice what they did under the default noise; the question left open on
  10-29 (would they still pay at the ATD's false-detection rate) is answered yes. Mixed copies
  buy robustness under stress at no cost on the degraded copies, but no gain there either.
- **Decision.** Keep v2 as it is (touch features in; mixed copies not accepted by the decision
  rule, within noise on the degraded copies). Mixed copies stay the candidate if the
  ATD's dropout turns out far from what the simulator makes. `models/v2.json` not refitted yet:
  the refit on calibrated noise waits for the dropout calibration (the team's description), so
  that the model changes once.

## 2026-10-29 — a tracker that loses the ball in stoppages (simulator and calibration, STRICT)

- **Prompt.** The team compared the ATD's dropout statistics with the fitted copies' and shared
  them (COMPLIANCE_LOG.md). In generic terms, the ATD's dropout differs in kind: gaps are long,
  rarely a flicker, nearly all of the missing time in long gaps, with fewer players near the
  ball and a slower ball before it is lost than in the copies.
- **Off-game check of one explanation** (the tracker loses the ball while it is dead). On the
  DFL matches (dead share 0.41-0.52, 92-113 stoppages per match, mean 23-27 s), with the ball
  removed exactly while it is dead and nowhere else: missing 0.45, mean gap 25 s, flicker 0.004,
  long share 0.99, crowd ratio 0.66, speed ratio 0.95 (0.00 if lost a second after the ball goes
  dead: the harness's ball stands still while dead). The same pattern as the team's description.
- **Simulator** (`degrade.lost_in_stoppages`, off by default, default copies unchanged): in
  `ball_dead_lost` of the stoppages the ball is lost from `ball_dead_on_s` after it goes dead to
  `ball_dead_off_s` after the restart (spread `ball_dead_lag_sd_s` per stoppage), but for brief
  sightings at `ball_dead_seen_per_min` (mean `ball_dead_seen_s`); the dropout score takes
  `ball_missing` of the other frames. Every draw fixed per stoppage (common random numbers).
  `stressed()` doubles the delays' spread and the sightings.
- **Calibration.** New label-free statistics: the ball's speed just after it is found again
  (still if found before the restart); the players' mean speed (from positions, over 1 s) in
  the 2 s after vs before the start and the end of each gap of 5 s or more (they slow down after
  the ball goes dead and speed up after the restart). Off-game sensitivity, lost stoppages only:
  slowdown 0.73 / 0.73 / 0.76 / 0.81 / 0.87 for the ball lost -0.5 / 0 / +0.5 / +1 / +1.5 s after
  it goes dead; speedup 1.17 / 1.29 / 1.42 / 1.50 / 1.51 / 1.46 for found -1.5 ... +1 s after
  the restart. The five dropout knobs are fitted together by least squares on eight statistics
  (missing, gap, flicker, long share, ball speed before and after a gap, the two player ratios),
  from a tracker that loses every stoppage; a trial that leaves no gap to measure counts as far
  off.
- **Recovery check on DFL** (copies with known knobs, fitted from another seed): share of
  stoppages lost 0.85 -> 0.854, sightings 2.0 -> 1.94 per min, other dropout 0.010 -> 0.010, loss
  delay 0.40 -> 0.32 s, recovery delay -0.60 -> -0.60 s; all statistics within 5% but flicker
  (+11%, few gaps). Without the player ratios the loss delay came back at 1.09 s: the ball's
  speed alone cannot separate it from the sightings.
- **On the ATD:** reproduced within 10%: players (tracked, tracklets, jitter), missing share,
  flicker, long share, ball speed before a gap, players' slowdown, jumps. Not: mean gap, crowd
  ratio, ball speed after a gap, players' speed-up after a gap, ball jitter. Ball and player
  jitter at the bottom of their ranges: the ATD is smoother than DFL's raw tracking, more so now
  that the copies no longer keep the standing dead ball. The two statistics of the recovery
  disagree with each other, most likely because the harness's dead ball stands still where the
  ATD's is placed or carried; the fit is their compromise. When the ball is lost -- where a
  possession ends -- is reproduced; when it is found again is less certain.
- **Decision.** Use this calibration (it replaces the partial one). Cross-validation under it
  running (v2; trained on mixed copies, whose stress side widens the delays and the sightings).

## 2026-10-29 — v2 under the noise with lost stoppages (no change to the model; refit pending)

- **Setup.** As on 10-29 above (leave one match out, three calibrated copies per match, stress
  copy from `degrade.stressed()`), the copies now from the calibration with lost stoppages.
  Variants: v2 as it stands; v2 trained on mixed copies. Decoder settings chosen as before (mean
  over degraded and stress, DFL and M1): v2 kappa 0.3, gamma 1.0 (was 1.5); mixed kappa 0.6,
  gamma 1.5.
- **Result** (spellF1 degraded [5-95%] / stress / clean):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | baseline_smooth50 | 0.268 / 0.179 / 0.309 | 0.281 / 0.184 / 0.329 | 0.286 / 0.181 / 0.343 |
  | v1 | 0.393 / 0.275 / 0.378 | 0.430 / 0.294 / 0.416 | 0.483 / 0.321 / 0.473 |
  | v2 | 0.711 [0.694, 0.729] / 0.477 / 0.754 | 0.753 [0.739, 0.766] / 0.516 / 0.797 | 0.745 [0.731, 0.760] / 0.559 / 0.779 |
  | v2, mixed copies | 0.715 [0.696, 0.733] / 0.571 / 0.749 | 0.749 [0.736, 0.762] / 0.612 / 0.786 | 0.739 [0.724, 0.754] / 0.648 / 0.769 |

  macroF1 degraded 0.943 (both). Predicted / true possessions on DFL 0.77 (mixed 0.80).
  Paired, mixed minus v2, degraded / stress: DFL +0.004 [-0.001, +0.008] / +0.094 [+0.080,
  +0.106], M1 -0.003 [-0.009, +0.002] / +0.096, M2 -0.006 [-0.013, +0.002] / +0.089; macroF1
  -0.001 / +0.043.
- **Reading.** Again a different test noise, so the rise over the partly calibrated numbers is
  not a gain of the method. A ball lost in stoppages is easier for every method than one lost in
  crowds: the gaps fall where possessions end, not where they change. The stress copies are now
  much harder for v2 (0.477 on DFL, 0.573 under the partial calibration): they widen the delays
  and double the sightings in stoppages, the side of the dropout (when the ball is found again)
  that the calibration does not reproduce. Mixed copies cost nothing on the degraded copies and
  keep 0.09 of it under stress.
- **Decision.** By the decision rule, mixed copies are not accepted (within noise on the degraded
  copies). They hedge exactly the part of the dropout the calibration leaves uncertain, at no
  measured cost; raised with the team. v2 with or without the reception features (plan step 4b)
  is decided next on these same copies; `models/v2.json` is refitted once, after that.

## 2026-10-30 — plan step 4b: reception features (rejected)

- **Hypothesis.** Who gets the ball next decides a switch: a pass to a teammate keeps
  possession, an opponent who receives and holds the ball takes it. Features that look across
  the ball's flight to the next contact should cut wrong and missed switches.
- **Change** (`features.matrix`, group `train_v2.RECEPTION`). A contact: the ball seen within the
  touch radius of a team's player with no opponent nearer. Per frame: time since the last contact
  and until the next, the ball's travel between them; per team, whether it made the last / next
  contact and how firmly it held the ball in the 1 s before letting go / after receiving it
  (share of frames nearest). Unknown -> NaN. A synthetic pass and interception come out as
  intended (`tests/test_v2.py`).
- **Result.** Same copies as the entry above, same matches and seeds; fitting is
  deterministic, so v2 there is the paired reference. Decoder setting chosen as before: kappa
  0.45, gamma 1.0. Paired, with minus without, spellF1 degraded / stress:

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | degraded | +0.009 [+0.004, +0.015] | +0.002 [-0.004, +0.009] | -0.004 [-0.010, +0.002] |
  | stress | -0.001 [-0.010, +0.008] | -0.005 [-0.015, +0.006] | -0.017 [-0.026, -0.008] |

  macroF1 degraded +0.001 / +0.000 / -0.000, stress -0.002 / -0.002 / -0.003. At v2's own
  setting (kappa 0.3, gamma 1.0) the degraded spellF1 is 0.711 / 0.752 / 0.747 with and
  0.711 / 0.753 / 0.745 without: DFL's gain comes with the other decoder setting, not from the
  features.
- **Reading.** Most likely the touch features and the second stage over ±8 s already carry what
  the next contact says: under the calibrated noise the ball is lost in stoppages rather than in
  flight, so the contacts on either side of a flight are usually seen, and the second stage
  reads them. Where the ball is found again late (stress), the new features cost a little.
- **Decision.** Rejected (decision rule: beyond its interval on one label set only, within
  noise on the others, worse under stress on M2). The group stays in the code as an ablation;
  the accepted variant leaves it out (`drop=RECEPTION`). `models/v2.json` refitted: v2 as
  accepted, trained on three calibrated copies per match, kappa 0.3, gamma 1.0.

## 2026-10-30 — `models/v2.json` refitted on the calibrated copies

- **Change.** v2 as accepted (85 features, reception group left out), trained on all seven
  matches with three copies each from the calibration with lost stoppages; kappa 0.3, gamma 1.0
  (was 1.5). The previous model is kept outside the repo.
- **Checks:** tests pass; two runs of the one command give byte-identical files; the official
  validator accepts the file.
- **Unsupervised diagnostics** (our predictions only; before -> after): 284 -> 309 possessions,
  duration p10/p50/p90 2.0 / 7.1 / 29.5 s, 1.9% under 1 s, 42.3% dead; pre-kickoff stretch dead
  97.5% -> 97.2%. Agreement with Smart Tagging team tags 0.72 -> 0.69 (attacking phases), 0.86
  -> 0.84 (defending phases). Stability with the ball removed on about 10% more frames: 1.8% of
  graded frames change (was 1.1%), possessions x1.02, spellF1 against the original 0.94 (was
  0.97). No sign of a bug. The lower stability is what the training noise predicts: the
  perturbation removes the ball in play, which the calibrated copies rarely do. These numbers
  catch bugs; they do not rank the two models. Public window not looked at.

## 2026-10-30 — mixed training copies adopted (team decision, exception to the decision rule)

- **Decision by the team.** Train v2 on mixed copies (three calibrated, three drawn between the
  calibrated and the stress noise), although they do not pass the decision rule: on the
  calibrated copies they are within noise (DFL +0.004 [-0.001, +0.008], M1 -0.003, M2 -0.006),
  under stress +0.09 on every label set (entry "v2 under the noise with lost stoppages"). The
  stress copies widen the delays and the sightings in stoppages, the side of the dropout the
  calibration does not reproduce on the ATD, so the mixed copies hedge the known gap of the
  simulator at no measured cost.
- **Change.** `train_v2.VARIANTS`: `Variant(stacked=True, copies="mixed", drop=RECEPTION)`;
  `models/v2.json` refitted at the setting the cross-validation chose for it, kappa 0.6, gamma
  1.5. Pipeline version v2.2.
- **Checks:** tests pass; two runs of the one command byte-identical; the official validator
  accepts the file. After the commit (c1d1d6a), `scripts/check_reproducibility.sh`: two
  clean-room runs (fresh export, fresh virtualenv, empty cache) byte-identical to each other and
  to the file of the working copy.
- **Unsupervised diagnostics** (our predictions only; the plain refit of the entry above ->
  mixed): 309 -> 322 possessions, duration p10/p50/p90 1.9 / 6.2 / 29.1 s, under 1 s 1.9% ->
  3.7%, 41.5% dead; pre-kickoff stretch dead 97.6%. Agreement with Smart Tagging team tags 0.71
  (attacking phases), 0.85 (defending phases). Stability with the ball removed on about 10% more
  frames: 1.3% of graded frames change, possessions x1.03, spellF1 against the original 0.94.
  More and shorter possessions, as on the harness (predicted / true 0.80 against 0.77 on DFL).
  No sign of a bug. Public window not looked at.

## 2026-10-31 — other providers' matches: Metrica's sample games and SkillCorner (check, no change)

- **Question.** Does what was chosen on DFL hold where tracking, label semantics and noise come
  from elsewhere? Not a tuning set: nothing is fitted or chosen here (`harness/external.py`).
- **Data** (DATASETS.md). Metrica's sample games 1-3 (`harness/metrica.py`): labels from
  Metrica's events -- possession from the latest pass, carry, set piece, shot or recovery; dead
  from a stoppage (ball out, shot out or goal, a foul, a hand ball, an offside) to the next
  restart, unless play goes on in between (advantage). 315-377 possessions, 89-92 stoppages,
  dead share 0.29-0.41 per game; the ball as provided is missing on 5-6% of live frames and
  85-100% of dead ones. SkillCorner's 20 open matches (`harness/skillcorner.py`): broadcast
  tracking at 10 fps, interpolated to 25; labels from SkillCorner's automatic phases of play and
  possession; about 16 of 22 players in view, ball missing 0.40, dead share median 0.45, about
  318 possessions per match.
- **Methods.** `models/v1.json`, `models/v2.json` (v2.2, mixed copies) as exported, trained on
  DFL only; v2 without Smart Tagging; baseline_smooth50. Metrica as provided, degraded with the
  noise calibrated to the ATD (the 10-29 calibration, seeds 0-2) and with the stress noise;
  SkillCorner as provided only (its noise is real). Official metric per half, 5-95% bootstrap
  interval over halves.
- **Result** (spellF1; M1 labels, own labels in brackets for v2; macroF1 of v2):

  | | smooth50 | v1 | v2 | v2 minus v1 | v2 macroF1 |
  |---|---|---|---|---|---|
  | Metrica, provided | 0.441 | 0.626 | 0.840 [0.824, 0.856] (own 0.827) | +0.214 [+0.195, +0.233] | 0.929 |
  | Metrica, degraded | 0.335 | 0.498 | 0.805 [0.790, 0.820] (own 0.792) | +0.308 [+0.298, +0.317] | 0.931 |
  | Metrica, stress | 0.198 | 0.305 | 0.712 [0.680, 0.743] (own 0.701) | +0.407 [+0.373, +0.438] | 0.900 |
  | SkillCorner, provided | 0.264 | 0.392 | 0.378 [0.367, 0.389] (own 0.357) | -0.015 [-0.023, -0.006] | 0.799 |

  On own labels v2 minus v1 is +0.214 / +0.305 / +0.399 on Metrica and -0.010 [-0.019, -0.002]
  on SkillCorner; v1's macroF1 is 0.854 / 0.817 / 0.728 / 0.776.
- **Reading.** On Metrica the ranking and margins of DFL hold, under Metrica's own labels too:
  v2 is far ahead of v1 and of the baseline, provided, degraded and under stress. Absolute
  levels are not comparable with DFL's (other label semantics, tracking and noise).
  On SkillCorner v2 loses its lead on spellF1 (slightly behind v1) and keeps it on macroF1. The
  broadcast camera sees about 16 players and loses the ball where it leaves the picture; v2's
  per-team counts and nearest distances were learnt with every player tracked, as the ATD's
  tactical camera allows. SkillCorner is further from the ATD than DFL is, so this is a warning
  about v2's dependence on complete player tracking, not a reason to change it.
- **Note.** numpy's matmul raised overflow and invalid-value warnings in v1 on Metrica; the
  design matrices and their products are finite (checked), so these are numpy's spurious
  warnings on this machine.
- **Decision.** No change.

## 2026-10-31 — simulator: a dead ball carried to its restart spot (calibrations v3 and v3b rejected)

- **Hypothesis.** The 10-29 calibration does not reproduce when the ball is found again (mean
  gap, the ball's speed after a gap, the players' speed-up), most likely because DFL's dead ball
  stands still where it went dead and then jumps to the restart spot, where a real one is
  carried there and placed. With the ball carried in the copies, the fit should reproduce that
  side of the dropout too.
- **Change.** `degrade.place_dead_ball`, knobs `ball_dead_carry_mps` and `ball_dead_place_s`
  (0 m/s: off, the copies unchanged bit for bit): in each stoppage the ball stays where it went
  dead, then is carried in a straight line at that speed to where it is at the restart, to lie
  there still for an exponential time of that mean; a stoppage that ends a half is left as it
  is. Its own random stream, so common random numbers hold. Calibration: both knobs in the
  least-squares group (1-6 m/s, 0-6 s) and a new statistic, the share of seen, plausible ball
  frames slower than 0.5 m/s.
- **Recovery check on DFL** (`harness/recovery.py`; known knobs, carried at 3 m/s and placed
  for 3 s; this run also fitted a smoothing of the ball's track and gave the two knobs wider
  ranges): the dropout knobs of 10-29 come back about as before (share lost 0.85 -> 0.847, other
  dropout 0.010 -> 0.009, false detections 0.010 -> 0.008, delays 0.40 -> 0.50 s and -0.60 ->
  -0.82 s, sightings 2.0 -> 2.8 per min); carry and place do not (9.2 m/s and 8.8 s), because
  the statistics barely depend on them. Every dropout statistic within 6%. The smoothing went to
  its bound, unidentifiable, and was dropped before the runs below.
- **On the ATD** (STRICT; only pass / fail seen). Rule set before the first run: adopt only if
  the fit reproduces (within 10%) every statistic the 10-29 one does, plus at least one more.
  - v3, with the still share: reproduces the players (3), missing share, long share, speed
    before a gap, jumps; no longer flicker or the players' slowdown. Fails the rule.
  - v3b, run after v3 failed, on the guess that the still share pulled the fit off those two:
    the fit without it reproduces exactly the 10-29 set (players, missing, flicker, long share,
    speed before a gap, slowdown, jumps) and nothing more; mean gap, crowd ratio, speed after a
    gap, speed-up, still share and ball jitter stay out. Fails the rule.
  Ball and player jitter at the bottom of their ranges in both, as before.
- **Decision.** Both rejected. `degrade.CALIBRATION` stays "" (the 10-29 files, untouched;
  `scripts/calibrate_noise.py` now writes each calibration under its own tag, so none
  overwrites another). The placed dead ball stays in the simulator, off; the calibrator is
  unchanged. When the ball is found again remains what the calibration cannot pin down; the
  mixed training copies (10-30) are the hedge for it.

## 2026-10-31 — ablations for the technical document (no change)

- **Setup.** The cross-validation of 10-30 (leave one match out; the 10-29 calibration; three
  calibrated copies per match, a stress copy; trained on mixed copies), now with first-stage
  ablations (`train_v2.ABLATIONS`). Each is paired with the first stage as accepted (85
  features, team swap, mixed copies, no second stage), each decoded at its own setting, chosen
  as v2's (mean over degraded and stress, DFL and M1): first stage kappa 0.3, gamma 1.5; no
  look-ahead 0.2 / 1.5; no team swap 0.3 / 1.5; trained clean 0.2 / 1.0. A second stage per
  ablation would cost seven more fits per fold each. "No look-ahead" makes every feature window
  end at the frame (what only the future tells is unknown); the decoder still sees the whole
  match.
- **Reference** (first stage, spellF1 clean / degraded [5-95%] / stress): DFL 0.725 / 0.689
  [0.667, 0.709] / 0.542; M1 0.767 / 0.727 / 0.578; M2 0.772 / 0.732 / 0.613. v2 as accepted
  reproduces the 10-30 numbers exactly (DFL 0.749 / 0.715 [0.696, 0.733] / 0.571).
- **Result** (paired spellF1 differences, degraded / stress):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | no look-ahead | -0.017 [-0.021, -0.012] / -0.031 | -0.021 [-0.026, -0.016] / -0.031 | -0.036 [-0.045, -0.028] / -0.028 |
  | no team swap | -0.006 [-0.012, +0.000] / -0.011 | -0.010 [-0.016, -0.004] / -0.008 | -0.011 [-0.016, -0.006] / -0.010 |
  | trained clean | -0.106 [-0.128, -0.085] / -0.268 | -0.120 [-0.141, -0.101] / -0.291 | -0.151 [-0.175, -0.131] / -0.325 |
  | second stage (v2 minus first stage) | +0.026 [+0.020, +0.032] / +0.029 | +0.022 [+0.017, +0.028] / +0.034 | +0.007 [+0.001, +0.013] / +0.035 |

  macroF1 degraded: no look-ahead -0.006 to -0.008, no team swap -0.002, trained clean -0.078
  to -0.081, second stage +0.005 to +0.007. On the clean copies the model trained clean is the
  best (DFL 0.765 against 0.725) and predicts 0.93 times as many possessions as there are (0.75
  for the first stage).
- **Reading.** Training on degraded copies is by far the largest single choice: a model fitted
  to clean tracking wins on clean tracking and collapses under the ATD's noise, most under
  stress. Looking ahead in the feature windows is worth 0.02-0.04 (when a team gains control
  shows in what follows), the second stage another 0.01-0.03 (more under stress), team swapping
  about 0.01.
- **Refit.** `models/v2.json`, refitted by the same run, differs from the committed one only in
  the sign of 10 zeros in first-stage trees (-0.0 against 0.0, equal as numbers), presumably
  from the feature code's refactoring for the causal ablation; the submission (v2.2) is byte for
  byte the same.
- **Decision.** No change; the results go into docs/TECHNICAL.md.

## 2026-10-31 — Smart Tagging fused into the decoder (implemented; rejected, weight 0)

- **Hypothesis.** Smart Tagging's attacking and defending phases name the team in possession;
  as a second observation in the decoder they should correct team errors where the tracking is
  unsure.
- **Method** (`mtah/tagging.py`, `pipeline.tagging_evidence`). Phases -> one vote per frame, H,
  A or none: attacking codes vote for their Team, defending codes for the other; the first and
  last 1 s of a phase and frames where phases disagree get none, as do N/A and set pieces. Team
  tag values map onto home / away by the words they share with the metadata's club names
  (`make_submission` requires the mapping to be one to one). P(vote | H, A, D) is estimated on
  the match itself by EM, the classifier's P(H, A, D) standing in for the unknown state: no
  labels, symmetric in the two teams. weight x log P(vote | state) is added to the decoder's
  tempered log emissions. At weight 0 the file is v2.2's, byte for byte (checked).
- **Setting the weight** (`harness/fuse.py`; rule fixed before the runs: the weight with the
  best mean spellF1 over both kinds of simulated tagger, degraded and stress copies, DFL and
  M1). Smart Tagging exists only for BAR-ATH, so a phase tagger is simulated on each held-out
  DFL match, on the saved out-of-fold outputs: from DFL's labels (errors independent of the
  tracking, the best case) or from v2's own path (an echo of the tracking, the worst case).
  - *First run*, a hand-set tagger (a phase per possession of 2 s or more, 20% missed, ends
    shifted by about 1 s, team flipped in 5 / 15 / 30% of phases): weight 0.2 (on DFL,
    independent +0.003 to +0.014 degraded, echo -0.001 to -0.013). Applied, it changed 1.8% of
    BAR-ATH's graded frames and cut its possessions from 322 to 294 (-9%), where on the harness
    the same weight cut them by at most 3%: the hand-set tagger was less coarse than Smart
    Tagging. The team decided to calibrate the tagger before choosing; the file with weight 0.2
    was never a candidate.
  - *Calibration* (`harness/tagger.py`, `scripts/calibrate_tagger.py`, STRICT): five label-free
    statistics of the votes on the graded frames after each kickoff -- share of frames voted,
    mean length of a run of one vote, share of runs under 5 s, team changes per minute,
    agreement with the tracking-only path -- measured on BAR-ATH (written, unseen) and fitted on
    DFL by a grid search over the tagger's knobs (shortest possession with a phase, merge gap,
    miss rate, flip rate), every draw made per possession (common random numbers). That family
    reproduced only the team changes and the agreement. With a fifth knob (each phase end moved
    out, or in, by `extend_s`: a phase may spill past its possession or cover part of it) and
    wider grids, all five within 10%, every knob inside its grid or at 0 / inf; the merge gap
    came out long (phases of one team merge across what lies between them).
  - *With the calibrated tagger*, paired spellF1 at weight 0.1 / 0.2, degraded (stress):

    | | DFL | M1 | M2 |
    |---|---|---|---|
    | independent | +0.003 [+0.001, +0.006] / +0.002 [-0.003, +0.006] (+0.004 / +0.005) | +0.004 / +0.003 (+0.004 / +0.005) | +0.003 / +0.003 (+0.010 / +0.009) |
    | echo | -0.007 [-0.008, -0.006] / -0.013 [-0.015, -0.011] (-0.005 / -0.012) | -0.005 / -0.009 (-0.005 / -0.010) | -0.002 / -0.003 (-0.002 / -0.007) |

    From 0.35 up both kinds lose (at 1.0: about -0.06 independent, -0.10 echo). Predicted /
    true possessions (DFL labels, degraded, independent) at weight 0 / 0.2 / 1: 0.796 / 0.782 /
    0.730. Chosen: weight 0.
- **Reading.** A tagger as coarse as Smart Tagging, even with errors independent of the
  tracking, adds at most +0.005, within noise; one that echoes the tracking costs up to -0.013
  at 0.2. Its long phases carry little the second stage over ±8 s does not already have. The
  calibrated tagger still lowers the harness's possession count less than Smart Tagging lowers
  BAR-ATH's (-1.8% against -9% at 0.2): Smart Tagging remains coarser, or more trusted by the
  EM, than any tagger simulated -- if anything, a further reason against it.
- **Decision.** Rejected: `pipeline.TAGGING_WEIGHT = 0`, the submission stays v2.2. The fusion,
  the simulated tagger and its calibration stay in the code. Smart Tagging is used only for the
  kickoff windows. For the record, BAR-ATH diagnostics of the 0.2 run (our predictions only):
  agreement with Smart Tagging 0.714 -> 0.758 in attacking phases, 0.846 -> 0.874 in defending
  phases; stability 1.4% of graded frames changed, spellF1 against the original 0.933; no sign
  of a bug. Public window not looked at.

## 2026-10-31 — where v2 loses possessions (`harness/segments.py`); short possessions weighed more in training (rejected)

- **Diagnosis** (v2.2 as submitted, from the saved out-of-fold outputs: all seven matches, the
  three calibrated degraded copies, kappa 0.6, gamma 1.5; the 10-29 diagnosis used four folds,
  one copy and the noise before calibration). What became of each true possession:

  | true length (DFL) | share | found | merged | split | missed | shifted |
  |---|---|---|---|---|---|---|
  | < 2 s | 30% | 0.33 (10-29: 0.15) | 0.16 (0.23) | 0.00 | 0.44 (0.53) | 0.07 |
  | 2-4 s | 23% | 0.63 (0.47) | 0.15 (0.21) | 0.00 | 0.19 (0.23) | 0.03 |
  | 4-10 s | 26% | 0.79 (0.73) | 0.11 (0.17) | 0.01 | 0.07 (0.04) | 0.01 |
  | >= 10 s | 21% | 0.95 (0.95) | 0.03 (0.04) | 0.01 | 0.00 | 0.00 |

  M1 / M2: under 2 s found 0.44 / 0.51, missed 0.42 / 0.41; predicted / true possessions 0.80
  (DFL), 0.90 (M1), 1.08 (M2). Two new splits:
  - *By ball visibility.* 96% of the possessions under 2 s (98% of 2-4 s) have the degraded
    copy's ball on at least half their frames, and those are found no more often (0.32, missed
    0.44) than the few without (0.46 / 0.33). The short possessions are lost with the ball in
    view: the reading of the ball near the players fails, not the ball's detection.
  - *Merges.* Of the segments separating two possessions of one team, stoppages are reproduced
    almost always (4-10 s 0.98, longer 1.00; under 4 s, n 24, 0.67-0.78); spells of the other
    team under 2 s in 0.56 (DFL; 2-4 s 0.83). A merge is a missed short possession of the
    other team, not a missed stoppage.
- **Hypothesis.** A short possession is a handful of training frames against minutes of long
  ones, so the trees' fit trades the short ones away. Weighing frames by the inverse length of
  their possession should make them read short possessions better.
- **Change** (`semantics.possession_weights`, `train_v2.Variant.short_s`; tested). In both
  classifier stages (not the change detector) and their class prior, every frame of a
  possession shorter than 4 s weighs 4 s / its length, a possession under 1 s as one of 1 s
  (shorter ones are mostly touches without control). Everything else as v2. Decoder setting
  chosen by the usual rule (DFL and M1, degraded and stress): kappa 0.45, gamma 1.
- **Result** (spellF1 degraded [5-95%] / stress; paired with v2):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | v2 (kappa 0.6, gamma 1.5) | 0.715 [0.696, 0.733] / 0.571 | 0.749 [0.736, 0.762] / 0.612 | 0.739 [0.724, 0.754] / 0.648 |
  | short weighted (0.45, 1) | 0.730 [0.717, 0.745] / 0.575 | 0.758 [0.748, 0.768] / 0.614 | 0.723 [0.708, 0.737] / 0.633 |
  | paired, degraded | +0.016 [+0.009, +0.024] | +0.009 [+0.002, +0.016] | **-0.016 [-0.022, -0.010]** |
  | paired, stress | +0.004 [-0.003, +0.012] | +0.002 [-0.007, +0.012] | **-0.015 [-0.025, -0.006]** |

  macroF1 within -0.003. Possessions under 4 s found: 0.458 -> 0.528 (DFL), 0.553 -> 0.636
  (M1), 0.628 -> 0.701 (M2); 4 s and longer the same. Predicted / true possessions 0.80 ->
  0.87 (DFL), 0.90 -> 0.99 (M1), 1.08 -> 1.18 (M2). No setting of it gains on DFL and M1
  without losing on M2 or under stress (at kappa 0.3, gamma 1: DFL +0.012 / -0.002, M2
  -0.002 / -0.003).
- **Reading.** The weighting does what it was meant to: more short possessions found, the
  long ones unharmed. But the short possessions it adds are, under a 2 s control time, touches
  without control, and M2 counts them as errors -- the label-semantics trade-off of 10-27
  again: a gain on DFL's notion of possession is a bet against the strictest reading of
  Metrica's. Which control time the ground truth follows is still organiser question 6.
- **Decision.** Rejected by the criterion fixed on 10-27 (judged on DFL and M1, must not lose
  on M2). The weights stay in the code (`short_s`, off), out of `VARIANTS`; `harness/segments.py`
  stays as a diagnostic. `models/v2.json` and the submission unchanged. Public window not
  looked at.
- **Reproduction.** After items 3-7 (a5ae200), `scripts/check_reproducibility.sh`: two
  clean-room runs (fresh export, fresh virtualenv, empty cache) byte-identical to each other and
  to the file of the working copy (v2.2). Again after the team name was set to `kicker`
  (c4f8ce7): both runs write `kicker.csv`, byte-identical to each other and to the working
  copy's, which is byte-identical to the earlier `TEAM_NAME.csv`.

## 2026-11-01 — M2's short possessions weighed more in training (accepted, v2.3)

- **Hypothesis.** The label sets disagree only on a short spell of one team between two of the
  other; every other possession is one under DFL, M1 and M2. The rejected weighting of 10-31
  measured lengths on DFL's possessions and so pushed the model towards the contested spells.
  Measured on M2's possessions, it rewards only possessions every label set counts.
- **Change.** `Variant.short_on = "M2"` (`train_v2._weights`): frames of an M2 possession under
  4 s weigh 4 s / its length (at most 4); target still DFL. Folded into `"gbm"`; decoder
  setting by the usual rule: kappa 0.45, gamma 1. `models/v2.json` refitted, pipeline v2.3.
- **Result** (spellF1 degraded [5-95%] / stress; paired with v2.2):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | v2.3 | 0.721 [0.706, 0.736] / 0.570 | 0.757 [0.746, 0.767] / 0.612 | 0.749 [0.738, 0.760] / 0.648 |
  | paired, degraded | +0.006 [+0.002, +0.010] | +0.007 [+0.003, +0.011] | +0.010 [+0.004, +0.016] |
  | paired, stress | -0.000 [-0.008, +0.008] | +0.000 [-0.009, +0.009] | -0.000 [-0.011, +0.010] |

  macroF1 -0.001. Possessions under 4 s found 0.458 -> 0.489 (DFL), 0.553 -> 0.598 (M1),
  0.628 -> 0.695 (M2); predicted / true 0.80 -> 0.82 / 0.90 -> 0.93 / 1.08 -> 1.12. v2.3 minus
  v1: +0.328 (DFL, degraded). The export run reproduced the experiment's scores exactly.
- **Reading.** Smaller on DFL than DFL's weighting (+0.016), which owed much of its gain to the
  contested spells, but no label set loses. M2 now also shapes the training weights (training
  folds only), so it is less of an independent check than before.
- **Decision.** Accepted (decision rule; the 10-27 rule). BAR-ATH (our predictions only; in
  brackets v2.2): 340 possessions (322), durations p10/p50/p90 1.7 / 5.8 / 27.8 s, under 1 s
  4.1% (3.7%), D 41.2%, no X on graded frames; pre-kickoff stretch dead 97.6% (97.6%).
  Agreement with Smart Tagging team tags 0.70 (0.71) in attacking phases, 0.84 (0.85) in
  defending phases. Stability with the ball removed on about 10% more frames: 1.5% (1.3%) of
  graded frames change, possessions x1.00 (x1.03), spellF1 against the original 0.944 (0.94).
  More and shorter possessions, as on the harness. No sign of a bug; the validator accepts the
  file. Smart Tagging weight left at 0 (chosen on v2.2's outputs). Public window not looked at.
- **Reproduction.** After the commit (d1e50b6), `scripts/check_reproducibility.sh`: two
  clean-room runs byte-identical to each other and to the working copy's `kicker.csv` (v2.3).
  Again on 2026-11-15 at b004008 (the last code change, fallbacks dropped): both runs
  byte-identical to each other and to the working copy's `kicker.csv`, the validator accepts
  it; 25 s and about 1.8 GB per run.

## 2026-11-15 — pre-registered: what an answer to organiser question 6 changes

Written before question 6 is posted and before any answer, and before the public-window alarm
of the same day, so that neither can shape it.

- **Question.** Do touches without control ever form a spell in the ground truth
  (`docs/ORGANISER_QUESTIONS.md`, 6)? v2.3's decoder setting (kappa 0.45, gamma 1) is chosen on
  DFL and M1 (the 10-27 rule); M2 is the strictest reading of Metrica's rule.
- **What the off-game outputs say** (v2.3's out-of-fold scores of 11-01, `external/idsse/oof/
  scores.pkl`; no fitting; paired against kappa 0.45, gamma 1, degraded [5-95%] / stress):

  | criterion | best setting | DFL | M1 | M2 |
  |---|---|---|---|---|
  | DFL + M1 (in force) | kappa 0.45, gamma 1 | -- | -- | -- |
  | DFL only | kappa 0.6, gamma 1.5 | +0.002 [-0.002, +0.006] / -0.001 | -0.004 / -0.005 | -0.015 / -0.016 |
  | M2, or M1 + M2 | kappa 0.2, gamma 1 | -0.017 [-0.021, -0.013] / -0.012 | -0.005 / -0.004 | +0.013 [+0.008, +0.017] / +0.015 |

  macroF1 within +-0.001 everywhere.
- **Rule.**
  - The answer says every spell of the ground truth is a controlled possession (a touch without
    control never makes one): the decoder setting is chosen on M1 and M2 -> kappa 0.2, gamma 1.
    Only `kappa` in `models/v2.json` changes; nothing is refitted. Then the BAR-ATH diagnostics,
    the validator and two clean-room reproductions; pipeline v2.4.
  - The answer says such touches can form spells, or it is unclear: v2.3 stays (DFL alone gains
    within noise; the decision rule keeps the simpler, unchanged setting).
  - No answer by 2026-11-16 12:00 (CET): v2.3 stays. A later answer goes into the technical
    document only.
- **Not affected by the public window.** The alarm run of today may stop a submission (a bug),
  it does not choose between these settings.

## 2026-11-15 — pre-registered: Smart Tagging's weight on v2.3

Written before the runs. The weight 0 of 10-31 was set on v2.2's outputs; the tagger is now
recalibrated on v2.3 (`scripts/calibrate_tagger.py`, STRICT; it overwrites the v2.2 calibration,
whose out-of-fold outputs were already replaced by v2.3's on 11-01) and `harness.fuse` rerun on
v2.3's out-of-fold outputs. `harness.fuse` picks the weight by the 10-31 rule; a weight other
than 0 is adopted only if, paired against 0, it gains beyond its interval on DFL and M1
(degraded) with the independent tagger and loses beyond its interval nowhere (echo tagger,
stress copies, M2). Otherwise 0 stays and the submission is unchanged.

## 2026-11-15 — Smart Tagging on v2.3: weight 0 confirmed; where its effect on BAR-ATH comes from (no change)

- **Recalibration** (`scripts/calibrate_tagger.py` on v2.3's out-of-fold outputs and v2.3's
  tracking-only path on BAR-ATH; STRICT): every knob inside its grid, all five vote statistics
  reproduced within 10%. 342 s.
- **Weight** (`harness.fuse`, v2.3's outputs, the calibrated tagger; paired against 0, degraded
  [5-95%] / stress):

  | | weight 0.1 | weight 0.2 |
  |---|---|---|
  | DFL independent | +0.001 [-0.000, +0.003] / +0.003 | -0.001 [-0.005, +0.003] / +0.009 |
  | M1 independent | +0.004 [+0.002, +0.006] / +0.005 | +0.002 [-0.001, +0.007] / +0.009 |
  | M2 independent | +0.004 [+0.002, +0.006] / +0.010 | +0.003 [-0.001, +0.006] / +0.010 |
  | DFL echo | -0.004 [-0.007, -0.002] / -0.002 | -0.015 [-0.017, -0.012] / -0.011 |
  | M1 echo | -0.003 [-0.006, -0.000] / -0.001 | -0.013 [-0.015, -0.011] / -0.009 |
  | M2 echo | -0.001 [-0.004, +0.003] / -0.004 | -0.007 [-0.011, -0.004] / -0.011 |

  From 0.35 up every kind loses. The 10-31 rule picks 0.1 (best mean over both kinds); by the
  rule pre-registered above it is not adopted: on DFL the independent tagger's gain is within
  its interval and the echo tagger loses beyond it. **Weight 0 stays; submission unchanged.**
- **What the fusion removes** (`tagging.fate_of_possessions`; per length of the tracking-only
  possessions, the share with no one-to-one match at IoU 0.5 after fusing at weight 0.2, and of
  those the share mostly under votes for the other team):

  | | < 2 s | 2-4 s | 4-10 s | >= 10 s |
  |---|---|---|---|---|
  | BAR-ATH, Smart Tagging (`scripts/diagnose.py`; 48 / 71 / 107 / 114 possessions) | 0.54, 0.62 | 0.45, 0.63 | 0.14, 0.67 | 0.05, 0.00 |
  | harness, independent tagger (degraded copies) | 0.12, 0.72 | 0.07, 0.61 | 0.04, 0.45 | 0.01, 0.28 |
  | harness, echo tagger | 0.13, 0.73 | 0.07, 0.57 | 0.04, 0.42 | 0.01, 0.17 |

  On BAR-ATH the fusion at 0.2 takes the possessions from 340 to 286 (x0.84; v2.2: x0.91); on the
  harness x0.98. Of the possessions the independent tagger removes, the share that were right
  (matched to a true possession): < 2 s 0.43 / 0.36 / 0.27 (DFL / M1 / M2), 2-4 s 0.39 / 0.31 /
  0.26, 4-10 s 0.69 / 0.60 / 0.52.
- **Reading.** The simulated tagger shows Smart Tagging's five vote statistics, yet Smart Tagging
  overrides four to six times as many of our short possessions as it does. Those statistics are
  frame counts dominated by long phases; they do not constrain what a tagger does where a
  possession is short. Two readings remain, with opposite consequences: Smart Tagging's phases
  run across short possessions more than a tagger built from possessions does (then fusing
  would delete right possessions), or our short possessions on BAR-ATH are wrong more often than
  on the harness (then it would delete wrong ones). Only labels could tell them apart. The share
  of our possessions under 4 s on BAR-ATH (35%) is below the harness's (44% of v2's predicted
  possessions), with a similar number of possessions per half (170 against 180), so v2 does not
  make more short possessions there. On the harness the removed short possessions are mostly
  wrong, and still the net effect is within noise.
- **Decision.** No change. The breakdown stays in `scripts/diagnose.py` and `harness/fuse.py`.
  Public window not looked at.

## 2026-11-15 — team name XHS, repository metrica-hack (no change to the file)

The team renamed itself `XHS` and the repository `github.com/ouyang1030/metrica-hack`.
`config.TEAM_NAME = "XHS"`; the technical document is `docs/XHS.md` (SUBMISSION.md: both
deliverables named after the team). `make_submission` writes `evaluation/results/XHS.csv`,
byte-identical to the earlier `kicker.csv` (`cmp`), which was then removed so that no file of
the old name can be uploaded. Earlier entries keep the names they were written with.

## 2026-11-15 — v2.4: the temperature chosen on Metrica's rule with 1-3 s of control (team decision)

- **Why.** CHALLENGE.md defines a change of possession as the other team gaining *controlled*
  possession -- a touch without control, a deflection that does not intercept, an aerial duel
  won without control do not count -- and suggests "a player keeps the ball for a few seconds"
  or "more than two consecutive passes" as proxies for control. DFL's labels count touches
  that this rule does not; M1-M3 (harness/semantics.py, control times of 1-3 s) are the readings
  of it. On 10-27 the temperature was chosen on DFL and M1 with M2 as a check. Read again
  today (and the official repository has no commit after the one copied, f8f2629), the text
  argues for the stricter readings. Not prompted by the public window: no setting was scored
  on it.
- **Analysis** (`harness/control.py`: v2.3's out-of-fold outputs, no fitting, every decoder
  setting of the grid and kappa 0.1 / 0.15; spellF1 degraded / stress, paired against kappa
  0.45, gamma 1, i.e. v2.3; the module prints them against the exported setting):

  | | DFL | M1 | M2 | M3 | M4 |
  |---|---|---|---|---|---|
  | kappa 0.1 | -0.057 / -0.044 | -0.040 / -0.035 | +0.009 / -0.003 | +0.037 / +0.037 | +0.044 / +0.067 |
  | kappa 0.15 | -0.031 / -0.024 | -0.016 / -0.015 | +0.013 / +0.010 | +0.030 / +0.037 | +0.033 / +0.056 |
  | **kappa 0.2** | -0.017 / -0.012 | -0.005 / -0.004 | +0.013 / +0.015 | +0.023 / +0.034 | +0.023 / +0.043 |
  | kappa 0.3 | -0.005 / -0.004 | -0.001 / +0.001 | +0.008 / +0.009 | +0.011 / +0.021 | +0.011 / +0.023 |
  | kappa 0.6, gamma 1.5 | +0.002 / -0.001 | -0.004 / -0.005 | -0.015 / -0.016 | -0.016 / -0.023 | -0.017 / -0.023 |

  (all with gamma 1 unless stated). The temperature sets the control time the decoder behaves as
  if it assumed: 0.45 for DFL and 1 s, 0.15-0.2 for 2 s, 0.1 or below for 3-4 s. Best mean over
  degraded and stress of M1-M3, over the whole grid and the two lower temperatures: kappa 0.2,
  gamma 1. macroF1 within +-0.001 of v2.3.
- **Decision (team).** kappa 0.2, gamma 1: `"kappa"` in `models/v2.json` changed, nothing
  refitted; `train_v2.CHOICE = ("M1", "M2", "M3")` (M3 added to `LABEL_SETS`) so that
  `harness.train_v2` exports the same setting; pipeline v2.4. All the differences are below
  the ~0.04 full-match band: an expected-value choice on the semantics, not a measured gain on
  BAR-ATH.
- **v2.4 on the harness** (spellF1 degraded [5-95%] / stress; macroF1 degraded 0.941):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | v2.4 | 0.704 [0.687, 0.720] / 0.558 | 0.751 [0.740, 0.762] / 0.608 | 0.761 [0.750, 0.773] / 0.663 |
  | minus v1 | +0.311 | +0.321 | +0.279 |
  | minus smooth50 | +0.436 | +0.470 | +0.475 |

  Possessions under 4 s found 0.42 / 0.53 / 0.63 (v2.3 0.49 / 0.60 / 0.70), 4 s and longer
  0.85 / 0.85 / 0.87 (0.86 / 0.86 / 0.87); predicted / true possessions 0.75 / 0.85 / 1.02
  (v2.3 0.82 / 0.93 / 1.12).
- **BAR-ATH** (our predictions only; v2.3 in brackets): 290 possessions (340), durations
  p10/p50/p90 2.2 / 7.1 / 30.6 s, under 1 s 0.7% (4.1%), D 41.1%, no X on graded frames;
  pre-kickoff stretch dead 97.6%; agreement with Smart Tagging 0.71 / 0.85 (0.70 / 0.84);
  stability with the ball removed on about 10% more frames: 1.4% of graded frames change,
  possessions x1.02, spellF1 against the original 0.954 (0.944). Validator passes. Public
  window not looked at.
- **Question 6 rule, restated.** v2.4 is the setting the pre-registered rule names for an answer
  that every spell is a controlled possession. If the organisers answer that touches without
  control can form spells, `kappa` goes back to 0.45 (v2.3, chosen on DFL and M1).
- **Reproduction.** After the commit (81ec264), `scripts/check_reproducibility.sh`: two
  clean-room runs (fresh export, fresh virtualenv, empty cache) byte-identical to each other and
  to the working copy's `XHS.csv` (v2.4); the validator accepts it; about 24 s and 1.8 GB each.

## 2026-11-15 — simulator: a ball the source already lacks stopped the in-play dropout (fixed)

- **Found by** the sensitivity sweep: more in-play dropout on Metrica's copies changed nothing.
  Metrica's sample data lacks the ball on some in-play frames; there the ball's speed is NaN,
  so the dropout score was NaN, `np.quantile` over the candidates NaN, and no in-play frame was
  ever dropped. Metrica's degraded and stress copies (10-31) therefore had lost stoppages, false
  detections and jitter but no in-play dropout: easier than intended.
- **Fix** (`degrade.drop_ball`): frames without a score leave the candidates (a ball already
  missing cannot be lost again). DFL's matches have ball position, speed and height on every
  in-play frame (checked on all seven), so their copies, the cross-validation, the models and
  the submission are unchanged. Test on a synthetic match with a source gap.

## 2026-11-15 — Smart Tagging on v2.4 (kappa 0.2): weight 0 again (no change)

The pre-registered rule above, rerun at v2.4's decoder setting: tagger recalibrated on v2.4's
tracking-only path and the out-of-fold outputs decoded at kappa 0.2 (every knob inside its grid,
all five statistics within 10%); `harness.fuse` now also scores M3 and picks on M1-M3.

- Paired against weight 0, degraded [5-95%] / stress, weight 0.1: independent DFL -0.007
  [-0.013, -0.001] / +0.002, M1 -0.005 / +0.001, M2 -0.002 / +0.005, M3 +0.003 / +0.015; echo
  -0.018 / -0.010, -0.020 / -0.010, -0.015 / -0.010, -0.007 / -0.008. Weight 0.2: independent
  -0.025 / -0.020 (DFL) to -0.000 / +0.001 (M3); echo -0.048 to -0.019. Every weight from 0.35
  up loses at least 0.02 everywhere. Best mean over both kinds and M1-M3: **0**.
- Fate at weight 0.2 (share of v2.4's possessions removed; BAR-ATH from `scripts/diagnose.py`
  in the v2.4 entry): < 2 s 0.29 on the harness (independent tagger) against 0.76 on BAR-ATH,
  2-4 s 0.18 against 0.67, 4-10 s 0.12 against 0.36, >= 10 s 0.04 against 0.16; possessions
  x0.93 on the harness, x0.75 on BAR-ATH. Of the short ones the harness tagger removes, 0.37-0.61
  were right (M3 to DFL). With the lower temperature the fusion removes more everywhere; Smart
  Tagging still removes two to three times what the simulated tagger does.
- **Decision.** Weight 0; submission unchanged (v2.4).

## 2026-11-15 — sensitivity to what the calibration could not pin down (`harness/sensitivity.py`; no change)

- **Question.** The calibration reproduces when the ATD loses the ball, not when it finds it
  again, how crowded the ball is when lost, or its jitter; and SkillCorner (10-31) suggested a
  dependence on complete player tracking. How much does v2 lose if the ATD differs from the
  simulator there?
- **Method.** The exported models (v2.4, v1) and smooth50 on Metrica's sample games (never
  trained on), degraded with the calibrated noise with one knob moved by an offset (STRICT: the
  calibrated values are never printed), seeds 0-2, paired with the calibrated noise; 6 halves,
  5-95% bootstrap. Run with the simulator fix of today (an earlier run, before it, had no
  in-play dropout on Metrica and is discarded; its in-play and crowd rows came out exactly 0,
  which is how the bug was found). A first version moved the crowd weight alone; that weight
  only shapes the in-play dropout, so it was replaced by more in-play dropout in crowds.
- **Result** (spellF1 change of v2.4 / v1, M1 labels; own labels within 0.002 of these):

  | the ATD's noise differs by | v2.4 | v1 |
  |---|---|---|
  | ball found again 0.5 / 1 s earlier after a stoppage | +0.000 / -0.001 | +0.000 / +0.002 |
  | ball found again 0.5 / 1 / 2 / 4 s later | -0.003 / -0.010 / -0.030 / -0.067 | -0.002 / -0.005 / -0.013 / -0.032 |
  | delays spread more by 0.5 / 1 / 2 s | -0.002 / -0.012 / -0.028 | -0.005 / -0.006 / -0.009 |
  | 1 / 2 / 4 more brief sightings per minute of a lost stoppage | -0.000 / -0.000 / -0.000 | 0.000 |
  | ball lost in play, in crowds, on 2 / 5 / 10% more frames | -0.014 / -0.032 / -0.054 | -0.006 / -0.013 / -0.026 |
  | ball jitter +0.3 m | -0.020 | -0.010 |
  | players untracked +5 / +10 / +20% of their frames | -0.004 / -0.001 / -0.025 | -0.004 / -0.017 / -0.033 |

  v2.4 at the calibrated noise 0.802 (M1; own labels 0.783), v1 0.497, smooth50 0.334; v2.4's
  lead over v1 stays between +0.27 and +0.32 in every row.
- **Reading.** v2 leans on a well-located ball more than v1 does and loses about twice as much
  when the ball is found late, lost in play or jitters -- never enough to lose its lead. A ball
  found again 1 s late costs 0.01, 2 s 0.03: that side of the calibration matters at the scale
  of seconds, and the mixed training copies are the hedge for it. Untracked players cost v2 no
  more than v1 (less at +10%), so SkillCorner's result is better explained by what a broadcast
  camera loses (players far from the ball, the ball out of the picture) than by missing players
  as such.

## 2026-11-15 — v2.4 on other providers' matches, with the simulator fix (check, no change)

`harness.external` rerun (v2.4 as exported; Metrica's degraded and stress copies now with the
in-play dropout they lacked). spellF1 [5-95%], M1 labels (own labels in brackets for v2):

| | smooth50 | v1 | v2.4 | v2.4 minus v1 | v2.4 macroF1 |
|---|---|---|---|---|---|
| Metrica, provided | 0.441 | 0.626 | 0.834 [0.818, 0.851] (0.814) | +0.208 [+0.185, +0.230] | 0.926 |
| Metrica, degraded | 0.334 | 0.496 | 0.802 [0.781, 0.823] (0.783) | +0.305 [+0.292, +0.314] | 0.928 |
| Metrica, stress | 0.190 | 0.312 | 0.674 [0.646, 0.703] (0.654) | +0.362 [+0.327, +0.400] | 0.888 |
| SkillCorner, provided | 0.264 | 0.392 | 0.390 [0.380, 0.400] (0.366) | -0.003 [-0.011, +0.006] | 0.794 |

Against 10-31 (v2.2, Metrica without in-play dropout): Metrica as provided 0.840 -> 0.834;
degraded and stress are not comparable (the noise changed). On SkillCorner v2 no longer trails
v1 (-0.015 -> -0.003, within noise; macroF1 +0.018), presumably because the lower temperature
predicts fewer of the short spells that broadcast tracking fakes.

## 2026-11-15 — where v2.4's remaining error is: data, timing, evidence, a minimum duration (`harness/headroom.py`; no change)

- **Hypothesis.** v2.4 loses most short possessions (harness.segments: under 2 s, 41-45% missed
  outright, under DFL / M1). Better data (the ball detected on the video, say), boundary
  refinement or a decoder that holds every spell a minimum time could win them back.
- **Analysis** (saved out-of-fold outputs, decoded at the exported setting; no fitting, nothing
  on BAR-ATH). spellF1 gain [5-95%], degraded copies:

  | what is made perfect | DFL | M1 | M2 | M3 |
  |---|---|---|---|---|
  | tracking (the clean copy through the same model) | +0.031 [+0.025, +0.036] | +0.032 [+0.027, +0.037] | +0.025 [+0.019, +0.032] | +0.018 [+0.010, +0.026] |
  | timing (every predicted boundary moved onto the nearest true one within 1 s) | +0.013 | +0.013 | +0.014 | +0.014 |

  The ball is seen on half or more of the frames of 91-96% of the lost short possessions.
  The classifier's evidence on them (M1, degraded): under 2 s, lost: frames favouring the true
  team 0.55, mean P(true team) 0.51, decoded as it on 0.29 of their frames; found: 0.98 / 0.81
  / 0.96. 2-4 s, lost: 0.58 / 0.58 / 0.49; found: 0.93 / 0.81 / 0.94. On 42% of the lost
  ones most frames favour the other team.
- **Minimum duration** (`min_duration_viterbi`: each state held at least k frames, the IOHMM's
  per-frame transitions after that; tested against the expanded HMM). k 1 / 13 / 25 (0.5 / 1 s)
  x kappa 0.2 / 0.3 / 0.45 / 0.6, paired against the export (k 1, kappa 0.2), degraded / stress:
  k 13 at kappa 0.2 is DFL +0.004 / -0.000, M1 +0.003 / -0.001, M2 +0.002 / -0.004, M3 -0.001
  / -0.004; at every kappa k 13 is within 0.003 of k 1 and k 25 is no better. Best mean over
  degraded and stress of M1-M3: k 1, kappa 0.2, the export. macroF1 unchanged (0.942).
- **Reading.** Even perfect ball and player tracking buys the same model only 0.02-0.03, below
  the ~0.04 full-match band, and a video ball detector would buy less than that. That cannot be
  checked off-game (none of the harness matches has video with the ball annotated), and it
  needs a video that is not in the bundle on this machine. Perfect timing buys 0.013. The lost
  short possessions are not the decoder's doing: the evidence it gets for them is even, and on
  42% of them it points to the wrong team. A decoder that predicts more short spells (a higher
  temperature) wins some of them back under DFL's labels and loses more under M2-M3
  (control.py), and a minimum duration does not move that trade. What is left is
  the classifier telling a short controlled possession from a touch, with the ball in view.
  Classifier targets on Metrica's rule instead of DFL's labels were tried on 10-28 (M1: -0.003,
  within noise) and are not rerun.
- **Decision.** No change; v2.4 stays. The technical document's limitation on short
  possessions gains these numbers.

## 2026-11-15 — what a ball detector on the video could buy, simulated (`harness/detector.py`; no change)

- **Hypothesis.** A ball detector run on the tactical video (local model, projected through the
  homography) would fill the ATD's ball gaps and win back possessions. Tried in simulation
  first: the video is not in the bundle on this machine, and none of the off-game datasets has
  video with the ball annotated to check a detector on.
- **Change.** None to the pipeline. On the calibrated ATD-like copies (seeds 0-2), a share of the
  frames a copy has lost the ball on while the source has it is given back at the true position
  plus a detector's error (0.5 m, correlated over 0.2 s like the ATD's jitter; or independent
  from frame to frame), as whole gaps at random or on the least crowded frames, on every frame
  or in play only, with or without false detections (a nearby player's position for the whole
  gap). The exported model, not retrained; Metrica's games held out, DFL's in-sample.
- **Result** (spellF1 change [5-95%], M1 labels; own labels within 0.002 of these):

  | ball given back | Metrica (6 halves) | DFL (14 halves) |
  |---|---|---|
  | in play, all gaps, exactly | +0.003 [+0.000, +0.007] | +0.000 [-0.001, +0.002] |
  | in play, all gaps, 0.5 m error | +0.003 [-0.002, +0.007] | -0.001 [-0.003, +0.001] |
  | every frame, all gaps, exactly | -0.005 [-0.015, +0.003] | +0.001 [-0.000, +0.003] |
  | every frame, 25% / 50% / 100% of gaps, 0.5 m error | -0.002 / -0.005 / -0.006 | -0.067 / -0.129 / -0.224 |
  | every frame, 50% of the least crowded frames | -0.007 | -0.109 |
  | every frame, 50% of gaps, a fifth of them false | -0.007 | -0.122 |
  | every frame, 50% of gaps, error independent per frame | -0.005 | -0.132 |

  macroF1 moves alike (DFL, every frame, 50% of gaps: 0.955 -> 0.793). A first run with
  independent error only was replaced by this one: the collapse on DFL was suspected to come
  from independent error making a still ball look fast, but correlated error collapses it as
  much.
- **Reading.** The ball's gaps are not what limits v2: the ball given back on every in-play frame
  the ATD lost, exactly, buys +0.003 at most. Given back in stoppages with any position error it
  costs up to 0.22, because the model has learnt from the ATD (calibrated: it loses the ball in
  most stoppages) that a lost ball means a stoppage, and has hardly seen a ball that jitters
  while play is stopped. DFL's source has the ball throughout its stoppages, so there is more to
  give back there than in Metrica's, which lacks it in most of them. A detector would thus need
  the simulator recalibrated to the ATD and the detector together and v2 retrained, for a gain
  bounded by the perfect-tracking +0.03 (`harness/headroom.py`), of which an exact ball in all
  of the ATD's gaps accounts for +0.001-0.003: the rest is the players and the ball's jitter and
  false detections. The simulation flatters the detector (it is told where the ball is; an
  aerial ball would also project to the wrong place).
- **Decision.** No detector; the video stays unused. The technical document's limitation on
  short possessions says why.

## 2026-11-15 — repository cleanup (no change to the file)

- **Removed** what the submitted version does not use and the documents it no longer needs, so
  the repository shows v2.4: the v1 / v2 plans (`docs/PLAN_v1.md`, `docs/PLAN_v2.md`);
  boundary refinement (`mtah/refine.py`, off in every accepted model; v1's variant with it and
  its tests); Smart Tagging's phases as decoder evidence -- the fusion in `mtah/pipeline.py`
  and `mtah/tagging.py` (weight 0 in every submitted version), the simulated tagger and its
  calibration (`harness/fuse.py`, `harness/tagger.py`, `scripts/calibrate_tagger.py`) and the
  fusion probe of `scripts/diagnose.py`. Everything removed is in commit `0aa1718`; the
  entries above that cite it describe that code. `train_v2.load_outputs()` replaces the
  loader `harness.control` and `harness.headroom` borrowed from `harness/fuse.py`. Question 1
  of `docs/ORGANISER_QUESTIONS.md` removed (the team's decision).
- **Figures.** `harness/figures.py` draws the technical document's four data figures into
  `docs/figures/` from off-game data only (DFL, Metrica, SkillCorner); the two diagrams (the
  pipeline, the validation protocol) are drawn by hand as SVG. The decoder figure shows
  a Metrica game under the simulator's default noise, as the calibrated knobs stay unseen
  (STRICT), in a window chosen by a fixed rule (median spellF1 among 2-minute windows with at
  least two possessions and the ball in play at least 75% of the time). Its numbers reproduce
  the table above exactly.
- **Check.** `make_submission.py` writes a file byte-identical to `XHS.csv` (`cmp`);
  `scripts/diagnose.py` gives the diagnostics of the v2.4 entry unchanged (290 possessions;
  0.976; 0.711 / 0.845; stability 0.014, 0.954); all tests pass.

## 2026-11-16 — question 6 settled by the team's reading; v2.4 is the final file (no change to the file)

- **No organiser's answer** to question 6 (the official repository has issues disabled). The
  team settled it by its own domain expertise: an opponent who only touches the ball without
  controlling it does not change possession, so a touch without control does not make a spell
  of the other team. A general statement about football, not a label of this match. (An earlier
  version of this entry, commit a710dcc, called it an answer; it is the team's reading.)
- **What follows.** Nothing changes. The rule first pre-registered on 11-15 kept v2.3 without
  an answer; the team's v2.4 decision of the same day replaced it (kappa 0.2, chosen on M1-M3,
  to be undone only by an organiser's answer that such touches can form spells). No such answer
  came, and today's reading is the one v2.4 was chosen on. How many seconds make control stays
  our reading (1-3 s); nothing was re-chosen.
- **Baselines.** The four `baseline_*.csv` stay in the repository (team decision; question 4
  has no answer; the official repository commits them on purpose).
- **Public window.** One alarm run on the final `XHS.csv`, smooth50 beside it, ranked table
  only, after the file was fixed and reproduced: no label swap, no misalignment
  (`COMPLIANCE_LOG.md`). Not compared with v2.3's run; nothing changed because of it.
- **Tidying.** Documents brought up to date (`docs/XHS.md` sections 2 and 7,
  `docs/ORGANISER_QUESTIONS.md`, `COMPLIANCE_LOG.md`, `harness/semantics.py`); stale notes
  removed (`mtah/config.py`, `DATASETS.md`, `harness/README.md`). No code path changed.
- **Reproduction.** At a710dcc, `scripts/check_reproducibility.sh evaluation/results/XHS.csv`:
  two clean-room runs (fresh export, fresh virtualenv, empty cache) byte-identical to each
  other and to the working copy's `XHS.csv`; the validator accepts it; all tests pass. About
  32 s and 1.7 GB each. The commits since change documents and one docstring only.

## 2026-11-16 — published from a fresh repository (no change to the file)

- **What.** The team recreates `github.com/ouyang1030/metrica-hack` under the same name, from
  one commit of the final state; the development history above is not published. The submitted
  version keeps its name, v2.4 (renaming it v2.0 was tried and dropped: an earlier model of
  10-29 already has that name in these logs). `docs/XHS.md` no longer cites commits of the
  unpublished history. No code path and no model changed.
- **Reproduction.** On the fresh commit, `scripts/check_reproducibility.sh
  evaluation/results/XHS.csv`: two clean-room runs (fresh export, fresh virtualenv, empty cache)
  byte-identical to each other and to the working copy's `XHS.csv`; the validator accepts it;
  all tests pass. About 33 s and 1.7 GB each. (This line was added to that commit afterwards;
  it changes this file only.)
