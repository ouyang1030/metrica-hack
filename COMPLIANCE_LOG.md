# Compliance log

Every look at the public 5-minute window, every external dataset, every rule question.
`scripts/smoke_test_public.py` appends a line to the last section for each
scorer run; add underneath, by hand, what was (not) changed because of it.

## Data handling

- 2026-10-27 — v0 scaffold. The coding agent (Claude Code, a hosted LLM) read only the official
  repository's documentation and code (commit `f8f2629`). The data bundle was not on the machine.
  The agent did not open `evaluation/gt/*.csv`, the four `baseline_*.csv` files or the official
  `data/README.md`; the CSVs were copied with `cp`. Tests and dry runs used synthetic bundles
  only (`tests/synthetic_bundle.py`, random positions).

- 2026-10-29 — The team put the data bundle in `data/` (tactical and broadcast ATD, homography,
  Smart Tagging; no videos). The agent never listed or opened anything under `data/`, `cache/`
  or `debug/`: a listing of `data/` was refused by the permission guard, and the layout was
  settled by the team against the paths `games.py` documents. Every script that read the data
  ran behind `mtah/safe_run.py`, and the agent saw only: pass/fail checks (parser against the
  four published baselines, byte for byte; two runs of the one command byte-identical; the
  official validator; Smart Tagging kickoff windows containing the published kickoffs; the team
  names mapping onto home and away), runtimes and memory, and summaries of our own predictions
  (`scripts/diagnose.py`: possessions, durations, dead share, the share of the pre-kickoff
  stretch predicted dead, agreement with Smart Tagging team tags, stability when 10% more ball
  frames are removed).
- 2026-10-29 — Simulator calibration under STRICT (`scripts/calibrate_noise.py`): the noise
  statistics of the ATD and the simulator knobs fitted to them were written to
  `cache/aggregates/` and not shown; the agent saw per knob only whether the ATD's statistic
  lay within the knob's search range, and per statistic whether the fit reproduced it within
  10%. The harness now trains and scores on the calibrated noise; the knobs are never printed.
- 2026-10-29 — The second half is graded from before its kickoff (DATA.md); the pipeline now
  marks graded frames before a half's Smart Tagging "Kick Off (Start)" window as dead. Prompted
  by the unsupervised diagnostic (only 31% of that stretch was predicted dead), not by any
  ground truth; no frame number of ours enters the code.
- 2026-10-29 — Disclosure by the team, under STRICT. The calibration could not reproduce the
  ATD's five ball-dropout statistics, and the agent asked the team to compare
  `atd_noise.json` with `fitted_noise.json` in their own terminal and to describe the
  difference in direction and rough size only. The team pasted the five statistics of both into
  the session instead: ten numbers, five of them aggregates of the ATD's graded frames (mean gap,
  flicker share, long-gap share, crowd and speed ratios), nothing frame-indexed. That goes beyond
  STRICT, which the team keeps until the organisers confirm aggregates
  (`docs/ORGANISER_QUESTIONS.md`); the protocol leaves what to share to the team. Use: only to
  choose the structure of the simulator's dropout model (a tracker that loses the ball in
  stoppages). No number from them enters code, tests or docs; the knobs are still fitted by
  `scripts/calibrate_noise.py` from the JSON and never printed.

## External datasets

Each one goes in `DATASETS.md` with source, licence and who labelled it.

- 2026-10-27 — IDSSE / DFL open data (7 Bundesliga matches, CC BY 4.0), downloaded from figshare
  into `external/idsse/` (gitignored). Other matches, labelled by DFL; nothing about BAR-ATH.
  Inspected normally (other games: the firewall does not apply).
- 2026-10-27 — Harness sanity check: the four baselines' spellF1 on the degraded harness was set
  beside the full-match scores the organisers publish (OVERVIEW.md, starter/baseline.py). Those
  published scores are only a comparison point: OVERVIEW.md withholds full-match ground-truth
  statistics precisely so that nothing is tuned against them, so no simulator parameter is set
  from them. The simulator is calibrated from ATD aggregates under STRICT only.
- 2026-10-30 — Metrica Sports sample data (games 1–3), downloaded from
  github.com/metrica-sports/sample-data into `external/metrica/` (gitignored). Other, anonymised
  matches; their event data is human-labelled by Metrica, which RULES.md allows for other
  games. Inspected normally. Used only to score the exported models, never to train or choose.
- 2026-10-30 — SkillCorner open data (20 A-League matches, MIT), read in place from a local copy
  through `external/skillcorner` (gitignored symlink). Labels are SkillCorner's automatic
  outputs. Inspected normally. Used only to score the exported models, never to train or choose.

## Rule questions

Drafted in `docs/ORGANISER_QUESTIONS.md`.

- 2026-11-16 — Question 6 (do touches without control ever form a spell in the ground truth?):
  no organiser's answer. The team settled it by its own domain expertise, as a general
  statement about football and with no frame or event of the match: an opponent who only
  touches the ball without controlling it does not change possession. This is a reading of
  CHALLENGE.md's rule, not a label and not information from the ground truth. Effect: none on
  the file; it is the reading v2.4 (kappa 0.2) was chosen on, off-game, on 2026-11-15.
- Questions 2-5: no organiser's answer by 2026-11-16. Nothing in the submission depends on them: the public
  window was used as an alarm only (the stricter reading of 2), the submission reproduces by
  inference with pinned dependencies (3), and the kickoffs are placed by Smart Tagging's
  windows, not by the published frames (5). Question 4 (may the four baseline CSVs be
  redistributed?) is open; the team decided on 2026-11-16 to keep them in the repository, as
  the official repository does (its `evaluation/results/README.md`: committed on purpose, read
  the ATD and nothing else).

## Public-window looks

- 2026-10-27 `smoke_test_public` on `baseline_raw.csv`, `baseline_smooth25.csv`, `baseline_smooth50.csv`, `baseline_smooth125.csv` -- reason: v0 scorer wiring check: reproduce the README's published public-window board for the four official baselines; no file of ours scored. Alarm only; not used to choose between variants or to set anything.
  - Board identical to the one published in README.md; nothing was changed because of it. Only
    the official files were scored.
- 2026-10-29 `smoke_test_public` on `TEAM_NAME.csv`, `baseline_smooth50.csv` -- reason: v2 alarm after the first run on the real bundle: alignment, label swap, half boundaries; smooth50 beside it as the published reference. Alarm only; not used to choose between variants or to set anything.
  - Only the ranked table was seen (per-file reports with ground-truth transition frames went to
    `cache/eval/`, unread). No alarm: no sign of misalignment, a label swap or a broken half
    boundary. Only `models/v2.json` was scored, no other variant, and nothing was changed because
    of it.
- 2026-11-15 `smoke_test_public` on `kicker.csv`, `baseline_smooth50.csv` -- reason: v2.3 alarm before submission (the last look was v2.0 on 10-29): alignment, label swap, half boundaries; smooth50 beside it as the published reference. Alarm only; not used to choose between variants or to set anything.
  - Only the ranked table was seen (per-file reports went to `cache/eval/`, unread by the
    agent). The rule for organiser question 6 was pre-registered in EXPERIMENTS.md before this
    run. No label swap (spellF1 above smooth50's, which matches the published board) and no
    misalignment (mean transition latency within a third of a second). macroF1 is lower than
    v2.0's on 10-29 while spellF1 is higher; with four stoppages in the window, macroF1's dead
    class rests on a few hundred frames. Not looked at error by error; no other variant scored;
    nothing changed because of it.
- 2026-11-16 `smoke_test_public` on `XHS.csv`, `baseline_smooth50.csv` -- reason: v2.4 alarm on the final file before submission (the last look was v2.3 on 11-15; v2.4 differs by the decoder temperature, chosen off-game on 11-15): alignment, label swap, half boundaries; smooth50 beside it as the published reference. Alarm only; not used to choose between variants or to set anything.
  - Only the ranked table was seen (per-file reports went to `cache/eval/`, unread by the
    agent). No alarm: no label swap (spellF1 and macroF1 above smooth50's, whose line matches
    the published board) and no misalignment (mean transition latency within a third of a
    second). Run after v2.4 was fixed as the final file and reproduced; not set beside v2.3's
    table of 11-15, not looked at error by error, no other variant scored, nothing changed
    because of it.
