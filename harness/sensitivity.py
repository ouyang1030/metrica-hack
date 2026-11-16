"""The exported models where the noise calibration is uncertain (EXPERIMENTS.md, 2026-11-15).

    python -m harness.sensitivity      # after harness.metrica and scripts/calibrate_noise.py

The calibration reproduces when the ATD loses the ball, not when it finds it again, how crowded
the ball is when it is lost, or the ball's jitter (EXPERIMENTS.md, 10-29 and 10-31); and v2
learnt its team features with nearly every player tracked (SkillCorner, 10-31). Each knob of that kind is
pushed away from the calibrated noise, one at a time, and models/v1.json, models/v2.json and
baseline_smooth50 are scored on Metrica's sample games degraded with it (seeds 0-2, the same
seeds for every setting), paired with the calibrated noise itself. Metrica's games are not
among the matches the models were trained on. Not a tuning set: nothing is fitted or chosen
here. STRICT: the knobs move by offsets from calibrated values that are never printed.
"""

import json
import multiprocessing
from dataclasses import replace

import numpy as np

from harness import degrade, external, metrica, score, semantics, train_v1, train_v2

# Setting -> {knob: offset added to its calibrated value}.
SHIFTS = {
    "calibrated": {},
    "found again 1 s earlier": {"ball_dead_off_s": -1.0},
    "found again 0.5 s earlier": {"ball_dead_off_s": -0.5},
    "found again 0.5 s later": {"ball_dead_off_s": 0.5},
    "found again 1 s later": {"ball_dead_off_s": 1.0},
    "found again 2 s later": {"ball_dead_off_s": 2.0},
    "found again 4 s later": {"ball_dead_off_s": 4.0},
    "delays spread +0.5 s": {"ball_dead_lag_sd_s": 0.5},
    "delays spread +1 s": {"ball_dead_lag_sd_s": 1.0},
    "delays spread +2 s": {"ball_dead_lag_sd_s": 2.0},
    "sightings +1/min": {"ball_dead_seen_per_min": 1.0},
    "sightings +2/min": {"ball_dead_seen_per_min": 2.0},
    "sightings +4/min": {"ball_dead_seen_per_min": 4.0},
    # The crowd weight shapes only the in-play dropout, so it moves with more of it: the ball
    # lost in play, in crowds, on that much more of the in-play frames.
    "in-play losses in crowds +2%": {"ball_missing": 0.02, "ball_weight_crowd": 1.2},
    "in-play losses in crowds +5%": {"ball_missing": 0.05, "ball_weight_crowd": 1.2},
    "in-play losses in crowds +10%": {"ball_missing": 0.10, "ball_weight_crowd": 1.2},
    "ball jitter +0.3 m": {"ball_jitter_m": 0.3},
    "players untracked +5%": {"person_missing": 0.05},
    "players untracked +10%": {"person_missing": 0.10},
    "players untracked +20%": {"person_missing": 0.20},
}
LABELS = ("own", "M1")
JOBS = 6  # worker processes, about 1.8 GB each


def shifted(params, offsets):
    """params with each named knob moved by its offset; a share of untracked frames stays <= 0.5."""
    moved = {knob: getattr(params, knob) + offset for knob, offset in offsets.items()}
    if "person_missing" in moved:
        moved["person_missing"] = min(moved["person_missing"], 0.5)
    return replace(params, **moved)


def score_one(task):
    """{(label set, method): (halves, 2) rows, each half averaged over the seeds} of one game
    under one setting."""
    game, setting = task
    with open(train_v2.MODEL_PATH) as handle:
        models = {"v2": json.load(handle)}
    with open(external.V1_MODEL) as handle:
        models["v1"] = json.load(handle)
    match = metrica.load(game)
    targets = {"own": match.labels, "M1": semantics.metrica(match.labels, match.halves, 1.0)}
    params = shifted(train_v2.NOISE, SHIFTS[setting])
    copies = [degrade.degrade(match, seed, params) for seed in train_v1.SEEDS]
    out = {}
    for method, predict in external.METHODS.items():
        predictions = [predict(m, models.get(method)) for m in copies]
        for label_set in LABELS:
            out[label_set, method] = np.mean([score.half_scores(targets[label_set], p,
                                                                match.halves)
                                              for p in predictions], axis=0)
    print("done", match.name, setting, flush=True)
    return out


def run():
    """{(setting, label set, method): (halves, 2) rows}, the games in order."""
    tasks = [(game, setting) for game in metrica.GAMES for setting in SHIFTS]
    with multiprocessing.Pool(JOBS) as pool:
        scored = pool.map(score_one, tasks, chunksize=1)
    rows = {}
    for (_, setting), out in zip(tasks, scored):
        for (label_set, method), part in out.items():
            rows.setdefault((setting, label_set, method), []).append(part)
    return {key: np.concatenate(parts) for key, parts in rows.items()}


def report(rows):
    for label_set in LABELS:
        print("\n%s labels: spellF1 of v2, and paired differences [5-95%%] over %d halves"
              % (label_set, len(rows["calibrated", label_set, "v2"])))
        print("%-28s %6s  %-24s %-24s %-24s %s" % ("noise", "v2", "v2 minus calibrated",
                                                  "v1 minus calibrated", "v2 minus v1",
                                                  "v2 macroF1"))
        for setting in SHIFTS:
            def get(method, at=setting):
                return rows[at, label_set, method]
            cells = []
            for diff in (get("v2") - get("v2", "calibrated"), get("v1") - get("v1", "calibrated"),
                         get("v2") - get("v1")):
                mean, low, high = score.interval(diff)
                cells.append("%+.3f [%+.3f, %+.3f]" % (mean[0], low[0], high[0]))
            print("%-28s %6.3f  %-24s %-24s %-24s %.3f" % (setting, get("v2")[:, 0].mean(), *cells,
                                                         get("v2")[:, 1].mean()))
        smooth = rows["calibrated", label_set, "baseline_smooth50"][:, 0].mean()
        print("baseline_smooth50 at the calibrated noise: %.3f" % smooth)


if __name__ == "__main__":
    report(run())
