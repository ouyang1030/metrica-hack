"""What a ball detector run on the video could buy v2, before building one (EXPERIMENTS.md,
2026-11-15).

    python -m harness.detector     # after harness.metrica and scripts/calibrate_noise.py

A detector would fill some of the frames on which the ATD has lost the ball, with its own
position error and some false detections. Here it is simulated on the calibrated ATD-like copies
(harness/degrade.py): of the frames a copy has lost the ball on while the source has it, a share
is given back -- whole gaps chosen at random ("gaps"), or the frames with the fewest players
near the ball, which an uncluttered view shows best ("open") -- at the true position plus an
error correlated over 0.2 s (or independent from frame to frame), on every frame or in play only
(the ATD also loses the ball in stoppages, and the model reads that); a share of the gaps given
back are a false detection instead, the position of a nearby player for the whole gap.
models/v2.json as exported (no retraining, so the model sees less noise than it was trained on)
is scored on Metrica's games, which it was not trained on, and on DFL's (in-sample for the
exported model), paired with the calibrated copies on the same seeds. Optimistic on purpose: the
detector is told where the ball is, it only decides whether it finds it; on the real video an
aerial ball also projects through the homography to the wrong place. STRICT: the copies use the
calibrated knobs, never printed.
"""

import json
import multiprocessing

import numpy as np

from evaluation import possession_metrics as pm
from harness import degrade, idsse, metrica, score, semantics, train_v1, train_v2
from harness.external import v2
from mtah import atd

JOBS = 6
CROWD_RADIUS_M = 3.0
ERROR_CORR_S = 0.2  # a tracked (smoothed) detector's error, correlated like the ATD's jitter
# Setting -> (share of the lost frames given back, how they are chosen, position error in
# metres, independent from frame to frame (an untracked detector) instead, share of the gaps
# given back that are a false detection, in play only).
SETTINGS = {
    "calibrated": (0.0, "gaps", 0.0, False, 0.0, False),
    "gaps 25%": (0.25, "gaps", 0.5, False, 0.0, False),
    "gaps 50%": (0.5, "gaps", 0.5, False, 0.0, False),
    "gaps 100%": (1.0, "gaps", 0.5, False, 0.0, False),
    "gaps 100%, no error": (1.0, "gaps", 0.0, False, 0.0, False),
    "open 50%": (0.5, "open", 0.5, False, 0.0, False),
    "gaps 50%, 20% false": (0.5, "gaps", 0.5, False, 0.2, False),
    "gaps 50%, untracked error": (0.5, "gaps", 0.5, True, 0.0, False),
    "in play: gaps 100%": (1.0, "gaps", 0.5, False, 0.0, True),
    "in play: gaps 100%, no error": (1.0, "gaps", 0.0, False, 0.0, True),
}
LABELS = ("own", "M1")


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return edges.reshape(-1, 2)


def detect(clean, copy, share, how, error_m, independent, false, play_only, rng):
    """copy's ball with share of its lost frames (where clean has the ball; in play only, if
    play_only) given back."""
    ball = copy.ball.copy()
    lost = (np.isnan(ball[:, atd.X]) & ~np.isnan(clean.ball[:, atd.X])
            & ~np.isin(clean.labels, (pm.X, pm.D) if play_only else (pm.X,)))
    length, width = clean.pitch
    found = np.zeros(len(ball), dtype=bool)
    if how == "gaps":
        gaps = _runs(lost)
        for start, stop in gaps[rng.random(len(gaps)) < share]:
            found[start:stop] = True
    else:
        near = (np.hypot((clean.players[:, :, atd.X] - clean.ball[:, None, atd.X]) * length,
                         (clean.players[:, :, atd.Y] - clean.ball[:, None, atd.Y]) * width)
                < CROWD_RADIUS_M).sum(axis=1)
        frames = np.flatnonzero(lost)
        order = frames[np.lexsort((rng.random(len(frames)), near[frames]))]
        found[order[:int(round(share * len(frames)))]] = True
    ball[found] = clean.ball[found]
    error = (rng.standard_normal((len(ball), 2)) if independent
             else degrade.smooth_noise(rng, (len(ball), 2), ERROR_CORR_S * degrade.FPS))
    ball[found, atd.X] += error[found, 0] * error_m / length
    ball[found, atd.Y] += error[found, 1] * error_m / width
    for start, stop in _runs(found):
        if rng.random() < false:
            seen = np.flatnonzero(~np.isnan(clean.players[start, :, atd.X]))
            person = seen[np.argsort(np.hypot(
                (clean.players[start, seen, atd.X] - clean.ball[start, atd.X]) * length,
                (clean.players[start, seen, atd.Y] - clean.ball[start, atd.Y]) * width))[
                rng.integers(1, 4)]]
            ball[start:stop, :2] = clean.players[start:stop, person, :2]
    return ball


def score_one(task):
    source, name, setting = task
    with open(train_v2.MODEL_PATH) as handle:
        model = json.load(handle)
    match = metrica.load(name) if source == "Metrica" else idsse.load(name)
    targets = {"own": match.labels, "M1": semantics.metrica(match.labels, match.halves, 1.0)}
    share, how, error_m, independent, false, play_only = SETTINGS[setting]
    preds = []
    for seed in train_v1.SEEDS:
        copy = degrade.degrade(match, seed, train_v2.NOISE)
        copy.ball[:] = detect(match, copy, share, how, error_m, independent, false,
                              play_only, np.random.default_rng(1000 + seed))
        preds.append(v2(copy, model))
    print("done", source, name, setting, flush=True)
    return {label_set: np.mean([score.half_scores(labels, p, match.halves) for p in preds],
                               axis=0) for label_set, labels in targets.items()}


def run():
    tasks = [(source, name, setting)
             for source, names in (("Metrica", metrica.GAMES), ("DFL", idsse.match_ids()))
             for name in names for setting in SETTINGS]
    with multiprocessing.Pool(JOBS) as pool:
        scored = pool.map(score_one, tasks, chunksize=1)
    rows = {}
    for (source, _, setting), out in zip(tasks, scored):
        for label_set, part in out.items():
            rows.setdefault((source, setting, label_set), []).append(part)
    return {key: np.concatenate(parts) for key, parts in rows.items()}


def report(rows):
    for source in ("Metrica", "DFL"):
        for label_set in LABELS:
            base = rows[source, "calibrated", label_set]
            print("\n%s (%s), %s labels: v2 spellF1, minus calibrated [5-95%%], macroF1; %d "
                  "halves" % (source, "held out" if source == "Metrica" else "in-sample",
                              label_set, len(base)))
            for setting in SETTINGS:
                mine = rows[source, setting, label_set]
                mean, low, high = score.interval(mine - base)
                print("  %-30s %.3f  %+.3f [%+.3f, %+.3f]  %.3f" % (
                    setting, mine[:, 0].mean(), mean[0], low[0], high[0], mine[:, 1].mean()))


if __name__ == "__main__":
    report(run())
