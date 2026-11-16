"""Where v2.4's remaining error is, and what better data, timing or decoding could buy
(EXPERIMENTS.md, 2026-11-15).

    python -m harness.headroom     # after harness.train_v2 (its out-of-fold outputs)

From the saved out-of-fold outputs of v2 (train_v2.save_outputs), decoded at the exported
setting (models/v2.json), no fitting:
  data      spellF1 of the clean copy minus the degraded copies: what perfect tracking (ball and
            players) would buy the same model;
  timing    spellF1 with every predicted boundary moved onto the nearest true one within 1 s:
            what perfect boundary refinement would buy;
  evidence  for the true possessions under 4 s (M1 labels, degraded copies), found or lost, the
            classifier's per-frame evidence for the true team: whether a lost one is the
            decoder's doing (evidence there, not followed) or the classifier's (no evidence);
  minimum   decoding with a minimum duration of k frames per state (min_duration_viterbi) over
            a grid of k and kappa, scored like harness/control.py on the degraded and stress
            copies, paired against the exported setting; best mean over train_v2.CHOICE.
Nothing here touches BAR-ATH.
"""

import json
import multiprocessing

import numpy as np

from evaluation import possession_metrics as pm
from harness import idsse, score, semantics, train_v2
from mtah import decoder, emissions

JOBS = 6
SNAP_S = 1.0
SHORT = 100  # frames: true possessions under 4 s
MIN_FRAMES = (1, 13, 25)  # no minimum, 0.5 s, 1 s
KAPPAS = (0.2, 0.3, 0.45, 0.6)
CONTROL_S = {"DFL": None, "M1": 1.0, "M2": 2.0, "M3": 3.0}
COPIES = {"degraded": train_v2.DEGRADED, "stress": (4,)}
CLEAN = 0


def min_duration_viterbi(log_emit, log_trans, k):
    """Most likely state per frame, each state held for at least k frames; log_trans (n, 3, 3)
    per frame (decoder.input_transitions). A state is k sub-states visited in a row: the first
    k - 1 move on with certainty, the last holds with log_trans[t, s, s] or leaves to another
    state's first sub-state with log_trans[t, s, r]. k = 1 is decoder.viterbi."""
    if k == 1:
        return decoder.viterbi(log_emit, log_trans)
    n, states = log_emit.shape
    off = ~np.eye(states, dtype=bool)
    best = np.full((states, k), -np.inf)
    best[:, 0] = log_emit[0]
    entered_from = np.zeros((n, states), dtype=np.int8)
    held = np.zeros((n, states), dtype=bool)
    for t in range(1, n):
        enter = best[:, k - 1][:, None] + np.where(off, log_trans[t], -np.inf)
        entered_from[t] = enter.argmax(axis=0)
        hold = best[:, k - 1] + np.diag(log_trans[t])
        held[t] = hold >= best[:, k - 2]
        new = np.empty_like(best)
        new[:, 0] = enter.max(axis=0)
        new[:, 1:k - 1] = best[:, :k - 2]
        new[:, k - 1] = np.where(held[t], hold, best[:, k - 2])
        best = new + log_emit[t][:, None]
    path = np.empty(n, dtype=np.int8)
    state, sub = np.unravel_index(best.argmax(), best.shape)
    for t in range(n - 1, 0, -1):
        path[t] = state
        if sub == 0:
            state, sub = entered_from[t, state], k - 1
        elif sub < k - 1 or not held[t, state]:
            sub -= 1
    path[0] = state
    return path


def decode(probs, change, model, k, kappa):
    log_trans = decoder.input_transitions(np.array(model["log_trans"]), model["mean_frames"],
                                          change, model["change_window"], 1.0)
    return min_duration_viterbi(emissions.tempered(probs, model["prior"], kappa), log_trans, k)


def snap(pred, labels, halves, window):
    """pred with each boundary moved onto the nearest true boundary within window frames."""
    out = pred.copy()
    for a, b in halves:
        true = np.flatnonzero(np.diff(labels[a:b])) + 1 + a
        for edge in np.flatnonzero(np.diff(pred[a:b])) + 1 + a:
            j = np.searchsorted(true, edge)
            near = [t for t in true[max(j - 1, 0):j + 1] if abs(t - edge) <= window]
            if near:
                t = min(near, key=lambda t: abs(t - edge))
                out[min(t, edge):max(t, edge)] = pred[edge] if t < edge else pred[edge - 1]
    return out


def one(match_id):
    """{key: (halves, 2) rows} and [(length bin, found, evidence...)] for one DFL match."""
    match = idsse.load(match_id)
    targets = {name: match.labels if control_s is None
               else semantics.metrica(match.labels, match.halves, control_s)
               for name, control_s in CONTROL_S.items()}
    probs, change, model = train_v2.load_outputs(match_id)
    with open(train_v2.MODEL_PATH) as handle:
        exported = json.load(handle)
    rows, short = {}, []
    for k in MIN_FRAMES:
        for kappa in KAPPAS:
            paths = {c: decode(probs[c], change[c], model, k, kappa)
                     for c in (CLEAN,) + train_v2.DEGRADED + COPIES["stress"]}
            for name, labels in targets.items():
                for copy, copies in COPIES.items():
                    rows["minimum", k, kappa, name, copy] = np.mean(
                        [score.half_scores(labels, paths[c], match.halves) for c in copies],
                        axis=0)
            if (k, kappa) != (1, exported["kappa"]):
                continue
            for name, labels in targets.items():
                rows["data", name] = (score.half_scores(labels, paths[CLEAN], match.halves)
                                      - rows["minimum", k, kappa, name, "degraded"])
                rows["timing", name] = np.mean([score.half_scores(
                    labels, snap(paths[c], labels, match.halves, int(SNAP_S * pm.FPS)),
                    match.halves) for c in train_v2.DEGRADED], axis=0) - rows[
                    "minimum", k, kappa, name, "degraded"]
            for c in train_v2.DEGRADED:
                short.extend(_evidence(targets["M1"], paths[c], probs[c], match.halves))
    print("done", match_id, flush=True)
    return rows, short


def _evidence(labels, pred, probs, halves):
    out = []
    for a, b in halves:
        truth = pm.possession_spells(labels[a:b], [(0, b - a)])
        found = pm.matched_gt_flags(pm.spell_overlaps(truth, pm.possession_spells(
            pred[a:b], [(0, b - a)])), len(truth), pm.SPELL_IOU_HEADLINE)
        for (start, stop, team), hit in zip(truth, found):
            if stop - start < SHORT:
                p = probs[a + start:a + stop]
                favoured = np.mean(p[:, team] > p[:, 1 - team])
                out.append((score._bin(stop - start), bool(hit), favoured, p[:, team].mean(),
                            np.mean(pred[a + start:a + stop] == team)))
    return out


def report(parts):
    rows = {key: np.concatenate([p[0][key] for p in parts]) for key in parts[0][0]}
    short = [r for p in parts for r in p[1]]
    with open(train_v2.MODEL_PATH) as handle:
        kappa = json.load(handle)["kappa"]
    for what in ("data", "timing"):
        print("%-6s" % what + "".join("  %s %+.3f [%+.3f, %+.3f]" % (
            name, *[v[0] for v in score.interval(rows[what, name])]) for name in CONTROL_S))
    for b, name in enumerate(("<2 s", "2-4 s")):
        for hit in (True, False):
            mine = np.array([r[2:] for r in short if r[0] == b and r[1] == hit])
            print("evidence %-5s %-5s n %5d: frames favouring the true team %.2f, mean P(true "
                  "team) %.2f, possessions with most frames favouring it %.2f, decoded as it "
                  "%.2f" % (name, "found" if hit else "lost", len(mine), mine[:, 0].mean(),
                            mine[:, 1].mean(), np.mean(mine[:, 0] >= 0.5), mine[:, 2].mean()))
    settings = [(k, kap) for k in MIN_FRAMES for kap in KAPPAS]
    for name in CONTROL_S:
        for copy in COPIES:
            print("minimum %-3s %-8s" % (name, copy) + "".join(
                "  k%d/kappa %g %.3f (%+.3f [%+.3f, %+.3f])" % (
                    k, kap, rows["minimum", k, kap, name, copy][:, 0].mean(),
                    *[v[0] for v in score.interval(rows["minimum", k, kap, name, copy]
                                                   - rows["minimum", 1, kappa, name, copy])])
                for k, kap in settings))
    best = max(settings, key=lambda s: np.mean([rows["minimum", *s, name, copy][:, 0].mean()
                                                for name in train_v2.CHOICE for copy in COPIES]))
    print("minimum: best mean over degraded and stress, %s: k %d, kappa %g (exported: k 1, "
          "kappa %g)" % ("/".join(train_v2.CHOICE), *best, kappa))


if __name__ == "__main__":
    with multiprocessing.Pool(JOBS) as pool:
        report(pool.map(one, idsse.match_ids()))
