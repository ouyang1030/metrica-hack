"""The decoder temperature against the control time Metrica's rule assumes (EXPERIMENTS.md,
2026-11-15).

    python -m harness.control      # after harness.train_v2 (its out-of-fold outputs)

CHALLENGE.md: a touch without control does not change possession, and control might be told by
"a player keeps the ball for a few seconds". How strict that is decides how many short spells
to predict, and the decoder's temperature is what sets it. The saved out-of-fold outputs of v2
(train_v2.save_outputs) are decoded at every setting of train_v2.DECODERS["iohmm"] and at two
lower temperatures, and scored against DFL's labels and under Metrica's rule with control times
of 1 to 4 s (harness/semantics.py), on the calibrated (degraded) and stress copies; paired
against the exported setting. Prints the setting with the best mean over the degraded and
stress copies of train_v2.CHOICE. No fitting; nothing here touches BAR-ATH.
"""

import json

import numpy as np

from harness import idsse, score, semantics, train_v2

SETTINGS = ([dict(kappa=k, gamma=1.0) for k in (0.1, 0.15)]
            + train_v2.DECODERS["iohmm"])
CONTROL_S = {"DFL": None, "M1": 1.0, "M2": 2.0, "M3": 3.0, "M4": 4.0}
COPIES = {"degraded": train_v2.DEGRADED, "stress": (4,)}


def run():
    """{(setting index, label set, copy): (halves, 2) rows; degraded rows average the seeds}."""
    rows = {}
    for match_id in idsse.match_ids():
        match = idsse.load(match_id)
        targets = {name: match.labels if control_s is None
                   else semantics.metrica(match.labels, match.halves, control_s)
                   for name, control_s in CONTROL_S.items()}
        probs, change, model = train_v2.load_outputs(match_id)
        for i, setting in enumerate(SETTINGS):
            paths = {c: train_v2.decode(probs[c], change[c], model, **setting)
                     for copies in COPIES.values() for c in copies}
            for name, labels in targets.items():
                for copy, copies in COPIES.items():
                    rows.setdefault((i, name, copy), []).append(np.mean(
                        [score.half_scores(labels, paths[c], match.halves) for c in copies],
                        axis=0))
        print("done", match_id, flush=True)
    return {key: np.concatenate(parts) for key, parts in rows.items()}


def report(rows):
    with open(train_v2.MODEL_PATH) as handle:
        exported = json.load(handle)
    ref = SETTINGS.index({key: exported[key] for key in ("kappa", "gamma")})
    for name in CONTROL_S:
        for copy in COPIES:
            print("%-3s %-8s" % (name, copy) + "".join(
                "  k%g/g%g %.3f (%+.3f [%+.3f,%+.3f])" % (
                    setting["kappa"], setting["gamma"], rows[i, name, copy][:, 0].mean(),
                    *[v[0] for v in score.interval(rows[i, name, copy]
                                                   - rows[ref, name, copy])])
                for i, setting in enumerate(SETTINGS)))
    best = max(range(len(SETTINGS)), key=lambda i: np.mean(
        [rows[i, name, copy][:, 0].mean() for name in train_v2.CHOICE for copy in COPIES]))
    print("best mean over degraded and stress, %s: %s (exported: %s)" % (
        "/".join(train_v2.CHOICE), SETTINGS[best], SETTINGS[ref]))


if __name__ == "__main__":
    report(run())
