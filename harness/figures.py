"""The data figures of the technical document (docs/figures/), from off-game data only. The two
diagrams there (smart_tagging_pipeline.svg, validation_protocol_redesign.svg) are drawn by hand.

    python -m harness.figures compute   # after harness.train_v2, harness.metrica,
                                        # harness.skillcorner and scripts/calibrate_noise.py
    python -m harness.figures plot      # draw from the saved numbers
    python -m harness.figures           # both

compute gathers every number the figures show into external/figures.pkl (gitignored): the
cross-validated scores train_v2 saved, and fresh runs of harness.control, harness.segments,
harness.headroom, harness.detector and harness.external (about half an hour). plot draws:
  decoder.png    one window of a Metrica game the model was not trained on: what the model
                 says frame by frame, the decoded path, and the true one under three readings
                 of "controlled possession". The window is chosen by a rule: of all 2-minute
                 windows on a 1-minute grid with at least two true possessions (M1) and the
                 ball in play at least MIN_ALIVE of the time, the one with the median
                 spellF1. The game is made ATD-like with the simulator's
                 default (uncalibrated) settings: under STRICT, what the calibrated noise
                 looks like is not shown
  kappa.png      spellF1 against the decoder temperature, per label set
  results.png    v2 against v1 and baseline_smooth50, on the harness and on other providers
  headroom.png   where the possessions are lost, and what perfect data, timing or decoding
                 would buy
Nothing here touches BAR-ATH.
"""

import json
import multiprocessing
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from evaluation import possession_metrics as pm  # noqa: E402
from harness import (control, degrade, detector, external, headroom, idsse, metrica,  # noqa: E402
                     score, segments, semantics, skillcorner, train_v2)
from mtah import atd, config, decoder, pipeline  # noqa: E402

OUT_DIR = os.path.join(config.ROOT, "docs", "figures")
CACHE = os.path.join(config.ROOT, "external", "figures.pkl")
JOBS = 6
WINDOW_S, STEP_S = 120, 60
MIN_ALIVE = 0.75  # of the decoder window's frames in play (M1)
FPS = pm.FPS
LABEL_SETS = ("DFL", "M1", "M2")
METHODS = ("smooth50", "v1", pipeline.VERSION)
BALL_SETTING = "in play: gaps 100%, no error"

LABEL_COLORS = {"H": "#2F6DB5", "A": "#E8892B", "D": "#B4B4B4"}
STRIP = ListedColormap([LABEL_COLORS["H"], LABEL_COLORS["A"], LABEL_COLORS["D"], "#FFFFFF"])
METHOD_COLORS = {"smooth50": "#A0A0A0", "v1": "#8DB0DC", pipeline.VERSION: "#1C3F7A"}
SET_COLORS = {"DFL": "#222222", "M1": "#2F6DB5", "M2": "#2A9D8F", "M3": "#E8892B",
              "M4": "#C0392B"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False})


def _exported():
    with open(train_v2.MODEL_PATH) as handle:
        return json.load(handle)


def _ci(rows):
    """(mean, low, high) of spellF1, and mean macroF1, over (halves, 2) rows."""
    mean, low, high = score.interval(rows)
    return float(mean[0]), float(low[0]), float(high[0]), float(mean[1])


def _saved_scores():
    class Unpickler(pickle.Unpickler):  # pickled by python -m harness.train_v2, from __main__
        def find_class(self, module, name):
            return train_v2.Scores if name == "Scores" else super().find_class(module, name)
    with open(os.path.join(train_v2.OOF_DIR, "scores.pkl"), "rb") as handle:
        return Unpickler(handle).load()


def main_results():
    """{(label set, method, copy): _ci} from train_v2's cross-validation."""
    model = _exported()
    v2 = train_v2._name(train_v2.ACCEPTED, {"kappa": model["kappa"], "gamma": model["gamma"]})
    scores = _saved_scores()
    keys = {"smooth50": "baseline_smooth50", "v1": "v1", pipeline.VERSION: v2}
    return {(label_set, method, copy): _ci(scores[label_set][key].rows(copy))
            for label_set in LABEL_SETS for method, key in keys.items()
            for copy in ("degraded", "stress")}


def kappa_curve():
    """{(kappa, label set, copy): _ci} at gamma 1 (harness.control)."""
    control.SETTINGS = [s for s in control.SETTINGS if s["gamma"] == 1.0]
    rows = control.run()
    return {(control.SETTINGS[i]["kappa"], name, copy): _ci(r)
            for (i, name, copy), r in rows.items()}


def outcomes():
    """{length bin: (share of the true possessions, {outcome: share})} under M1 (segments)."""
    rows = segments.run()["M1"][0]
    out = {}
    for b in range(len(score.BINS) - 1):
        mine = [o for k, o, _ in rows if k == b]
        out[b] = (len(mine) / len(rows), {o: mine.count(o) / len(mine) for o in score.OUTCOMES})
    return out


def gains(pool):
    """{what: (mean, low, high) spellF1 gain under M1, degraded copies}."""
    parts = pool.map(headroom.one, idsse.match_ids())
    rows = {key: np.concatenate([p[0][key] for p in parts]) for key in parts[0][0]}
    kappa = _exported()["kappa"]
    out = {"timing": _ci(rows["timing", "M1"]), "tracking": _ci(rows["data", "M1"]),
           "minimum": _ci(rows["minimum", 13, kappa, "M1", "degraded"]
                          - rows["minimum", 1, kappa, "M1", "degraded"])}
    tasks = [(source, name, setting)
             for source, names in (("Metrica", metrica.GAMES), ("DFL", idsse.match_ids()))
             for name in names for setting in ("calibrated", BALL_SETTING)]
    scored = dict(zip(tasks, pool.map(detector.score_one, tasks, chunksize=1)))
    for source in ("Metrica", "DFL"):
        def cat(setting):
            return np.concatenate([out_["M1"] for (s, _, st), out_ in scored.items()
                                   if s == source and st == setting])
        out["ball " + source] = _ci(cat(BALL_SETTING) - cat("calibrated"))
    return out


def external_results():
    """{(set, copy, method): _ci} under M1 (harness.external)."""
    rows = external.run({
        "Metrica": ([metrica.load(game) for game in metrica.GAMES], False),
        "SkillCorner": ([skillcorner.load(m) for m in skillcorner.match_ids()], True)})
    names = {"baseline_smooth50": "smooth50", "v1": "v1", "v2": pipeline.VERSION}
    return {(set_name, copy, names[method]): _ci(r)
            for (set_name, copy, label_set, method), r in rows.items() if label_set == "M1"}


def window():
    """The decoder figure's window (see the module docstring) and everything drawn in it."""
    model = _exported()
    games, candidates = {}, []
    n = WINDOW_S * FPS
    for game in metrica.GAMES:
        match = metrica.load(game)
        copy = degrade.degrade(match, 0, degrade.Params())
        probs, change = pipeline.outputs(copy.players, copy.ball, copy.slot_team, copy.pitch,
                                         model)
        labels = {"own": match.labels}
        labels.update({"M%d" % s: semantics.metrica(match.labels, match.halves, float(s))
                       for s in (1, 2)})
        paths = {kappa: pipeline.decode_probs(probs, model, kappa, change, model["gamma"])
                 for kappa in (model["kappa"], 1.0)}
        games[game] = dict(seen=~np.isnan(copy.ball[:, atd.X]), probs=probs, change=change,
                           labels=labels, paths=paths, halves=match.halves)
        for a, b in match.halves:
            for start in range(a, b - n + 1, STEP_S * FPS):
                if (len(pm.possession_spells(labels["M1"], [(start, start + n)])) >= 2
                        and np.mean(labels["M1"][start:start + n] != pm.D) >= MIN_ALIVE):
                    f1 = score.half_scores(labels["M1"], paths[model["kappa"]],
                                           [(start, start + n)])[0, 0]
                    candidates.append((f1, game, start))
        print("window: done game", game, flush=True)
    candidates.sort()
    f1, game, start = candidates[(len(candidates) - 1) // 2]
    g, cut = games[game], slice(start, start + n)
    half = next(i for i, (a, b) in enumerate(g["halves"]) if a <= start < b)
    return dict(game=game, half=half + 1, start_s=(start - g["halves"][half][0]) / FPS,
                f1=f1, n_windows=len(candidates), kappa=model["kappa"],
                seen=g["seen"][cut], probs=g["probs"][cut], change=g["change"][cut],
                labels={k: v[cut] for k, v in g["labels"].items()},
                paths={k: v[cut] for k, v in g["paths"].items()})


def compute():
    numbers = {"main": main_results(), "kappa": kappa_curve(), "outcomes": outcomes(),
               "window": window()}
    with multiprocessing.Pool(JOBS) as pool:
        numbers["gains"] = gains(pool)
    numbers["external"] = external_results()
    with open(CACHE, "wb") as handle:
        pickle.dump(numbers, handle)
    print("wrote", os.path.relpath(CACHE, config.ROOT))


# -- drawing --------------------------------------------------------------------------------------

def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), metadata={"Software": None})
    plt.close(fig)
    print("wrote", os.path.join("docs", "figures", name))


def _strip(ax, codes, label):
    ax.imshow(codes[None, :], aspect="auto", cmap=STRIP, vmin=0, vmax=3, interpolation="nearest",
              extent=(0, len(codes) / FPS, 0, 1))
    ax.set_yticks([0.5])
    ax.set_yticklabels([label])
    ax.tick_params(axis="y", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def draw_decoder(w):
    t = np.arange(len(w["seen"])) / FPS
    rows = [("ball", 0.45), ("probs", 1.6), ("change", 1.1), ("strip", 0.38), ("strip", 0.38),
            ("gap", 0.18), ("strip", 0.38), ("strip", 0.38), ("strip", 0.38)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.6, 5.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[r[1] for r in rows], hspace=0.28))
    ax = axes[0]
    ax.fill_between(t, 0, w["seen"].astype(float), step="post", color="#444444", linewidth=0)
    ax.set_yticks([0.5])
    ax.set_yticklabels(["ball located"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0, 1)
    ax.spines["left"].set_visible(False)
    end_s = w["start_s"] + len(w["seen"]) / FPS
    ax.set_title("Metrica game %s, %s half, %d:%02d–%d:%02d · spellF1 (M1) %.2f, the median of "
                 "%d windows" % (w["game"], ("1st", "2nd")[w["half"] - 1], w["start_s"] // 60,
                                 w["start_s"] % 60, end_s // 60, end_s % 60, w["f1"],
                                 w["n_windows"]), loc="left")
    ax = axes[1]
    ax.stackplot(t, w["probs"].T, colors=[LABEL_COLORS[k] for k in "HAD"], linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("stage 2:\nP(H, A, D)", rotation=0, ha="right", va="center")
    ax = axes[2]
    for kind, name, color in ((decoder.SWITCH, "switch", "#6A3D9A"),
                              (decoder.STOP, "stop", "#555555"),
                              (decoder.RESTART, "restart", "#2A9D8F")):
        ax.plot(t, w["change"][:, kind], color=color, lw=0.9, label=name)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(change\nwithin 0.2 s)", rotation=0, ha="right", va="center")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, handlelength=1.2)
    _strip(axes[3], w["paths"][w["kappa"]], "v2.4, κ %g" % w["kappa"])
    _strip(axes[4], w["paths"][1.0], "κ 1 (untempered)")
    axes[5].axis("off")
    _strip(axes[6], w["labels"]["own"], "true: Metrica's events")
    _strip(axes[7], w["labels"]["M1"], "true: control ≥ 1 s (M1)")
    _strip(axes[8], w["labels"]["M2"], "true: control ≥ 2 s (M2)")
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xlabel("seconds")
    handles = [Patch(color=LABEL_COLORS[k], label=name) for k, name in
               (("H", "home"), ("A", "away"), ("D", "dead ball"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    _save(fig, "decoder.png")


def draw_kappa(numbers):
    kappas = sorted({k for k, _, _ in numbers})
    sets = [s for s in SET_COLORS if any(n == s for _, n, _ in numbers)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=False)
    for ax, (copy, title) in zip(axes, (("degraded", "calibrated noise"),
                                        ("stress", "stress noise"))):
        for name in sets:
            mean = [numbers[k, name, copy][0] for k in kappas]
            ax.plot(kappas, mean, marker="o", ms=3, lw=1.3, color=SET_COLORS[name],
                    label={"DFL": "DFL's labels"}.get(name, "%s: control ≥ %s s" % (
                        name, name[1:])))
        ax.axvline(0.2, color="#1C3F7A", ls="--", lw=0.9)
        ax.text(0.205, ax.get_ylim()[1], " submitted", color="#1C3F7A", fontsize=7, va="top")
        ax.axvline(0.45, color="#999999", ls=":", lw=0.9)
        ax.text(0.455, ax.get_ylim()[1], " v2.3", color="#777777", fontsize=7, va="top")
        ax.set_xlabel("decoder temperature κ (lower: fewer, longer spells)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("spellF1, 7 held-out DFL matches")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    _save(fig, "kappa.png")


def _dots(ax, groups, getter, offsets=(-0.22, 0, 0.22), hollow=None):
    for g, group in enumerate(groups):
        for method, dx in zip(METHODS, offsets):
            mean, low, high, _ = getter(group, method)
            filled = hollow is None or not hollow(group)
            ax.errorbar(g + dx, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=4.5,
                        color=METHOD_COLORS[method], mfc=METHOD_COLORS[method] if filled
                        else "white", capsize=2, lw=1.1,
                        label=method if g == 0 else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.6, len(groups) - 0.4)


def draw_results(main, ext):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[3, 2.2]))
    ax = axes[0]
    groups = [(s, c) for s in LABEL_SETS for c in ("degraded", "stress")]
    _dots(ax, groups, lambda g, m: main[g[0], m, g[1]], hollow=lambda g: g[1] == "stress")
    ax.set_xticklabels(["%s\n%s" % (s, "calibrated" if c == "degraded" else "stress")
                        for s, c in groups], fontsize=7.2)
    ax.set_ylabel("spellF1 [5–95%]")
    ax.set_title("Harness: 7 held-out DFL matches", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)
    ax = axes[1]
    groups = [("Metrica", "provided"), ("Metrica", "degraded"), ("Metrica", "stress"),
              ("SkillCorner", "provided")]
    _dots(ax, groups, lambda g, m: ext[g[0], g[1], m])
    ax.set_xticklabels(["Metrica\nas given", "Metrica\ncalibrated", "Metrica\nstress",
                        "SkillCorner\nbroadcast"], fontsize=7.2)
    ax.set_title("Other providers, M1 labels", loc="left")
    for a in axes:
        a.set_ylim(0, 1)
        a.grid(axis="y", color="#EEEEEE", lw=0.6)
        a.set_axisbelow(True)
    _save(fig, "results.png")


def draw_headroom(out, gain):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8),
                             gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.85))
    ax = axes[0]
    names = ["< 2 s", "2–4 s", "4–10 s", "≥ 10 s"]
    colors = {"found": "#1C3F7A", "merged": "#E8892B", "split": "#6A3D9A", "missed": "#C0392B",
              "shifted": "#BBBBBB"}
    for b, name in enumerate(names):
        share, parts = out[b]
        left = 0
        for o in score.OUTCOMES:
            ax.barh(b, parts[o], left=left, color=colors[o], height=0.65,
                    label=o if b == 0 else None)
            left += parts[o]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(["%s\n%.0f%% of all" % (name, 100 * out[b][0])
                        for b, name in enumerate(names)], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the true possessions of that length")
    ax.set_title("What becomes of each true possession (M1)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5, fontsize=7,
              handlelength=1, columnspacing=0.8)
    ax = axes[1]
    rows = [("perfect boundary timing\n(oracle, within 1 s)", gain["timing"]),
            ("perfect ball and players\n(clean tracking)", gain["tracking"]),
            ("perfect ball in the ATD's\ngaps (Metrica, held out)", gain["ball Metrica"]),
            ("perfect ball in the ATD's\ngaps (DFL)", gain["ball DFL"]),
            ("minimum spell of 0.5 s\nin the decoder", gain["minimum"])]
    for i, (name, (mean, low, high, _)) in enumerate(rows):
        ax.barh(i, mean, color="#8DB0DC", height=0.6)
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="none", color="#1C3F7A",
                    capsize=2, lw=1)
    ax.axvline(0.04, color="#C0392B", ls="--", lw=0.9)
    ax.text(0.0405, len(rows) - 0.55, "full-match\nnoise band", color="#C0392B", fontsize=7,
            va="bottom")
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(-0.01, 0.05)
    ax.set_xlabel("spellF1 gain, M1, calibrated noise")
    ax.set_title("What would help (upper bounds)", loc="left")
    _save(fig, "headroom.png")


def plot():
    if not os.path.exists(CACHE):
        print("no", os.path.relpath(CACHE, config.ROOT), "-- run: python -m harness.figures "
              "compute")
        return
    with open(CACHE, "rb") as handle:
        numbers = pickle.load(handle)
    draw_decoder(numbers["window"])
    draw_kappa(numbers["kappa"])
    draw_results(numbers["main"], numbers["external"])
    draw_headroom(numbers["outcomes"], numbers["gains"])
    for (label_set, method, copy), (mean, low, high, macro) in sorted(numbers["main"].items()):
        if copy == "degraded":
            print("table: %-4s %-9s %.3f [%.3f, %.3f] macroF1 %.3f" % (label_set, method, mean,
                                                                       low, high, macro))


if __name__ == "__main__":
    what = sys.argv[1:] or ["compute", "plot"]
    if "compute" in what:
        compute()
    if "plot" in what:
        plot()
