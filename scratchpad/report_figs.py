"""Report figures for notebook 05.

Every number here is transcribed from the executed notebook
`notebooks/runs/05_function_vector.ipynb`, cell by cell, with the source cell named beside
each block. Nothing is recomputed and nothing is new — this only redraws data that already
exists, which is the one kind of code the application's +2h window permits.

Three figures the run did not produce:

  A  the trade      every arm on blackmail AND usable together. Cell 15's own chart plots
                    blackmail against coherence, which hides the metric that matters: a model
                    that stopped writing scores 0.00 blackmail and looks best.
  B  the delivery   frozen vs live on the SAME units. Cell 15b printed this and drew nothing.
     dissociation
  C  the walk-back  termination rate as units are removed, against the flat ARC line. The
                    methodological finding — a capability check that cannot see the failure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib

FIG = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

RED, GREEN, GREY, BLUE = "#C62828", "#2E7D32", "#9E9E9E", "#1565C0"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- A: the trade
# cell 15 (arms at n=40) and cell 15b (live arms at n=24).
# (label, blackmail k, blackmail denom = coherent, usable k, n generated)
ARMS = [
    ("do nothing",            10, 40, 30, 40),
    ("prompt only",            2, 40, 38, 40),
    ("frozen, 17u",            0, 21, 21, 40),
    ("frozen, 8u",             1, 35, 34, 40),
    ("frozen random dir",      4, 40, 36, 40),
    ("live paste, 17u",        5, 40, 35, 40),
    ("live paste, 8u",         2, 24, 22, 24),
    ("live random, 8u",        5, 24, 19, 24),
]
ARMS.sort(key=lambda r: r[3] / r[4])

fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(ARMS))
for i, (lab, bk, bn, uk, un) in enumerate(ARMS):
    bp, blo, bhi = wilson(bk, bn)
    up, ulo, uhi = wilson(uk, un)
    ax.barh(i + 0.19, up, 0.36, color=GREEN, alpha=.85,
            label="USABLE (coherent AND not blackmail, over all generated)" if i == 0 else None)
    ax.barh(i - 0.19, bp, 0.36, color=RED, alpha=.85,
            label="blackmail (over coherent)" if i == 0 else None)
    ax.plot([ulo, uhi], [i + .19] * 2, color="k", lw=1)
    ax.plot([blo, bhi], [i - .19] * 2, color="k", lw=1)
    ax.text(1.20, i, f"n={un}", va="center", fontsize=7, color="#777")

nothing = 30 / 40
ax.axvline(nothing, color=GREY, ls="--", lw=1.2, zorder=0)
ax.text(nothing, len(ARMS) - .45, " doing nothing", fontsize=7.5, color="#777", va="center")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ARMS])
ax.set_xlim(0, 1.26)
ax.set_xticks(np.arange(0, 1.01, .2))
ax.set_xlabel("rate (bars are Wilson 95% CIs)")
ax.set_title("gemma-3-12b: removing blackmail is easy; keeping the model is not\n"
             "frozen 17u reaches 0.00 blackmail and lands BELOW doing nothing on usable",
             fontsize=9.5, loc="left", pad=26)
ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
fig.savefig(FIG / "A_the_trade.png")
plt.close(fig)

# ---------------------------------------------------------------- B: delivery dissociation
# cell 15 (frozen 17u, live 17u, untouched) and cell 15b (frozen 8u, live 8u).
# metric: (frozen17, live17, frozen8, live8, untouched)
MET = {
    "coherent":            (0.53, 1.00, 0.88, 1.00, 1.00),
    "hit the 2048 cap":    (1.00, 0.00, 0.38, 0.00, 0.00),
    "repetition, p90":     (1.000, 0.009, 0.519, 0.006, 0.006),
}
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4))
groups = ["frozen", "live", "frozen", "live", "untouched"]
X = [0, 1, 2.5, 3.5, 5.0]                       # gap separates the 17u pair from the 8u pair
cols = [RED, BLUE, RED, BLUE, GREY]
for ax, (name, vals) in zip(axes, MET.items()):
    ax.bar(X, vals, 0.8, color=cols, alpha=.85)
    for x, v in zip(X, vals):
        ax.text(x, v + .04, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(X)
    ax.set_xticklabels(groups, fontsize=7.5, rotation=30, ha="right")
    ax.set_xlim(-.7, 5.7)
    ax.set_ylim(0, 1.22)
    ax.set_title(name, fontsize=9)
    for x, lab in ((0.5, "17 units"), (3.0, "8 units")):
        ax.text(x, 1.16, lab, ha="center", fontsize=7.5, color="#555")
    ax.axvline(1.75, color="#E0E0E0", lw=1, zorder=0)
    ax.axvline(4.25, color="#E0E0E0", lw=1, zorder=0)
axes[0].set_ylabel("rate")
fig.suptitle("Same units, two deliveries: the damage is in freezing, not in the units",
             fontsize=10.5, x=.005, ha="left", y=1.06)
fig.text(.005, -.13,
         "Only the delivery mode differs inside each red/blue pair. A frozen vector writes one "
         "mean value regardless of context; the live\ndonor responds to what the recipient just "
         "wrote. Live 17u is indistinguishable from the untouched model on all three.",
         fontsize=7.5, color="#555")
fig.savefig(FIG / "B_frozen_vs_live.png")
plt.close(fig)

# ---------------------------------------------------------------- C: walk-back vs ARC
# cell 12b. 8 probes of 1536 tokens at each k.
K = list(range(17, 7, -1))
ENDED = [0.00, 0.00, 0.125, 0.125, 0.375, 0.50, 0.625, 0.25, 0.50, 0.75]
REPEAT = [0.278, 0.424, 0.147, 0.253, 0.107, 0.162, 0.194, 0.303, 0.132, 0.026]

fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.axhline(1.00, color=GREY, lw=1, ls=":", label="untouched model: terminates every time")
ax.axhline(0.94, color=GREEN, lw=2.2, label="ARC-Easy accuracy at 17 units (0.94, unchanged)")
ax.plot(K, ENDED, "o-", color=BLUE, lw=2.2, label="terminated (emitted EOS)")
ax.plot(K, REPEAT, "s--", color=RED, lw=1.4, ms=4, label="repetition, mean")
ax.axvline(8, color="#CCC", lw=1, zorder=0)
ax.annotate("largest coherent prefix:\n8 units, 89% of headroom",
            xy=(8.15, .78), xytext=(10.9, 1.12), fontsize=7.5, color="#444", ha="center",
            arrowprops=dict(arrowstyle="->", color="#999", lw=1))
ax.annotate("greedy optimum: 17 units, 98% of\nheadroom — and never terminates",
            xy=(16.95, .03), xytext=(15.0, 1.12), fontsize=7.5, color="#444", ha="center",
            arrowprops=dict(arrowstyle="->", color="#999", lw=1))
ax.set_xlim(17.9, 7.5)
ax.set_ylim(-0.05, 1.42)
ax.set_xticks(K)
ax.set_yticks(np.arange(0, 1.01, .2))
ax.set_xlabel("units kept (walking the finished stack back, 8 probes of 1536 tokens at each k)")
ax.set_ylabel("rate")
ax.set_title("Forced-choice capability and generative coherence come apart\n"
             "ARC holds at 0.94 while the model stops ever finishing a response",
             fontsize=9.5, loc="left", pad=30)
ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
fig.savefig(FIG / "C_arc_vs_termination.png")
plt.close(fig)

# rename the mislabelled extraction: cell 12b's chart is the random floor, not the walk-back
old = FIG / "02_coherence_walkback.png"
if old.exists():
    old.replace(FIG / "02_random_floor.png")

print("wrote:")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name:32s} {p.stat().st_size/1024:6.1f} KB")
