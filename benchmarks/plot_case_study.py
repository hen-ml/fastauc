"""Charts for benchmarks/case_study.py results.

    python benchmarks/plot_case_study.py case_study_istella_s.json docs/img/case_study
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # categorical slot 1
ORANGE = "#eb6834"   # categorical slot 2


def log_rounds(ax, upto):
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= upto]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(path, prefix):
    r = json.load(open(path))
    b = r["curves"]["lightgbm_builtin"]
    fa = np.asarray(r["curves"]["fastauc"])
    first, orig, last = (np.asarray(b[o]) for o in ("relevant_first", "original", "relevant_last"))
    it = np.arange(1, len(fa) + 1)
    n = min(len(fa), 200)

    # 1. LightGBM's own NDCG@10 under three row orders, against fastauc.
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.fill_between(it[:n], last[:n], first[:n], color=BLUE, alpha=0.18, linewidth=0,
                    label="LightGBM built-in, range over row orders")
    ax.plot(it[:n], first[:n], color=BLUE, linewidth=1.2, alpha=0.7)
    ax.plot(it[:n], last[:n], color=BLUE, linewidth=1.2, alpha=0.7)
    ax.plot(it[:n], orig[:n], color=ORANGE, linewidth=2, label="LightGBM built-in, rows as distributed")
    ax.plot(it[:n], fa[:n], color=INK, linewidth=2, linestyle=(0, (4, 2)), label="fastauc (ties averaged)")
    beaten = int(np.argmax(first[1:] > first[0])) + 2 if np.any(first[1:] > first[0]) else None
    if beaten and beaten > 2:
        ax.hlines(first[0], 1, beaten, colors=INK_2, linestyles=(0, (1, 2)), linewidth=1)
        ax.annotate(f"Relevant rows first: round 1 scores {first[0]:.3f},\n"
                    f"not beaten until round {beaten}",
                    xy=(1, first[0]), xytext=(1.25, first[0] + 0.03), fontsize=9, color=INK)
    log_rounds(ax, n)
    ax.set_xlim(1, n)
    ax.set_xlabel("Boosting round (log scale)", color=INK_2, fontsize=10)
    ax.set_ylabel("Validation NDCG@10", color=INK_2, fontsize=10)
    ax.set_title(f"{r['dataset']}: the same model, scored three ways", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{prefix}_ndcg.png", facecolor=SURFACE)

    # 2. How often a tie can change NDCG@10.
    rows = r["checkpoints"]
    x = np.array([c["iteration"] for c in rows])
    share = np.array([c["queries_with_mixed_tie_in_top_k"] for c in rows]) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.plot(x, share, color=BLUE, linewidth=2, marker="o", markersize=5)
    for xi, yi in zip(x, share):
        if xi in (x[0], x[len(x) // 2], x[-1]):
            ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=INK)
    log_rounds(ax, x.max())
    ax.set_ylim(0, max(5, share.max() * 1.2))
    ax.set_xlabel("Boosting round (log scale)", color=INK_2, fontsize=10)
    ax.set_ylabel("% of validation queries", color=INK_2, fontsize=10)
    ax.set_title("Queries where a tie mixes relevance grades inside the top 10", color=INK,
                 fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}_ties.png", facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
