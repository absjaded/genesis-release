"""Plot the actual 16-cohort shared-component replay.

Run after installing the package with its optional [figures] dependency.
Outputs a static SVG by default; no empirical feature matrices are required.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np

from genesis_core.experiments import construction


def plot(output):
    rows = construction()["cohorts"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "path", "svg.hashsalt": "genesis-construction",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    fig.patch.set_facecolor("white")
    variants = ("raw", "covariance_only", "cross_degree")
    labels = ("Raw shared\nfeatures", "Covariance\ncorrection", "Removal +\ncorrection")
    panels = (
        ("cross_moment_max_abs", "Linear coupling to latent controls",
         r"$\max\,|Q^\top X/(n-1)|$"),
        ("covariance_max_abs_error", "Deviation from target covariance",
         r"$\max\,|X^\top X/(n-1)-D|$"),
    )
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for n, color, marker, offset in (
            (512, "#136b94", "o", -.07), (768, "#b75b27", "^", .07)
        ):
            group = sorted((r for r in rows if r["n"] == n), key=lambda r: r["replica"])
            for j, variant in enumerate(variants):
                values = [r["variants"][variant][metric] for r in group]
                if any(v <= 0 for v in values):
                    raise ValueError("log plot requires strictly positive measured errors")
                ax.scatter(j + offset + np.linspace(-.035, .035, len(group)),
                           values, color=color, marker=marker, s=30, alpha=.8,
                           label=f"n = {n} (8 cohorts)" if j == 0 else None,
                           edgecolors="white", linewidths=.35, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-19, 1)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(range(3), labels)
        ax.set_xlim(-.45, 2.45)
        ax.set_title(title, loc="left", fontsize=12, pad=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(axis="x", length=0, pad=8)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Covariance correction alone leaves finite-sample coupling",
                 x=.065, ha="left", fontsize=15, fontweight="bold")
    fig.text(.065, .025,
             "All 16 historical shared components regenerated. Each point is one cohort; lower is better.\n"
             "Shared-feature diagnostics under raw, covariance-only and joint orthogonality/covariance constraints.",
             fontsize=9, color="#465260")
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.24, wspace=.32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, metadata={"Date": None} if output.suffix.lower() == ".svg" else {})
    plt.close(fig)
    print(f"Plotted {len(rows)} regenerated cohorts: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/figures/construction.svg")
    plot(parser.parse_args().output)
