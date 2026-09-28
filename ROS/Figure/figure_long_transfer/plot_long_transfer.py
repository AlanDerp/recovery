from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication/export settings: preserve editable SVG/PDF text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Restrained, color-blind-friendly palette with redundant shape/hatch encoding.
TASK_BLUE = "#B7D8EC"
TASK_BLUE_LINE = "#6FAFD6"
SWAP_ORANGE = "#F7C9A8"
SWAP_ORANGE_LINE = "#EFA36D"
GRID = "#E7E9EC"
TEXT = "#2F3337"


def load_and_validate() -> list[dict[str, float | int]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, float | int]] = []
    for raw in raw_rows:
        row = {
            "retention": int(raw["knowledge_retention_percent"]),
            "task_successes": int(raw["task_successes"]),
            "task_trials": int(raw["task_trials"]),
            "task_rate": float(raw["task_success_rate_percent"]),
            "swap_successes": int(raw["swap_successes"]),
            "swap_trials": int(raw["swap_trials"]),
            "swap_rate": float(raw["swap_success_rate_percent"]),
            "long_successes": int(raw["long_successes"]),
            "long_trials": int(raw["long_trials"]),
            "long_rate": float(raw["long_success_rate_percent"]),
        }

        expected_long_successes = row["task_successes"] + row["swap_successes"]
        expected_long_trials = row["task_trials"] + row["swap_trials"]
        if row["long_successes"] != expected_long_successes:
            raise ValueError(f"Long successes do not sum at {row['retention']}% retention")
        if row["long_trials"] != expected_long_trials:
            raise ValueError(f"Long trials do not sum at {row['retention']}% retention")

        for prefix in ("task", "swap", "long"):
            computed = 100 * row[f"{prefix}_successes"] / row[f"{prefix}_trials"]
            if not np.isclose(computed, row[f"{prefix}_rate"], atol=0.051):
                raise ValueError(f"{prefix.title()} rate mismatch at {row['retention']}% retention")
        rows.append(row)

    if [row["retention"] for row in rows] != [0, 25, 50, 75, 100]:
        raise ValueError("Unexpected knowledge-retention categories or order")
    return rows


def annotate_bars(ax: mpl.axes.Axes, containers: list[mpl.container.BarContainer]) -> None:
    for container in containers:
        for bar in container:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value + 0.9, 1.0),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=TEXT,
                clip_on=False,
            )


def main() -> None:
    rows = load_and_validate()
    x = np.arange(len(rows), dtype=float)
    retention = np.array([row["retention"] for row in rows])
    task = np.array([row["task_rate"] for row in rows])
    swap = np.array([row["swap_rate"] for row in rows])
    long = np.array([row["long_rate"] for row in rows])

    # Single-column Nature-style canvas.
    width_mm, height_mm = 89, 67
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_width = 0.32
    task_bars = ax.bar(
        x - bar_width / 2,
        task,
        width=bar_width,
        color=TASK_BLUE,
        edgecolor=TASK_BLUE_LINE,
        linewidth=0.6,
        label="Task",
        zorder=2,
    )
    swap_bars = ax.bar(
        x + bar_width / 2,
        swap,
        width=bar_width,
        color=SWAP_ORANGE,
        edgecolor=SWAP_ORANGE_LINE,
        linewidth=0.6,
        hatch="///",
        label="Swap",
        zorder=2,
    )

    # Each line is sampled at the exact geometric center of its own bars.
    task_line, = ax.plot(
        x - bar_width / 2,
        task,
        color=TASK_BLUE_LINE,
        linewidth=1.25,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgecolor=TASK_BLUE_LINE,
        markeredgewidth=0.9,
        label="Task",
        clip_on=False,
        zorder=4,
    )
    swap_line, = ax.plot(
        x + bar_width / 2,
        swap,
        color=SWAP_ORANGE_LINE,
        linewidth=1.25,
        marker="s",
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=SWAP_ORANGE_LINE,
        markeredgewidth=0.9,
        label="Swap",
        clip_on=False,
        zorder=4,
    )

    annotate_bars(ax, [task_bars, swap_bars])

    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_ylim(0, 48)
    ax.set_xticks(x, [f"{value}%" for value in retention])
    ax.set_yticks(np.arange(0, 49, 10))
    ax.set_xlabel("Knowledge retention ratio", fontsize=7.0, labelpad=5)
    ax.set_ylabel("Success rate (%)", fontsize=7.0, labelpad=5)

    ax.yaxis.grid(True, color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.2, length=2.5, width=0.7, color=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT)
        ax.spines[side].set_linewidth(0.75)

    handles = [task_line, swap_line]
    labels = ["Task", "Swap"]
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        fontsize=6.1,
        handlelength=1.8,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0,
        frameon=False,
    )
    ax.text(
        0.5,
        1.045,
        "Observed rates; 500 trials per bar",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#676C72",
    )

    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.20, top=0.79)
    stem = OUT / "long_transfer_retention"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)

    print(f"Exported figure set to {OUT}")


if __name__ == "__main__":
    main()
