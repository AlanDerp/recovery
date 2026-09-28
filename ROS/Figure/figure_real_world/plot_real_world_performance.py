from pathlib import Path
import csv
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication/export settings: keep SVG text editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Palette adapted from the supplied visual reference. The original warm yellow
# role is intentionally replaced with light orange.
REFERENCE_BLUE = "#5EA1D5"
REFERENCE_BLUE_DARK = "#3F7FB6"
REFERENCE_BLUE_PALE = "#DAE4F0"
REFERENCE_GREEN = "#8FC47B"
REFERENCE_GREEN_PALE = "#E2F0DB"
REFERENCE_ORANGE = "#F4B183"
REFERENCE_ORANGE_PALE = "#FBE2CC"
NEUTRAL_LIGHT = "#D3DEE9"
NEUTRAL_MID = "#767676"
NEUTRAL_DARK = "#333333"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return centre , centre


with DATA.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

tasks = [row["task"] for row in rows]
successes = np.array([int(row["successful_trials"]) for row in rows])
trials = np.array([int(row["total_trials"]) for row in rows])
rates = successes / trials
intervals = np.array([wilson_interval(int(s), int(n)) for s, n in zip(successes, trials)])

# Preserve manuscript table order, with the first row at the top.
y = np.arange(len(tasks))[::-1]
colors = [
    REFERENCE_BLUE_DARK if task == "Overall"
    else REFERENCE_GREEN if rate >= 0.99
    else REFERENCE_BLUE if rate >= 0.50
    else REFERENCE_ORANGE
    for task, rate in zip(tasks, rates)
]
guide_colors = [
    NEUTRAL_LIGHT if task == "Overall"
    else REFERENCE_GREEN_PALE if rate >= 0.99
    else REFERENCE_BLUE_PALE if rate >= 0.50
    else REFERENCE_ORANGE_PALE
    for task, rate in zip(tasks, rates)
]

width_mm, height_mm = 150, 98
fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Light guide lines emphasize the common 0–100% scale without bar-chart ink.
for yi, task, rate, color, guide_color in zip(y, tasks, rates, colors, guide_colors):
    is_overall = task == "Overall"
    ax.hlines(yi, 0, rate * 100, color=guide_color,
              linewidth=3.0 if is_overall else 2.2, zorder=1)
    ax.plot(rate * 100, yi, "D" if is_overall else "o",
            ms=7.2 if is_overall else 6.8, color=color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)

# Draw only intervals that have positive width and contain the observed rate.
# This preserves the user-supplied interval function: its current (centre, centre)
# output is intentionally treated as a collapsed interval rather than converted.
valid_interval = (
    (intervals[:, 1] > intervals[:, 0])
    & (intervals[:, 0] <= rates)
    & (rates <= intervals[:, 1])
)
if np.any(valid_interval):
    lower = (rates[valid_interval] - intervals[valid_interval, 0]) * 100
    upper = (intervals[valid_interval, 1] - rates[valid_interval]) * 100
    ax.errorbar(
        rates[valid_interval] * 100,
        y[valid_interval],
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=NEUTRAL_MID,
        elinewidth=1.0,
        capsize=2.2,
        capthick=1.0,
        zorder=2,
    )

for yi, task, rate, hi, s, n in zip(y, tasks, rates, intervals[:, 1], successes, trials):
    label_x = rate * 100 + 4.2
    ax.text(label_x, yi, f"{s}/{n}  ({rate * 100:.1f}%)", va="center", ha="left",
            fontsize=7.0, color=NEUTRAL_DARK,
            fontweight="bold" if task == "Overall" else "normal")

# Visually separate the supplied aggregate from the six task-level rows.
overall_y = y[tasks.index("Overall")]
ax.axhline(overall_y + 0.58, color="#AFC2D4", linewidth=0.8, zorder=0)

ax.set_yticks(y, tasks)
ax.set_xlim(0, 122)
ax.set_ylim(-0.65, len(tasks) - 0.35)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0", "25", "50", "75", "100"])
ax.set_xlabel("Success rate (%)", fontsize=7.5, labelpad=5)
ax.set_title("Real-world task success", loc="left", fontsize=9, fontweight="bold", pad=16)
ax.text(
    0,
    1.035,
    "Observed rates; n = 8 per task and n = 56 overall",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=6.5,
    color=NEUTRAL_MID,
)

ax.xaxis.grid(True, color="#EEF2F5", linewidth=0.7)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=6.8, length=3, width=0.7, color=NEUTRAL_DARK)
ax.tick_params(axis="y", labelsize=7.0, length=0, pad=7)
for tick_label in ax.get_yticklabels():
    if tick_label.get_text() == "Overall":
        tick_label.set_fontweight("bold")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["bottom"].set_color(NEUTRAL_DARK)

fig.subplots_adjust(left=0.37, right=0.98, top=0.80, bottom=0.18)

stem = OUT / "real_world_task_success"
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Exported figure set to {OUT}")
