"""results/outcomes.png: what each checker does with the bad and the good test photos, from results/summary.json."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"rules": "Rules + CLIP", "haiku-4.5": "Haiku 4.5", "sonnet-5.5": "Sonnet 5.5",
         "both-haiku": "Rules + Haiku", "both-sonnet": "Rules + Sonnet",
         "strict-haiku": "Strict Rules + Haiku*", "strict-sonnet": "Strict Rules + Sonnet*"}  # fmt: skip
RIGHT, PERSON, WRONG = "#2a78d6", "#a9a8a2", "#eb6834"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _panel(ax, rows, total, title):
    for i, (_, segs) in enumerate(rows):
        left = 0
        for value, color, _ in segs:
            if value:
                ax.barh(i, value, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
                if value >= 2:
                    ax.text(left + value / 2, i, str(value), ha="center", va="center", fontsize=9,
                            color="white" if color != PERSON else INK)  # fmt: skip
            left += value
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.set_xlim(0, total)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)


def draw(summary: dict, path: Path = ROOT / "results" / "outcomes.png") -> None:
    arms = list(NAMES)
    g = {a: summary["arms"][a]["gate"] for a in arms}
    nb, ng = g["rules"]["bad_photos"], g["rules"]["good_photos"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE, sharey=True)
    _panel(axes[0], [(NAMES[a], [(g[a]["bad_flagged"], RIGHT, ""), (g[a]["bad_to_person"], PERSON, ""),
                                 (g[a]["bad_let_through"], WRONG, "")]) for a in arms], nb,
           f"{nb} bad test photos (blur, exposure, tilt)")  # fmt: skip
    _panel(axes[1], [(NAMES[a], [(g[a]["good_ready"], RIGHT, ""), (g[a]["good_to_person"], PERSON, ""),
                                 (g[a]["good_flagged"], WRONG, "")]) for a in arms], ng,
           f"{ng} good test photos")  # fmt: skip
    axes[0].invert_yaxis()  # shared y: invert once
    for ax in axes:
        ax.set_facecolor(SURFACE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RIGHT, PERSON, WRONG)]
    fig.legend(handles, ["right call (bad: flagged · good: ready)", "sent to a person (can't tell)",
                         "wrong call (bad: passed as ready · good: flagged)"],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
