"""Plot video-level emotion distributions from video_evaluator.py outputs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ORDER = [
    "Angry",
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Sadness",
    "Surprise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot majority-emotion distributions for model/dataset video runs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/video_results"),
        help="Directory containing run subfolders with video_summary.csv files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit video_summary.csv paths. If omitted, auto-discovers under --results-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/video_results/distribution_plots"),
        help="Where to save plots and the combined counts CSV.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Plot percentages instead of raw video counts.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="both",
        help="Output figure format.",
    )
    return parser.parse_args()


def discover_summary_csvs(results_root: Path) -> list[Path]:
    csvs = sorted(results_root.glob("*/video_summary.csv"))
    direct = results_root / "video_summary.csv"
    if direct.exists():
        csvs.insert(0, direct)
    return csvs


def run_label(summary_path: Path, results_root: Path) -> str:
    try:
        return summary_path.parent.relative_to(results_root).as_posix()
    except ValueError:
        return summary_path.parent.name


def count_emotions(summary_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "majority_emotion" not in (reader.fieldnames or []):
            raise ValueError(f"{summary_path} does not contain a majority_emotion column")
        for row in reader:
            emotion = (row.get("majority_emotion") or "").strip()
            if emotion:
                counts[emotion] += 1
    return counts


def ordered_classes(all_classes: set[str]) -> list[str]:
    known = [label for label in DEFAULT_ORDER if label in all_classes]
    extra = sorted(all_classes - set(known))
    return known + extra


def save_counts_csv(
    output_path: Path,
    run_counts: dict[str, Counter[str]],
    classes: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "total_videos", *classes])
        for run, counts in run_counts.items():
            total = sum(counts.values())
            writer.writerow([run, total, *[counts.get(label, 0) for label in classes]])


def plot_distributions(
    run_counts: dict[str, Counter[str]],
    classes: list[str],
    output_dir: Path,
    normalize: bool,
    output_format: str,
) -> None:
    n_runs = len(run_counts)
    ncols = 3 if n_runs > 2 else n_runs
    nrows = (n_runs + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (run, counts) in zip(axes_flat, run_counts.items()):
        total = sum(counts.values())
        values = [counts.get(label, 0) for label in classes]
        if normalize and total:
            values = [value / total * 100 for value in values]

        bars = ax.bar(classes, values, color="#4C78A8")
        ax.set_title(run.replace("_", " ").title(), fontsize=11, pad=8)
        ax.set_ylabel("Videos (%)" if normalize else "Videos")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

        for bar, value in zip(bars, values):
            if value == 0:
                continue
            label = f"{value:.1f}%" if normalize else str(int(value))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for ax in axes_flat[n_runs:]:
        ax.axis("off")

    metric = "Percentage" if normalize else "Count"
    fig.suptitle(f"DepVidMood Video-Level Emotion Distribution by Model/Dataset ({metric})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "video_emotion_distribution_percent" if normalize else "video_emotion_distribution_counts"
    if output_format in {"png", "both"}:
        fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if output_format in {"pdf", "both"}:
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csvs = args.summary_csv or discover_summary_csvs(args.results_root)
    if not summary_csvs:
        raise SystemExit(f"No video_summary.csv files found under {args.results_root}")

    run_counts: dict[str, Counter[str]] = {}
    for summary_path in summary_csvs:
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        run_counts[run_label(summary_path, args.results_root)] = count_emotions(summary_path)

    classes = ordered_classes(set().union(*(set(counts) for counts in run_counts.values())))
    save_counts_csv(args.output_dir / "video_emotion_distribution_counts.csv", run_counts, classes)
    plot_distributions(run_counts, classes, args.output_dir, args.normalize, args.format)

    print(f"Saved counts CSV to: {args.output_dir / 'video_emotion_distribution_counts.csv'}")
    print(f"Saved distribution plot(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
