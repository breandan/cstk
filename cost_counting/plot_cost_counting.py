"""Render the decoder benchmark CSV without rerunning the benchmark.

Examples:
    python plot_cost_counting.py
    python plot_cost_counting.py --metric construction
    python plot_cost_counting.py --metric total

The CSV contains one measured row per decoder, bin count, and slice length.
Incomplete batches remain visible as hollow markers; they are never scaled
to an estimated 1,000-decode time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields as dataclass_fields
import math
from pathlib import Path


@dataclass(frozen=True)
class Measurement:
    decoder: str
    bins: int
    length: int
    available_count: int
    decoded_count: int
    requested_count: int
    prepare_seconds: float
    decode_seconds: float
    total_seconds: float
    repeats: int = 1


def read_measurements(csv_path: str | Path) -> list[Measurement]:
    """Read measured values, rejecting malformed or ambiguous rows."""
    fields = {field.name for field in dataclass_fields(Measurement)} - {"repeats"}
    measurements = []
    seen = set()
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                measurement = Measurement(
                    decoder=row["decoder"],
                    bins=int(row["bins"]),
                    length=int(row["length"]),
                    available_count=int(row["available_count"]),
                    decoded_count=int(row["decoded_count"]),
                    requested_count=int(row["requested_count"]),
                    prepare_seconds=float(row["prepare_seconds"]),
                    decode_seconds=float(row["decode_seconds"]),
                    total_seconds=float(row["total_seconds"]),
                    repeats=int(row.get("repeats") or 1),
                )
                if measurement.decoder not in {"fft", "ordinary"}:
                    raise ValueError("decoder must be 'fft' or 'ordinary'")
                if not 1 <= measurement.bins <= 10 or measurement.length < 1:
                    raise ValueError("bins must be 1–10 and length must be positive")
                if measurement.requested_count < 1:
                    raise ValueError("requested_count must be positive")
                if measurement.repeats < 1:
                    raise ValueError("repeats must be positive")
                if not 0 <= measurement.decoded_count <= min(
                    measurement.available_count, measurement.requested_count
                ):
                    raise ValueError("decoded_count exceeds the available or requested count")
                timings = (
                    measurement.prepare_seconds,
                    measurement.decode_seconds,
                    measurement.total_seconds,
                )
                if any(not math.isfinite(value) or value < 0 for value in timings):
                    raise ValueError("timings must be finite and nonnegative")
                key = measurement.decoder, measurement.bins, measurement.length
                if key in seen:
                    raise ValueError(f"duplicate measurement for {key}")
                seen.add(key)
                measurements.append(measurement)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid CSV row {line}: {error}") from error
    if not measurements:
        raise ValueError("CSV has no measurements")
    return measurements


def make_plots(
    csv_path: str | Path = "benchmark_results/cost_counting.csv",
    output_dir: str | Path = "benchmark_results",
    metric: str = "decode",
    grammar_name: str = "python.cnf",
) -> list[Path]:
    """Save separate FFT and ordinary plots, each as PNG and editable SVG."""
    if metric not in {"decode", "total", "construction"}:
        raise ValueError("metric must be 'decode', 'total', or 'construction'")
    measurements = read_measurements(csv_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    construction = metric == "construction"
    metric_field = "prepare_seconds" if construction else f"{metric}_seconds"
    maximum_ms = max(getattr(row, metric_field) * 1_000 for row in measurements)
    shared_ylim = (0, max(maximum_ms * 1.10, 0.001))
    lengths = sorted({row.length for row in measurements})
    x_margin = max((lengths[-1] - lengths[0]) * 0.025, 0.5)
    shared_xlim = (lengths[0] - x_margin, lengths[-1] + x_margin)
    requested_counts = {row.requested_count for row in measurements}
    requested_label = (
        f"{next(iter(requested_counts)):,}"
        if len(requested_counts) == 1
        else "the requested number of"
    )
    colors = (
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    )
    paths = []

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#243247",
        "axes.labelcolor": "#243247",
        "xtick.color": "#526071",
        "ytick.color": "#526071",
        "axes.edgecolor": "#CED5DE",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        for decoder, title in (("fft", "FFT decoder"), ("ordinary", "Ordinary decoder")):
            fig, ax = plt.subplots(figsize=(10.4, 6.7), facecolor="white")
            fig.subplots_adjust(left=0.105, right=0.965, bottom=0.255, top=0.775)
            if construction:
                title = f"{'FFT' if decoder == 'fft' else 'Ordinary'} parser · Slice construction"
            fig.text(0.105, 0.938, title, fontsize=22, weight="bold", va="top")
            subtitle = f"{grammar_name}  ·  Exact-length slices"
            if not construction:
                subtitle += f"  ·  First up to {requested_label} ranked derivations"
            fig.text(
                0.105,
                0.872,
                subtitle,
                fontsize=11,
                va="top",
            )
            description = {
                "decode": "Batch decoding after parsing; chart construction excluded.",
                "total": "Chart construction plus batch decoding.",
                "construction": "Chart, forest, counts and decoding index; decoding excluded.",
            }[metric]
            fig.text(0.105, 0.832, description, fontsize=10, color="#647184", va="top")

            decoder_rows = [row for row in measurements if row.decoder == decoder]
            for bins in range(1, 11):
                rows = sorted(
                    (row for row in decoder_rows if row.bins == bins),
                    key=lambda row: row.length,
                )
                if not rows:
                    continue
                color = colors[bins - 1]
                x = [row.length for row in rows]
                y = [getattr(row, metric_field) * 1_000 for row in rows]
                ax.plot(x, y, color=color, linewidth=1.8, alpha=0.92, zorder=2)
                for complete in (False, True):
                    points = [
                        (xx, yy)
                        for row, xx, yy in zip(rows, x, y)
                        if (construction or row.decoded_count >= row.requested_count) == complete
                    ]
                    if points:
                        ax.scatter(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            s=28,
                            facecolors=color if complete else "white",
                            edgecolors=color,
                            linewidths=1.25,
                            zorder=3,
                        )
            if not decoder_rows:
                ax.text(0.5, 0.5, "No measurements available", transform=ax.transAxes, ha="center")

            ax.set(xlim=shared_xlim, ylim=shared_ylim)
            ax.set_xlabel("Slice length (tokens)", labelpad=11)
            ax.set_ylabel(
                {"decode": "Batch decode time (ms)",
                 "total": "Chart + batch decode time (ms)",
                 "construction": "Slice construction time (ms)"}[metric],
                labelpad=12,
            )
            if len(lengths) <= 16:
                ax.set_xticks(lengths)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.tick_params(axis="both", length=0, pad=7)
            ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
            handles = [
                Line2D(
                    [0], [0], color=colors[bins - 1], linewidth=2,
                    label=f"{bins} {'bin' if bins == 1 else 'bins'}",
                )
                for bins in range(1, 11)
            ]
            fig.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.535, 0.090),
                ncol=5, frameon=False, columnspacing=2.0, handlelength=2.4,
                handletextpad=0.7, labelspacing=0.8,
            )
            if construction:
                note = "Cumulative chart work through each length + that slice's forest, counts and provenance."
                detail = "One construction measurement per point. Grammar loading excluded. Shared axes."
            else:
                note = f"Filled markers: full batch.  Hollow markers: fewer than {requested_label} decoded."
                actual_counts = [row.decoded_count for row in decoder_rows]
                repetitions = {row.repeats for row in decoder_rows}
                detail = (f"Actual batches: {min(actual_counts):,}–{max(actual_counts):,} derivations.  "
                          if actual_counts else "")
                if len(repetitions) == 1:
                    detail += f"Median of {next(iter(repetitions))} runs.  "
                detail += "No padding or extrapolation. Shared axes."
            fig.text(0.105, 0.065, note, fontsize=9, color="#647184")
            fig.text(
                0.105, 0.037,
                detail,
                fontsize=9, color="#647184",
            )
            for suffix in ("png", "svg"):
                stem = (f"slice_construction_times_{decoder}" if construction
                        else f"cost_counting_{decoder}_{metric}")
                path = output_dir / f"{stem}.{suffix}"
                fig.savefig(path, dpi=180)
                paths.append(path)
            plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results/cost_counting.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--metric", choices=("decode", "total", "construction"), default="decode")
    parser.add_argument("--grammar-name", default="python.cnf")
    args = parser.parse_args()
    for path in make_plots(args.csv, args.output_dir, args.metric, args.grammar_name):
        print(path)


if __name__ == "__main__":
    main()
