#!/usr/bin/env python3
"""Measure exact-length CNF slices with 1..10 retained cost bins.

Run with the project's Python interpreter. Requires NumPy and Matplotlib.
The default experiment parses python.cnf slices of 10,20,...,100 tokens,
then times decoding ranks 0..min(999, retained_count-1). It does not repeat
ranks to pad short slices, deduplicate ambiguous derivations, or extrapolate
timings to 1,000 outputs. The plotted decode time excludes preparation.
Separate slice_construction_times plots show preparation in milliseconds.

Both backends use a common exact first-K cost chart to avoid constructing
irrelevant forest edges. Each builds its own CostIndex using its own
convolution implementation. Charts grow incrementally within each bin cap;
prepare_seconds records cumulative chart work plus this slice's index
construction, excluding indexes previously constructed for shorter slices.
The chart has no cross-bin or cross-backend cache. Decoding is unchanged.

CSV rows are flushed after each point. JSON includes individual repetitions,
source hashes, and workload details. Replot without rerunning using
plot_cost_counting.py --csv benchmark_results/cost_counting.csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

from cost_counting_benchmark_chart import CNFTopKChart


FIELDS = ("decoder", "bins", "length", "available_count", "decoded_count",
          "requested_count", "repeats", "prepare_seconds", "decode_seconds",
          "total_seconds", "decode_min_seconds", "decode_max_seconds",
          "chart_seconds", "index_seconds", "fft_forward_transforms", "output_sha256")


def positive_list(value):
    try:
        result = sorted(set(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def digest_outputs(index, outputs):
    """Check costs and token yields outside the timed region."""
    digest = hashlib.sha256()
    for rank, output in enumerate(outputs):
        data = json.dumps([index.locate(rank)[0], output], ensure_ascii=False).encode()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def decode_batch(backend, index, count):
    return [index.derivation(rank, backend.cnf_tokens) for rank in range(count)]


def compile_measured(backend, chart, length):
    """Count actual FFT use; the guarded variant can choose sparse products."""
    fft = getattr(getattr(backend, "np", None), "fft", None)
    original = fft.rfft if fft is not None else None
    calls = 0

    def counted_rfft(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    if fft is not None:
        fft.rfft = counted_rfft
    try:
        started = perf_counter()
        forest, index = chart.compile_length(length)
        seconds = perf_counter() - started
    finally:
        if fft is not None:
            fft.rfft = original
    return forest, index, seconds, calls


def measure_decode(backend, index, count, repeats, length):
    # A few actual queries warm the decoding path; no answers are memoized.
    for rank in range(min(count, 3)):
        index.derivation(rank, backend.cnf_tokens)
    times = []
    expected_digest = None
    for _ in range(repeats):
        gc.collect()
        started = perf_counter()
        outputs = decode_batch(backend, index, count)
        times.append(perf_counter() - started)
        if any(len(output.split()) != length for output in outputs):
            raise AssertionError("decoder returned a yield outside the exact-length slice")
        digest = digest_outputs(index, outputs)
        if expected_digest is not None and digest != expected_digest:
            raise AssertionError("decoding changed across repetitions")
        expected_digest = digest
        del outputs
    return times, expected_digest


def benchmark(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "cost_counting.csv"
    json_path = output / "cost_counting.json"
    names = {"fft": "cost_counting_fft", "ordinary": "cost_counting"}
    backends = {label: importlib.import_module(name) for label, name in names.items()}
    grammars = {label: backend.load_cnf(args.grammar, args.start)
                for label, backend in backends.items()}
    records = []
    metadata = {
        "complete": False,
        "grammar": str(args.grammar.resolve()),
        "grammar_sha256": hashlib.sha256(args.grammar.read_bytes()).hexdigest(),
        "start": args.start, "scale": args.scale,
        "lengths": args.lengths, "bins": args.bins,
        "requested_count": args.samples, "repeats": args.repeats,
        "python": sys.version, "platform": platform.platform(),
        "workload": "All grammar derivations of exactly N tokens; ranks ordered by integer cost.",
        "sampling": "Decode ranks 0..min(requested_count, retained_count)-1; no padding or deduplication.",
        "decode_seconds": "Median batch time after parsing; includes token-string rendering and list collection.",
        "prepare_seconds": "Cumulative chart work through N plus index/provenance for N; grammar loading excluded.",
        "preparation": "Common incremental first-K cost chart; backend-specific exact counting and provenance.",
        "source_sha256": {Path(backend.__file__).name: hashlib.sha256(
            Path(backend.__file__).read_bytes()).hexdigest() for backend in backends.values()},
        "records": records,
    }
    for filename in ("cost_counting_benchmark_chart.py", "benchmark_cost_counting.py"):
        path = Path(__file__).with_name(filename)
        metadata["source_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    started_all = perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for bins in args.bins:
            charts, elapsed = {}, {}
            for label, backend in backends.items():
                started = perf_counter()
                charts[label] = CNFTopKChart(backend, grammars[label], bins, scale=args.scale)
                elapsed[label] = perf_counter() - started
            for point, length in enumerate(args.lengths):
                # Alternate order to avoid always warming one backend first.
                order = tuple(backends) if (point + bins) % 2 else tuple(reversed(backends))
                reference = None
                for label in order:
                    backend, chart = backends[label], charts[label]
                    gc.collect()
                    started = perf_counter()
                    chart.extend(length)
                    elapsed[label] += perf_counter() - started
                    forest, index, index_seconds, fft_calls = compile_measured(backend, chart, length)
                    count = min(args.samples, index.total)
                    times, digest = measure_decode(backend, index, count, args.repeats, length)
                    # Verify every retained root coefficient and every decoded
                    # cost/yield before accepting a paired benchmark point.
                    signature = (tuple(index.H[index.start].items()), count, digest)
                    if reference is not None and reference != signature:
                        raise AssertionError(f"backends disagree: length={length}, bins={bins}")
                    reference = signature
                    prepare = elapsed[label] + index_seconds
                    decode = median(times)
                    row = dict(decoder=label, bins=bins, length=length,
                               available_count=index.total, decoded_count=count,
                               requested_count=args.samples, repeats=args.repeats,
                               prepare_seconds=prepare, decode_seconds=decode,
                               total_seconds=prepare + decode,
                               decode_min_seconds=min(times), decode_max_seconds=max(times),
                               chart_seconds=elapsed[label], index_seconds=index_seconds,
                               fft_forward_transforms=fft_calls,
                               output_sha256=digest)
                    writer.writerow(row)
                    stream.flush()
                    records.append({**row, "decode_repetitions_seconds": times})
                    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                    print(f"{label:8} bins={bins:2} length={length:3} "
                          f"decoded={count:4}/{args.samples} "
                          f"prepare={prepare:.3f}s decode={decode * 1000:.3f}ms", flush=True)
                    del forest, index
            del charts
    metadata["complete"] = True
    metadata["elapsed_seconds"] = perf_counter() - started_all
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path, default=Path(__file__).with_name("python.cnf"))
    parser.add_argument("--start", default="START")
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--lengths", type=positive_list, default=list(range(10, 101, 10)))
    parser.add_argument("--bins", type=positive_list, default=list(range(1, 11)))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--plot-metric", choices=("decode", "total", "construction"), default="decode",
                        help="primary plots; slice-construction plots are also always generated")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if min(args.scale, args.samples, args.repeats) <= 0:
        parser.error("scale, samples, and repeats must be positive")
    if not args.no_plot and args.bins[-1] > 10:
        parser.error("plots support bins 1–10; use --no-plot for larger bin caps")
    csv_path = benchmark(args)
    print(f"Saved {csv_path}")
    if not args.no_plot:
        from plot_cost_counting import make_plots
        paths = make_plots(csv_path, args.output_dir, metric=args.plot_metric,
                           grammar_name=args.grammar.name)
        if args.plot_metric != "construction":
            paths += make_plots(csv_path, args.output_dir, metric="construction",
                                grammar_name=args.grammar.name)
        for path in paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
