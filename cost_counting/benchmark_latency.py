#!/usr/bin/env python3
"""Latency of full cost-index construction and decoding with four bin maps.

Usage: python benchmark_latency.py [grammar.cnf] [10..100:10]
Edit the globals below for sampling, binning, repetition, and resource budgets.
Each slice gets a fresh chart and provenance index. Decode times are medians
of warmed batches, including rendering; loading and validation are untimed.
All maps retain every derivation, using phi(cost + phi(left + right)) for
binary products. Failures are recorded without substituting partial indexes.
Results and standalone plots are saved in OUTPUT_DIR; see README for details.
"""
from __future__ import annotations
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import platform
import resource
import signal
import sys
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from itertools import accumulate
from multiprocessing.connection import Connection
from pathlib import Path
from statistics import median
from time import perf_counter
from types import FrameType
from typing import Any
import cost_counting as backend
from benchmark_inputs import load_grammar, parse_inputs

ROOT = Path(__file__).resolve().parent
GRAMMAR_PATH = ROOT / "python.cnf"
START: str | None = None              # Auto-select START, S, or the first rule head.
COST_SCALE = 1000
LENGTHS = tuple(range(10, 101, 10))
BINNINGS = ("identity", "uniform", "truncated", "geometric")
UNIFORM_WIDTH = 1000
TRUNCATED_BINS = 4                     # Four finite bins plus overflow.
GEOMETRIC_RATIO = 2                    # Integer >=2: exact boundary arithmetic.
SAMPLES = 1000
REPEATS = 3
MAX_PREPARE_SECONDS = 120             # A failed attempt is recorded, never approximated.
MAX_PEAK_MEMORY_GIB = 16              # Same per-process budget for every map/length.
OUTPUT_DIR = ROOT / "benchmark_results" / "latency"
PLOT_DPI = 180

Cost = int | float                    # Infinity is the overflow bin.
Record = dict[str, Any]
Counts = list[dict[str, int]]

FIELDS = ("decoder", "binning", "length", "status", "failure_phase", "root_bins",
          "available_count", "full_count", "decoded_count", "requested_count",
          "repeats", "prepare_seconds", "decode_seconds", "total_seconds",
          "chart_count_seconds", "provenance_seconds", "provenance_blocks",
          "decode_min_seconds", "decode_max_seconds", "output_sha256",
          "failure_elapsed_seconds", "peak_memory_bytes", "error")


def bin_map(kind: str, width: int = UNIFORM_WIDTH,
            finite_bins: int = TRUNCATED_BINS,
            ratio: int = GEOMETRIC_RATIO) -> Callable[[Cost], Cost]:
    """Sec. 2 maps, extended by phi(infinity)=infinity."""
    if kind not in BINNINGS or min(width, finite_bins) < 1 or not isinstance(ratio, int) or ratio < 2:
        raise ValueError("Invalid binning configuration")
    @lru_cache(maxsize=None)
    def phi(t: Cost) -> Cost:
        if t == math.inf: return math.inf
        if kind == "identity": return t
        if kind == "uniform": return t // width
        if kind == "truncated": return t // width if t < finite_bins*width else math.inf
        # floor(log_rho(1+(rho-1)*t/width)), with exact integer thresholds.
        value, exponent = 1+(ratio-1)*t//width, 0
        while value >= ratio: value //= ratio; exponent += 1
        return exponent
    return phi


class BinnedChart:
    """Full histograms with grammar provenance; preserve overflow."""
    def __init__(self, grammar: backend.Forest, binning: str,
                 scale: int = COST_SCALE, width: int = UNIFORM_WIDTH,
                 finite_bins: int = TRUNCATED_BINS,
                 ratio: int = GEOMETRIC_RATIO) -> None:
        self.grammar, self.phi = grammar, bin_map(binning, width, finite_bins, ratio)
        self.rules, self.pairs = defaultdict(list), defaultdict(list)
        self.H = [defaultdict(dict), defaultdict(dict)]
        for edge in grammar.edges:
            c = edge.cost if edge.cost is not None else backend.quantize_cost(edge.weight, scale)
            self.rules[edge.head].append((edge, c))
            if len(edge.tails) == 1:
                row = self.H[1][edge.head]; row[c] = row.get(c, 0)+1
            else: self.pairs[edge.tails].append((edge.head, c))
        self.H[1] = {a: dict(sorted(h.items())) for a, h in self.H[1].items()}

    def extend(self, length: int) -> None:
        phi = self.phi
        while len(self.H) <= length:
            n = len(self.H); row = defaultdict(dict)
            # Share convolution across productions with identical children.
            for (left, right), parents in self.pairs.items():
                for split in range(1, n):
                    a, b = self.H[split].get(left, {}), self.H[n-split].get(right, {})
                    if not a or not b: continue
                    product = defaultdict(int)
                    for x, nx in a.items():
                        for y, ny in b.items(): product[phi(x+y)] += nx*ny
                    for head, cost in parents:
                        out = row[head]
                        for s, count in product.items():
                            target = phi(cost+s); out[target] = out.get(target, 0)+count
            self.H.append({a: dict(sorted(h.items())) for a, h in row.items()})

    def compile_length(self, length: int) -> BinnedIndex:
        self.extend(length)
        return BinnedIndex(self, length)


class BinnedIndex(backend.CostIndex):
    """Build native cumulative provenance buckets before any decoding query."""
    def __init__(self, chart: BinnedChart, length: int) -> None:
        started = perf_counter()
        self.start, self.H, self.buckets = (chart.grammar.start, length), {}, {}
        pending, phi = [self.start], chart.phi
        while pending:
            state = pending.pop()
            if state in self.H: continue
            head, n = state
            self.H[state] = hist = chart.H[n].get(head, {})
            self.buckets[state] = buckets = {c: backend.Bucket(count) for c, count in hist.items()}
            for edge, cost in chart.rules[head]:
                if len(edge.tails) == 1:
                    if n != 1 or cost not in buckets: continue
                    terminal = edge.tails[0]
                    if terminal not in self.H:
                        self.H[terminal] = {0: 1}
                        leaf = backend.Bucket(1)
                        leaf.add(backend.Edge(terminal, (), cost=0), (), 1)
                        self.buckets[terminal] = {0: leaf}
                    production = backend.Edge(state, (terminal,), edge.weight, edge.cost)
                    buckets[cost].add(production, (0,), 1)
                    continue
                left, right = edge.tails
                for split in range(1, n):
                    production = None
                    for a, na in chart.H[split].get(left, {}).items():
                        for b, nb in chart.H[n-split].get(right, {}).items():
                            target = phi(cost+phi(a+b))
                            if target not in buckets: continue
                            if production is None:
                                tails = ((left, split), (right, n-split))
                                production = backend.Edge(state, tails, edge.weight, edge.cost)
                                pending.extend(t for t in tails if t not in self.H)
                            buckets[target].add(production, (a, b), na*nb)
            for bucket in buckets.values():
                if not bucket.ends or bucket.ends[-1] != bucket.count:
                    raise ArithmeticError("Histogram and provenance disagree")
        self.root_costs = tuple(self.H[self.start])
        self.root_ends = tuple(accumulate(self.H[self.start].values()))
        self.total = self.root_ends[-1] if self.root_ends else 0
        self.provenance_seconds = perf_counter()-started


def full_counts(grammar: backend.Forest, length: int) -> Counts:
    """Independent unbinned inside counts, outside timed preparation."""
    H = [defaultdict(int) for _ in range(length+1)]; binary = []
    for edge in grammar.edges:
        if len(edge.tails) == 1: H[1][edge.head] += 1
        else: binary.append(edge)
    for n in range(2, length+1):
        for edge in binary:
            b, c = edge.tails
            H[n][edge.head] += sum(H[i].get(b, 0)*H[n-i].get(c, 0) for i in range(1, n))
    return [{str(node): count for node, count in row.items()} for row in H]


def digest_outputs(index: BinnedIndex, outputs: list[str]) -> str:
    digest = hashlib.sha256()
    for rank, output in enumerate(outputs):
        cost = index.locate(rank)[0]
        data = json.dumps(["overflow" if cost == math.inf else cost, output]).encode()
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def measure_decode(index: BinnedIndex, count: int, length: int,
                   repeats: int) -> tuple[list[float], str | None]:
    for rank in range(min(count, 3)): index.derivation(rank, backend.cnf_tokens)
    times, expected = [], None
    for _ in range(repeats):
        gc.collect(); started = perf_counter()
        outputs = [index.derivation(rank, backend.cnf_tokens) for rank in range(count)]
        times.append(perf_counter()-started)
        if any(len(w.split()) != length for w in outputs): raise AssertionError("Wrong yield length")
        digest = digest_outputs(index, outputs)
        if expected is not None and expected != digest: raise AssertionError("Unstable decoding")
        expected = digest
    return times, expected


def empty_record(name: str, length: int, settings: Record, totals: Counts) -> Record:
    row = dict.fromkeys(FIELDS)
    row.update(decoder="ordinary", binning=name, length=length, status="error",
               full_count=totals[length].get(settings["start"], 0),
               requested_count=settings["samples"], repeats=settings["repeats"])
    return row


def run_point(name: str, length: int, settings: Record, totals: Counts) -> Record:
    """A failed process still contributes an explicit, untimed result row."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=measure_point, args=(name, length, settings, totals, sender))
    started = perf_counter()
    worker.start()
    sender.close()
    try:
        if not receiver.poll(settings["seconds"] + 60):
            raise TimeoutError("Worker exceeded its watchdog deadline")
        return receiver.recv()
    except (EOFError, TimeoutError) as error:
        row = empty_record(name, length, settings, totals)
        row.update(status="timeout" if isinstance(error, TimeoutError) else "error",
                   failure_phase="worker", failure_elapsed_seconds=perf_counter()-started,
                   error=f"{type(error).__name__}: {error or 'Worker exited without a result'}")
        return row
    finally:
        receiver.close()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join()


def measure_point(name: str, length: int, settings: Record,
                  totals: Counts, connection: Connection) -> None:
    """Isolate every full preparation; limits stop execution, never trim support."""
    row = empty_record(name, length, settings, totals)
    phase, started = "loading", perf_counter()
    def peak_bytes() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    def guard(signum: int = 0, frame: FrameType | None = None) -> None:
        if peak_bytes() > settings["memory_gib"] * 2**30:
            raise MemoryError(f"Full preparation exceeded {settings['memory_gib']} GiB")
        if perf_counter()-started >= settings["seconds"]:
            raise TimeoutError(f"Full preparation exceeded {settings['seconds']} seconds")
    try:
        grammar = load_grammar(Path(settings["grammar"]), settings["start"])
        gc.collect(); phase = "chart"; started = perf_counter()
        signal.signal(signal.SIGALRM, guard)
        signal.setitimer(signal.ITIMER_REAL, 0.25, 0.25)
        chart = BinnedChart(grammar, name, scale=settings["scale"], width=settings["width"],
                            finite_bins=settings["finite_bins"], ratio=settings["ratio"])
        chart.extend(length)
        phase = "provenance"
        index = chart.compile_length(length)
        guard()
        prepare = perf_counter()-started
        signal.setitimer(signal.ITIMER_REAL, 0)
        phase = "validation"
        for n in range(1, length+1):
            actual = {str(a): sum(hist.values()) for a, hist in chart.H[n].items()}
            for a, count in totals[n].items():
                if actual.get(a, 0) != count: raise AssertionError("Binning lost derivations")
        if index.total != row["full_count"]: raise AssertionError("Incomplete full index")
        phase = "decode"
        count = min(settings["samples"], index.total)
        times, digest = measure_decode(index, count, length, settings["repeats"])
        decode = median(times)
        row.update(status="ok", root_bins=len(index.root_costs), available_count=index.total, decoded_count=count,
                   prepare_seconds=prepare, decode_seconds=decode, total_seconds=prepare+decode,
                   chart_count_seconds=prepare-index.provenance_seconds, provenance_seconds=index.provenance_seconds,
                   provenance_blocks=index.statistics()['provenance_blocks'],
                   decode_min_seconds=min(times), decode_max_seconds=max(times), output_sha256=digest,
                   decode_repetitions_seconds=times)
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        row.update(status="timeout" if isinstance(error, TimeoutError) else
                   "memory_limit" if isinstance(error, MemoryError) else "error",
                   failure_phase=phase, failure_elapsed_seconds=perf_counter()-started,
                   error=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    row["peak_memory_bytes"] = peak_bytes()
    connection.send(row)
    connection.close()


def validate_settings(lengths: tuple[int, ...]) -> None:
    for name, minimum in (("COST_SCALE", 1), ("UNIFORM_WIDTH", 1),
                          ("TRUNCATED_BINS", 1), ("GEOMETRIC_RATIO", 2),
                          ("SAMPLES", 0), ("REPEATS", 1), ("PLOT_DPI", 1)):
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("MAX_PREPARE_SECONDS", "MAX_PEAK_MEMORY_GIB"):
        value = globals()[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if (not BINNINGS or len(set(BINNINGS)) != len(BINNINGS)
            or set(BINNINGS) - {"identity", "uniform", "truncated", "geometric"}):
        raise ValueError("BINNINGS must contain distinct supported bin maps")
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("lengths must contain positive integers")


def benchmark(grammar_path: Path, lengths: tuple[int, ...]) -> list[Record]:
    validate_settings(lengths)
    grammar = load_grammar(grammar_path, START)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Validating complete derivation counts...", flush=True)
    totals = full_counts(grammar, max(lengths))
    records = []
    settings = dict(grammar=str(grammar_path), start=str(grammar.start), scale=COST_SCALE, width=UNIFORM_WIDTH,
                    finite_bins=TRUNCATED_BINS, ratio=GEOMETRIC_RATIO, samples=SAMPLES,
                    repeats=REPEATS, seconds=MAX_PREPARE_SECONDS, memory_gib=MAX_PEAK_MEMORY_GIB)
    metadata = dict(finished=False, complete=False, grammar=str(grammar_path), grammar_sha256=hashlib.sha256(grammar_path.read_bytes()).hexdigest(),
        start=str(grammar.start), scale=COST_SCALE, lengths=lengths, binnings=BINNINGS,
        width=UNIFORM_WIDTH, finite_truncated_bins=TRUNCATED_BINS, geometric_ratio=GEOMETRIC_RATIO,
        requested_count=SAMPLES, repeats=REPEATS, python=sys.version, platform=platform.platform(),
        max_prepare_seconds=MAX_PREPARE_SECONDS, max_peak_memory_gib=MAX_PEAK_MEMORY_GIB,
        maps={"identity": "t", "uniform": "floor(t/Delta)", "truncated": "floor(t/Delta) if t<m*Delta else infinity",
              "geometric": "floor(log_rho(1+(rho-1)*t/Delta))"},
        recurrence="terminal x^c; binary x^c tensor_phi (H_B tensor_phi H_C): phi(c+phi(a+b)); infinity absorbing",
        prepare_seconds="Cold full chart/counts, complete cumulative provenance blocks and root bin totals per length; grammar loading and validation excluded.",
        chart_count_seconds="Preparation excluding provenance construction and root cumulative totals; includes chart/counts and setup.",
        provenance_seconds="All provenance buckets, cumulative block totals and root cumulative bin totals, built during preparation before decoding.",
        decode_seconds="Median warmed batch through unchanged CostIndex.derivation/Bucket.select, including token rendering; no construction, answer caches, padding or extrapolation.",
        full_histograms="Every map retains every derivation. Identity retains every exact cost; truncated maps retain overflow.",
        resource_limits="Identical preparation time/peak-RSS limits, checked every 0.25s in isolated workers. Failures have no successful preparation/decode timing; no partial index is used.",
        decoder="All maps inherit CostIndex.locate and CostIndex.derivation unchanged: binary-search precomputed Bucket blocks, then divmod child ranks.",
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                      ("benchmark_latency.py", "benchmark_inputs.py", "cost_counting.py")}, records=records)
    csv_path, json_path = OUTPUT_DIR/"latency.csv", OUTPUT_DIR/"latency.json"
    started_all = perf_counter()
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore"); writer.writeheader()
        for point, length in enumerate(lengths):
            offset = point % len(BINNINGS)
            for name in BINNINGS[offset:]+BINNINGS[:offset]:
                print(f"Preparing full {name}, length={length}...", flush=True)
                row = run_point(name, length, settings, totals)
                writer.writerow(row); stream.flush(); records.append(row)
                json_path.write_text(json.dumps(metadata, indent=2)+"\n")
                if row["status"] == "ok":
                    print(f"{name:10} length={length:3} bins={row['root_bins']:6} decoded={row['decoded_count']:4} "
                          f"parse={row['prepare_seconds']:.3f}s decode={row['decode_seconds']*1000:.3f}ms", flush=True)
                else:
                    print(f"{name:10} length={length:3} {row['status']} in {row['failure_phase']}: {row['error']}", flush=True)
    metadata.update(finished=True, complete=all(r["status"] == "ok" for r in records),
                    successful_points=sum(r["status"] == "ok" for r in records),
                    elapsed_seconds=perf_counter()-started_all)
    json_path.write_text(json.dumps(metadata, indent=2)+"\n")
    return records


def plot_results(records: list[Record], grammar_path: Path) -> None:
    """Plot measured batches; failed preparations leave gaps in both curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axis = plt.subplots(figsize=(8, 5))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")
    for name, color in zip(BINNINGS, colors):
        rows = sorted((row for row in records if row["binning"] == name),
                      key=lambda row: row["length"])
        for field, style in (("prepare_seconds", "-"), ("decode_seconds", "--")):
            times = [1000 * row[field] if row["status"] == "ok" else math.nan for row in rows]
            axis.plot([row["length"] for row in rows], times, style, color=color,
                      marker="o", markersize=3)
    failures = [f"{row['binning']} n={row['length']}: {row['status']}"
                for row in records if row["status"] != "ok"]
    if failures:
        axis.text(0.02, 0.02, "Failed points omitted:\n" + "\n".join(failures),
                  transform=axis.transAxes, fontsize=7, va="bottom")
    axis.set(xlabel="Length (tokens)", ylabel="Time (ms)", yscale="log",
             title=f"{grammar_path.stem}: construction and decoding up to {SAMPLES:,} outputs")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, which="both", alpha=0.2)
    handles = [Line2D([], [], color=color, label=name.capitalize())
               for name, color in zip(BINNINGS, colors)]
    handles += [Line2D([], [], color="black", linestyle=style, label=label)
                for style, label in (("-", "Construction"), ("--", "Decoding"))]
    axis.legend(handles=handles, fontsize=8, ncol=2)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"latency.{extension}", dpi=PLOT_DPI)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_inputs(__doc__, GRAMMAR_PATH, LENGTHS, argv)
    records = benchmark(args.grammar, args.lengths)
    plot_results(records, args.grammar)
    print(f"Saved {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
