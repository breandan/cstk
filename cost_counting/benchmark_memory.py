#!/usr/bin/env python3
"""Peak resident memory for forest rescoring and cost counting.

Usage: python benchmark_memory.py [grammar.cnf] [2..32:2]
Edit the settings below to change decoding, repetition, or resource budgets.
Each method/length/repetition uses a fresh process and the same exact-length
packed forest. RSS includes the interpreter, graph, index, and retained outputs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gc
import hashlib
from importlib.metadata import version
import json
from math import comb, isfinite
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from pathlib import Path
import platform
import resource
import signal
from statistics import median
import sys
from time import perf_counter
from typing import Any, TYPE_CHECKING

from benchmark_inputs import load_grammar, parse_inputs

if TYPE_CHECKING:
    import cost_counting as cnf
    import cost_counting_orig as counted
    import forest_py3 as rescored

ROOT = Path(__file__).resolve().parent
GRAMMAR_PATH = ROOT / "dyck.cnf"
LENGTHS = tuple(range(2, 33, 2))
SAMPLES = 256_000
REPEATS = 3
COST_SCALE = 10 ** 9
COST_TOP_K = True
RETAIN_OUTPUTS = True
JOBS = 1
MAX_SECONDS = 120
MAX_MEMORY_GIB = 4
START: str | None = None
OUTPUT_DIR = ROOT / "benchmark_results" / "memory_comparison_current"

Record = dict[str, Any]
METHODS = ("forest", "cost_counting")
LABELS = {"forest": "Forest rescoring", "cost_counting": "Cost counting (full index)"}
FIELDS = (
    "method", "length", "repeat", "status", "failure_phase", "full_count",
    "requested_count", "decoded_count", "baseline_peak_rss_bytes",
    "prepare_peak_rss_bytes", "peak_rss_bytes", "elapsed_seconds", "nodes",
    "edges", "root_costs", "provenance_blocks", "cached_derivations",
    "retained_count", "output_list_shallow_bytes", "output_strings_bytes",
    "retained_output_bytes", "output_sha256", "error",
)


def peak_rss_bytes() -> int:
    """ru_maxrss is bytes on macOS, KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def slice_size(length: int) -> int:
    """Closed-form Dyck_2 count, retained for independent regression checks."""
    if length < 2 or length % 2:
        return 0
    pairs = length // 2
    return comb(length, pairs) // (pairs + 1) * 2 ** pairs


def derivation_counts(rules: Iterable[Any], start: Any, max_length: int) -> dict[int, int]:
    """Independent integer inside recurrence; no costs or decoder state."""
    rules = tuple(rules)
    heads = {rule.head for rule in rules}
    if start not in heads:
        raise ValueError(f"unknown start symbol: {start}")
    counts = [defaultdict(int) for _ in range(max_length + 1)]
    binary = []
    for rule in rules:
        if len(rule.tails) == 1:
            counts[1][rule.head] += 1
        else:
            binary.append((rule.head, *rule.tails))
    for size in range(2, max_length + 1):
        for head, left, right in binary:
            counts[size][head] += sum(counts[k].get(left, 0) * counts[size-k].get(right, 0)
                                      for k in range(1, size))
    return {n: counts[n].get(start, 0) for n in range(1, max_length + 1)}


def native_forest(method: str, grammar: cnf.Forest, length: int) -> counted.Forest | rescored.Forest:
    """Adapt the same tail-first unrolling to each backend's own graph types."""
    import cost_counting as cnf
    import cost_counting_orig as counted
    import forest_py3 as rescored
    from fractions import Fraction

    packed = cnf.cnf_length_forest(grammar, length, exact_length=True)
    if method == "forest":
        forest = rescored.Forest(packed.start, viterbi=True)
        for edge in packed.edges:
            forest.add(rescored.Edge(edge.head, edge.tails, Fraction(edge.weight)))
    elif method == "cost_counting":
        forest = counted.Forest(packed.start)
        for edge in packed.edges:
            forest.add(counted.Edge(edge.head, edge.tails, edge.weight, edge.cost))
    else:
        raise ValueError(f"unknown method: {method}")
    # No reference to the temporary common graph survives this function.
    return forest


def measure_point(method: str, length: int, repeat: int,
                  settings: Record, connection: Connection) -> None:
    """Send one measurement; a failed run never becomes a successful point."""
    row = dict.fromkeys(FIELDS)
    row.update(method=method, length=length, repeat=repeat, status="error",
               full_count=settings.get("full_counts", {}).get(str(length)),
               requested_count=settings["samples"],
               decoded_count=0, retained_count=0, output_list_shallow_bytes=0,
               output_strings_bytes=0, retained_output_bytes=0)
    retained_outputs = None
    phase, started = "imports", perf_counter()

    def guard(*_: Any) -> None:
        if peak_rss_bytes() > settings["memory_gib"] * 2 ** 30:
            raise MemoryError(f"worker exceeded {settings['memory_gib']} GiB peak RSS")
        if perf_counter() - started > settings["seconds"]:
            raise TimeoutError(f"worker exceeded {settings['seconds']} seconds")

    try:
        signal.signal(signal.SIGALRM, guard)
        signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
        # Import the same dependencies in both workers. Plotting stays in the parent.
        import cost_counting as cnf
        import cost_counting_orig as counted

        phase = "grammar"
        grammar = load_grammar(Path(settings["grammar"]), settings.get("start"))
        terminals = {edge.tails[0] for edge in grammar.edges if len(edge.tails) == 1}
        if row["full_count"] is None:
            row["full_count"] = derivation_counts(grammar.edges, grammar.start, length)[length]
        gc.collect()
        row["baseline_peak_rss_bytes"] = peak_rss_bytes()
        if row["full_count"] == 0:
            row.update(status="empty", prepare_peak_rss_bytes=peak_rss_bytes(),
                       peak_rss_bytes=peak_rss_bytes(), elapsed_seconds=perf_counter() - started)
            signal.setitimer(signal.ITIMER_REAL, 0)
            connection.send(row)
            connection.close()
            return
        phase = "forest"
        forest = native_forest(method, grammar, length)
        gc.collect()
        row.update(nodes=len(forest.nodes), edges=len(forest.edges))
        phase = "index"
        if method == "cost_counting":
            top_k = settings["samples"] if settings.get("cost_top_k") else None
            decoder = counted.CostIndex(forest, max_cost=None, scale=settings["scale"], top_k=top_k)
            expected = row["full_count"] if top_k is None else min(top_k, row["full_count"])
            if decoder.total != expected:
                raise AssertionError("cost index disagrees with the expected slice/prefix count")
            row.update(root_costs=len(decoder.root_costs), provenance_blocks=sum(
                len(bucket.choices) for buckets in decoder.buckets.values()
                for bucket in buckets.values()))
        else:
            decoder = forest
        row["prepare_peak_rss_bytes"] = peak_rss_bytes()

        phase = "decode"
        output_count = min(settings["samples"], row["full_count"])
        if settings.get("retain_outputs", False):
            # Both backends return the same representation: a space-separated
            # token string. The known output count avoids list over-allocation.
            retained_outputs = [None] * output_count
        digest, previous = hashlib.sha256(), None
        for rank in range(output_count):
            word = decoder.derivation(rank, cnf.cnf_tokens)
            tokens = word.split()
            if len(tokens) != length or any(token not in terminals for token in tokens):
                raise AssertionError("decoder returned an invalid token or yield length")
            key = (-forest._kbest[forest.start][rank][0] if method == "forest"
                   else decoder.locate(rank)[0])
            if previous is not None and key < previous:
                raise AssertionError("decoder keys are out of order")
            previous = key
            digest.update(word.encode() + b"\n")
            if retained_outputs is not None:
                retained_outputs[rank] = word
                row["retained_count"] += 1
            row["decoded_count"] += 1
        if row["decoded_count"] == row["full_count"]:
            # Tiny slices are exhaustive; confirm the next rank is exhausted.
            try:
                decoder.derivation(row["full_count"], cnf.cnf_tokens)
            except IndexError:
                pass
            else:
                raise AssertionError("decoder returned more than the independently counted slice size")
        if method == "forest":
            row["cached_derivations"] = sum(len(kbest.result) for kbest in forest._kbest.values())
        guard()
        row.update(status="ok", output_sha256=digest.hexdigest())
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        row.update(status="timeout" if isinstance(error, TimeoutError) else
                   "memory_limit" if isinstance(error, MemoryError) else "error",
                   failure_phase=phase, error=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if retained_outputs is not None:
        # Object sizes describe the returned list, not allocator/RSS overhead.
        # Strings are retained exactly as emitted, without deduplication.
        row["output_list_shallow_bytes"] = sys.getsizeof(retained_outputs)
        row["output_strings_bytes"] = sum(sys.getsizeof(word) for word in retained_outputs
                                           if word is not None)
        row["retained_output_bytes"] = (row["output_list_shallow_bytes"]
                                        + row["output_strings_bytes"])
    # The output list and native decoder state are still alive at this reading.
    row.update(peak_rss_bytes=peak_rss_bytes(), elapsed_seconds=perf_counter() - started)
    connection.send(row)
    connection.close()


def run_point(context: BaseContext, method: str, length: int,
              repeat: int, settings: Record) -> Record:
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=measure_point, args=(method, length, repeat, settings, sender))
    started = perf_counter()
    worker.start()
    sender.close()
    try:
        if receiver.poll(settings["seconds"] + 10):
            try:
                return receiver.recv()
            except EOFError:
                status, error = "error", "worker exited without reporting a measurement"
        else:
            status, error = "timeout", "worker exceeded parent watchdog deadline"
        # No RSS is invented for a worker that cannot report its own high-water mark.
        row = dict.fromkeys(FIELDS)
        row.update(method=method, length=length, repeat=repeat, status=status,
                   failure_phase="worker", full_count=settings.get("full_counts", {}).get(str(length)),
                   requested_count=settings["samples"], error=error,
                   elapsed_seconds=perf_counter() - started)
        return row
    finally:
        receiver.close()
        worker.join(timeout=1)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join()


def summarize(records: list[Record], repeats: int) -> list[Record]:
    """Only complete successful repetition groups receive plotted statistics."""
    rows = []
    for method in METHODS:
        for length in sorted({r["length"] for r in records}):
            group = [r for r in records if (r["method"], r["length"]) == (method, length)]
            complete = len(group) == repeats and all(r["status"] == "ok" for r in group)
            empty = len(group) == repeats and all(r["status"] == "empty" for r in group)
            peaks = [r["peak_rss_bytes"] for r in group] if complete else []
            full_count = next((r.get("full_count") for r in group if r.get("full_count") is not None), None)
            rows.append(dict(method=method, length=length,
                             status="ok" if complete else "empty" if empty else "incomplete",
                             repeats=len(group), full_count=full_count,
                             decoded_count=group[0]["decoded_count"] if complete else None,
                             median_peak_rss_bytes=median(peaks) if peaks else None,
                             min_peak_rss_bytes=min(peaks) if peaks else None,
                             max_peak_rss_bytes=max(peaks) if peaks else None))
    return rows


def write_csv(path: Path, rows: list[Record], fields: Iterable[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(metadata: Record, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    rows = summarize(metadata["records"], metadata["settings"]["repeats"])
    failed = sum(r["status"] != "ok" for r in rows)
    if failed:
        fig, (ax, status_ax) = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True,
                                           gridspec_kw={"height_ratios": [5, 0.7], "hspace": 0.08})
        status_ax.set_yticks([1, 0], ["Forest", "Cost counting"], fontsize=8)
        status_ax.set_ylim(-0.6, 1.6)
        status_ax.spines[["top", "right", "left"]].set_visible(False)
        status_ax.tick_params(axis="y", length=0)
    else:
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        status_ax = None
    colors = {"forest": "#0072B2", "cost_counting": "#D55E00"}
    top_k = metadata["settings"].get("cost_top_k", False)
    for method in METHODS:
        points = [r for r in rows if r["method"] == method]
        x = [r["length"] for r in points]
        def values(key: str) -> list[float]:
            return [r[key] / 2 ** 20 if r[key] is not None else float("nan") for r in points]
        ax.plot(x, values("median_peak_rss_bytes"), marker="o" if method == "forest" else "s",
                markersize=4.5, linewidth=2, color=colors[method],
                label="Cost counting (top-k)" if method == "cost_counting" and top_k else LABELS[method])
        ax.fill_between(x, values("min_peak_rss_bytes"), values("max_peak_rss_bytes"),
                        color=colors[method], alpha=0.15, linewidth=0)
        if status_ax is not None:
            for row in points:
                group = [r for r in metadata["records"] if
                         (r["method"], r["length"]) == (method, row["length"])]
                statuses = {r["status"] for r in group}
                marker = ("_" if not group else "." if row["status"] == "ok" else "o" if row["status"] == "empty" else
                          "x" if "memory_limit" in statuses else "v" if "timeout" in statuses else "D")
                status_ax.scatter(row["length"], 1 if method == "forest" else 0,
                                  marker=marker, s=18 if marker == "." else 30,
                                  color=colors[method] if group else "#888888")
    has_completed = any(row["median_peak_rss_bytes"] is not None for row in rows)
    if not has_completed:
        ax.set_ylim(1, 2)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    if not has_completed:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No completed memory measurements", transform=ax.transAxes,
                ha="center", va="center", color="#666666")
    (status_ax if status_ax is not None else ax).set_xlabel("Length (tokens)")
    ax.set_ylabel("Peak resident memory (MiB, log scale)")
    lengths = list(metadata["settings"]["lengths"])
    ticks = lengths[::max(1, len(lengths) // 12)]
    ax.set_xticks(sorted(set(ticks + [lengths[-1]])))
    ax.grid(axis="both", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    settings = metadata["settings"]
    fig.suptitle(f"{settings.get('title', 'Dyck₂')}: memory by token length", fontsize=16, y=0.98)
    cost_mode = "top-k cost index" if top_k else "full cost index"
    ax.set_title(f"Build + first min({settings['samples']:,}, slice size) derivations; "
                 f"{cost_mode}, scale {settings['scale']:,}", fontsize=10, pad=12)
    footer = (f"Median of {settings['repeats']} fresh processes per point; band = min–max. "
              "Interpreter/import overhead included.\n"
              + ("Decoded strings retained in a preallocated Python list. "
                 if settings.get("retain_outputs", False) else "Decoded strings discarded. ")
              + "Native decoder caches retained.\n"
              +
              "Forest: lazy k-best with rational scores. Cost counting: "
              + ("demand-driven prefix cost/provenance index." if top_k else
                 "eager full cost/provenance index."))
    if failed:
        footer += (f"\nStatus strip: · completed, × memory limit ({settings['memory_gib']:g} GiB), "
                   f"▽ time limit ({settings['seconds']:g} s), ○ empty.\n"
                   "— no completed trial; ◆ incomplete repeats/error. Only fully decoded groups enter the memory curves.")
    fig.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.20 if not failed else 0.25)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"memory.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def benchmark(grammar_path: Path, lengths: tuple[int, ...]) -> Record:
    """Snapshot globals into worker settings, then measure each requested slice."""
    for name, value in (("SAMPLES", SAMPLES), ("REPEATS", REPEATS),
                        ("COST_SCALE", COST_SCALE), ("JOBS", JOBS)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if any(not isfinite(value) or value <= 0 for value in (MAX_SECONDS, MAX_MEMORY_GIB)):
        raise ValueError("MAX_SECONDS and MAX_MEMORY_GIB must be finite and positive")
    if not lengths or any(n < 1 for n in lengths):
        raise ValueError("lengths must be positive")
    grammar_path = grammar_path.resolve()
    grammar = load_grammar(grammar_path, START)
    full_counts = derivation_counts(grammar.edges, grammar.start, max(lengths))
    title = "Dyck₂" if grammar_path == ROOT / "dyck.cnf" else grammar_path.stem
    settings = dict(grammar=str(grammar_path), start=str(grammar.start), title=title,
                    lengths=lengths, samples=SAMPLES, repeats=REPEATS, scale=COST_SCALE,
                    seconds=MAX_SECONDS, memory_gib=MAX_MEMORY_GIB, jobs=JOBS,
                    cost_top_k=COST_TOP_K, retain_outputs=RETAIN_OUTPUTS,
                    full_counts={str(n): count for n, count in full_counts.items()})
    sources = ("benchmark_memory.py", "benchmark_inputs.py", "cost_counting.py",
               "cost_counting_orig.py", "forest_py3.py")
    metadata = dict(
        finished=False, complete=False, settings=settings, python=sys.version,
        platform=platform.platform(), packages={name: version(name) for name in ("numpy", "matplotlib")},
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        grammar_sha256=hashlib.sha256(grammar_path.read_bytes()).hexdigest(),
        metric="Process-lifetime peak RSS (ru_maxrss), bytes; median and min/max of fresh workers. "
               "Includes interpreter, imports, grammar, graph construction, decoding, and retained outputs.",
        baseline="Peak RSS after imports and grammar loading; never subtracted from the measurement.",
        workload="First min(samples, full_count) derivations per exact-length slice. Independent integer "
                 "inside recurrence counts derivations; ambiguous grammars can repeat a terminal string.",
        output_storage=("Native space-separated strings retained in a preallocated list through final RSS. "
                        "retained_output_bytes sums sys.getsizeof(list) and sys.getsizeof(each string); "
                        "this logical size excludes allocator overhead and is not added to RSS."
                        if RETAIN_OUTPUTS else "Outputs hashed and discarded; retained-output fields are zero."),
        forest="forest_py3.Forest.derivation: native lazy k-best with rational scores and retained caches.",
        cost_counting=("cost_counting_orig.CostIndex: integer costs at the configured scale; "
                       + ("demand-driven top-k prefix with capped counts and partial cutoff buckets."
                          if COST_TOP_K else "full support and provenance, without a cost cap.")),
        forest_builder="cost_counting.cnf_length_forest(exact_length=True), converted to native graph types. "
                       "Temporary common graph released before indexing in both methods.",
        limits="Wall-time and peak-RSS budgets checked every 0.1 s, with a parent watchdog. "
               "Every trial is attempted; only fully successful groups enter the curves.",
        records=[],
    )
    output = OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    context = multiprocessing.get_context("spawn")
    started = perf_counter()
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    tasks = []
    for point, length in enumerate(lengths):
        for repeat in range(1, settings["repeats"] + 1):
            methods = METHODS if (point + repeat) % 2 else METHODS[::-1]
            tasks.extend((method, length, repeat) for method in methods)
    print(f"{title}: {len(grammar.edges):,} rules; start={grammar.start}; {len(tasks)} fresh workers; "
          f"up to {settings['jobs']} concurrently", flush=True)
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=settings["jobs"]) as pool:
            futures = [pool.submit(run_point, context, method, length, repeat, settings)
                       for method, length, repeat in tasks]
            for future in as_completed(futures):
                row = future.result()
                metadata["records"].append(row)
                writer.writerow(row)
                stream.flush()
                manifest.write_text(json.dumps(metadata, indent=2) + "\n")
                label = ("Cost counting (top-k)" if row["method"] == "cost_counting" and settings["cost_top_k"]
                         else LABELS[row["method"]])
                prefix = (f"[{len(metadata['records'])}/{len(tasks)}] {label} "
                          f"n={row['length']}, repeat {row['repeat']}/{settings['repeats']}:")
                if row["status"] == "ok":
                    print(f"{prefix} peak {row['peak_rss_bytes'] / 2 ** 20:,.2f} MiB; "
                          f"decoded {row['decoded_count']:,}/{row['full_count']:,}", flush=True)
                else:
                    print(f"{prefix} {row['status']}; decoded {row['decoded_count']}; {row['error']}", flush=True)
    metadata.update(finished=True, complete=all(r["status"] in ("ok", "empty") for r in metadata["records"]),
                    elapsed_seconds=perf_counter() - started)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summarize(metadata["records"], settings["repeats"])
    write_csv(output / "summary.csv", summary, list(summary[0]))
    plot_results(metadata, output)
    print(f"Saved measurements and plots in {output}", flush=True)
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_inputs(__doc__, GRAMMAR_PATH, LENGTHS, argv)
    metadata = benchmark(args.grammar, args.lengths)
    if not metadata["complete"]:
        raise SystemExit("Some measurements failed; see measurements.csv for details.")


if __name__ == "__main__":
    main()
