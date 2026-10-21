#!/usr/bin/env python3
"""Exhaustively compare fixed-length Dyck word rankings against rational PCFG scores.

Requires NumPy (forest_py3) and Matplotlib (plots). Uses the native decoders:
cost_counting_orig.CostIndex with full cost histograms, and forest_py3.
cost_counting supplies only neutral CNF forest construction and token
rendering. No coarsening or polynomial-support cap is applied.
The cost-indexed decoder rounds negative-log rule weights to
integer costs. This experiment's loader reads rational weights from dyck.cnf.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
from importlib.metadata import version
import json
from math import comb, fsum, sqrt
from pathlib import Path
import platform
import sys
from time import perf_counter

from benchmark_inputs import Rule, load_weighted_cnf, make_grammar
import cost_counting as cnf
import cost_counting_orig as histogram
import forest_py3


@dataclass(frozen=True)
class WordScore:
    probability: Fraction
    costs: tuple[int, ...]


def enumerate_reference(rules, max_length, scales):
    """Independent exhaustive string DP; no histogram or k-best decoding.

    A word's PCFG probability is the product of its rule probabilities,
    because this grammar is unambiguous. Reject duplicate yields instead of
    silently comparing derivations to marginal word probabilities.
    """
    chart = defaultdict(dict)
    rule_costs = {rule: tuple(histogram.quantize_cost(rule.weight, scale)
                             for scale in scales) for rule in rules}

    def insert(head, length, word, score):
        if word in chart[head, length]:
            raise ValueError(f"ambiguous grammar: {head} derives {' '.join(word)!r} twice")
        chart[head, length][word] = score

    for rule in rules:
        if len(rule.tails) == 1:
            insert(rule.head, 1, rule.tails, WordScore(rule.weight, rule_costs[rule]))
    for length in range(2, max_length + 1):
        for rule in rules:
            if len(rule.tails) != 2:
                continue
            for split in range(1, length):
                left = chart.get((rule.tails[0], split), {})
                right = chart.get((rule.tails[1], length - split), {})
                for word_l, score_l in left.items():
                    for word_r, score_r in right.items():
                        costs = tuple(a + b + c for a, b, c in
                                      zip(rule_costs[rule], score_l.costs, score_r.costs))
                        insert(rule.head, length, word_l + word_r,
                               WordScore(rule.weight * score_l.probability * score_r.probability,
                                         costs))
    return chart


def average_ranks(values, descending=False):
    """One-based ranks, comparing Fractions exactly before averaging ties."""
    order = sorted(range(len(values)), key=values.__getitem__, reverse=descending)
    ranks = [0.0] * len(values)
    lo = 0
    while lo < len(order):
        hi = lo + 1
        while hi < len(order) and values[order[hi]] == values[order[lo]]:
            hi += 1
        for index in order[lo:hi]:
            ranks[index] = (lo + 1 + hi) / 2
        lo = hi
    return ranks


def rank_correlation(x, y):
    """Pearson correlation of rank vectors = Spearman rho (including ties)."""
    if len(x) != len(y) or not x:
        raise ValueError("rank vectors must be nonempty and have equal lengths")
    mean_x, mean_y = fsum(x) / len(x), fsum(y) / len(y)
    dx, dy = [v - mean_x for v in x], [v - mean_y for v in y]
    denominator = sqrt(fsum(v * v for v in dx) * fsum(v * v for v in dy))
    if denominator == 0:
        return None
    return max(-1.0, min(1.0, fsum(a * b for a, b in zip(dx, dy)) / denominator))


def dyck_word(word):
    stack = []
    for token in word:
        if token in ("(", "["):
            stack.append(token)
        elif token in (")", "]"):
            if not stack or stack.pop() != {")": "(", "]": "["}[token]:
                return False
        else:
            return False
    return not stack


def check_order(words, reference, length):
    if len(words) != len(reference) or len(set(words)) != len(words) or set(words) != set(reference):
        raise AssertionError(f"decoder did not enumerate every unique word once at n={length}")


def positive_list(value):
    try:
        result = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


SUMMARY_FIELDS = ("length", "word_count", "decoded_count", "method", "scale", "spearman_scores",
                  "spearman_positions", "score_tie_ceiling", "exact_score_levels",
                  "method_score_levels", "adjacent_score_inversions")
WORD_FIELDS = ("length", "scale", "word", "exact_probability", "exact_rank", "reference_position",
               "histogram_cost", "histogram_rank", "histogram_position", "forest_rank", "forest_position")


def benchmark(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rules = load_weighted_cnf(args.grammar)
    grammar = make_grammar(rules, args.start)
    started = perf_counter()
    print(f"Enumerating exact rational scores through n={max(args.lengths)}...", flush=True)
    chart = enumerate_reference(rules, max(args.lengths), args.scales)
    sources = [Path(__file__), Path(__file__).with_name("benchmark_inputs.py"), Path(histogram.__file__), Path(cnf.__file__), Path(forest_py3.__file__),
               Path(__file__).with_name("plot_rank_correlation.py")]
    metadata = {
        "complete": False, "grammar": args.grammar.name,
        "grammar_sha256": hashlib.sha256(args.grammar.read_bytes()).hexdigest(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "python": sys.version, "platform": platform.platform(), "start": args.start,
        "packages": {name: version(name) for name in ("numpy", "matplotlib")},
        "lengths": args.lengths, "scales": args.scales,
        "polynomial_backend": "cost_counting_orig.CostIndex",
        "polynomial_max_cost": None, "polynomial_coarsening": False,
        "polynomial_support_truncation": False,
        "forest_builder": "cost_counting.cnf_length_forest; construction and rendering only, no counting/decoding from this module",
        "reference": "Exhaustive independent string DP with Fraction products; descending exact probability.",
        "universe": "Every distinct word in each complete fixed-token-length slice; unambiguous Dyck_2, no sampling.",
        "spearman_scores": "Pearson of exact-score midranks and decoder-key midranks; Fraction scores for forest, summed rounded rule costs for histogram.",
        "spearman_positions": "Pearson of exact-score midranks and actual one-based decoder emission positions.",
        "score_tie_ceiling": "Perfect descending-score emission correlation; below one when exact scores tie.",
        "adjacent_score_inversions": "Adjacent emitted pairs whose exact probability increases; zero means the actual emission order respects descending PCFG score.",
        "ties": "Reference display order breaks exact ties lexicographically. Decoders retain native tie ordering; forest uses object IDs and may reorder equal-score words across runs. Midranks remove this irrelevant variability from forest correlations.",
        "costs": "Sum of cost_counting_orig.quantize_cost(p, scale) = round(-scale * log(p)) per rule. Exact sparse polynomial addition/convolution; all coefficients retained, no coarsening, merging distinct exponents, or cost cap.",
        "weights": "One rational PCFG and one common exact-length packed forest; forest weights including synthetic unit edges are Fraction.",
        "records": [],
    }
    manifest = output / "experiment.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as summary_stream, \
            gzip.open(output / "word_ranks.csv.gz", "wt", newline="") as word_stream:
        summary = csv.DictWriter(summary_stream, fieldnames=SUMMARY_FIELDS)
        word_writer = csv.DictWriter(word_stream, fieldnames=WORD_FIELDS)
        summary.writeheader()
        word_writer.writeheader()
        for length in args.lengths:
            slice_started = perf_counter()
            reference = chart[args.start, length]
            pairs = length // 2
            expected = comb(2 * pairs, pairs) // (pairs + 1) * 2 ** pairs
            if len(reference) != expected or not all(dyck_word(word) for word in reference):
                raise AssertionError(f"n={length}: expected {expected} distinct Dyck_2 words")
            canonical = sorted(reference, key=lambda word: (-reference[word].probability, word))
            probabilities = [reference[word].probability for word in canonical]
            exact_ranks = average_ranks(probabilities, descending=True)
            ceiling = rank_correlation(exact_ranks, list(range(1, len(canonical) + 1)))
            packed = cnf.cnf_length_forest(grammar, length, exact_length=True)
            forest = forest_py3.Forest(packed.start, viterbi=True)
            for edge in packed.edges:
                forest.add(forest_py3.Edge(edge.head, edge.tails, Fraction(edge.weight)))
            forest_words, forest_scores = [], []
            for rank in range(len(canonical)):
                word = tuple(forest.derivation(rank, cnf.cnf_tokens).split())
                score = forest._kbest[forest.start][rank][0]
                if score != reference[word].probability:
                    raise AssertionError("forest score disagrees with independent exact reference")
                forest_words.append(word)
                forest_scores.append(score)
                if (rank + 1) % 100_000 == 0:
                    print(f"n={length} forest decoded {rank + 1:,}/{len(canonical):,}", flush=True)
            check_order(forest_words, reference, length)
            if any(a < b for a, b in zip(forest_scores, forest_scores[1:])):
                raise AssertionError("forest scores are not descending")
            try:
                forest.derivation(len(canonical), cnf.cnf_tokens)
            except IndexError:
                pass
            else:
                raise AssertionError("forest emitted more derivations than words")
            forest_positions = {word: pos for pos, word in enumerate(forest_words, 1)}
            forest_rank_map = dict(zip(forest_words, average_ranks(forest_scores, descending=True)))

            def record(method, scale, keys, ranks, positions):
                emitted_scores = [None] * len(canonical)
                for probability, position in zip(probabilities, positions):
                    emitted_scores[position - 1] = probability
                row = dict(length=length, word_count=len(canonical), decoded_count=len(positions),
                           method=method, scale=scale,
                           spearman_scores=rank_correlation(exact_ranks, ranks),
                           spearman_positions=rank_correlation(exact_ranks, positions),
                           score_tie_ceiling=ceiling, exact_score_levels=len(set(probabilities)),
                           method_score_levels=len(set(keys)),
                           adjacent_score_inversions=sum(a < b for a, b in zip(emitted_scores, emitted_scores[1:])))
                summary.writerow(row)
                metadata["records"].append(row)
                return row

            record("exact", "", probabilities, exact_ranks, list(range(1, len(canonical) + 1)))
            record("forest", "", forest_scores, [forest_rank_map[word] for word in canonical],
                   [forest_positions[word] for word in canonical])
            for scale_index, scale in enumerate(args.scales):
                # The original decoder computes every exact polynomial
                # coefficient; no bounded-bin or coarsening path is involved.
                index = histogram.CostIndex(packed, max_cost=None, scale=scale)
                if index.total != len(reference):
                    raise AssertionError("histogram count differs from exhaustive word count")
                hist_words, hist_costs = [], []
                for rank in range(index.total):
                    word = tuple(index.derivation(rank, cnf.cnf_tokens).split())
                    cost = index.locate(rank)[0]
                    if cost != reference[word].costs[scale_index]:
                        raise AssertionError("histogram cost disagrees with independent rule-cost sum")
                    hist_words.append(word)
                    hist_costs.append(cost)
                    if (rank + 1) % 100_000 == 0:
                        print(f"n={length} polynomial scale={scale} decoded "
                              f"{rank + 1:,}/{index.total:,}", flush=True)
                check_order(hist_words, reference, length)
                if hist_costs != sorted(hist_costs):
                    raise AssertionError("histogram costs are not ascending")
                positions = {word: pos for pos, word in enumerate(hist_words, 1)}
                rank_map = dict(zip(hist_words, average_ranks(hist_costs)))
                row = record("histogram", scale, hist_costs, [rank_map[word] for word in canonical],
                             [positions[word] for word in canonical])
                for pos, word in enumerate(canonical, 1):
                    word_writer.writerow(dict(
                        length=length, scale=scale, word="".join(word),
                        exact_probability=str(reference[word].probability), exact_rank=exact_ranks[pos - 1],
                        reference_position=pos, histogram_cost=reference[word].costs[scale_index],
                        histogram_rank=rank_map[word], histogram_position=positions[word],
                        forest_rank=forest_rank_map[word], forest_position=forest_positions[word]))
                rho = row["spearman_scores"]
                print(f"n={length:2} words={len(canonical):6,} scale={scale:4} "
                      f"Spearman={rho if rho is None else format(rho, '.9f')}", flush=True)
            summary_stream.flush()
            print(f"n={length} full slice checked in {perf_counter() - slice_started:.1f}s", flush=True)
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["complete"] = True
    metadata["elapsed_seconds"] = perf_counter() - started
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return output / "summary.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path, default=Path(__file__).with_name("dyck.cnf"))
    parser.add_argument("--start", default="S")
    parser.add_argument("--lengths", type=positive_list, default=list(range(2, 19, 2)))
    parser.add_argument("--scales", type=positive_list, default=[1000],
                        help="integer rule-cost scale; does not coarsen or truncate the polynomial")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("rank_correlation"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if any(length % 2 for length in args.lengths):
        parser.error("Dyck slices must have positive even token lengths")
    csv_path = benchmark(args)
    print(f"Saved {csv_path}")
    if not args.no_plot:
        from plot_rank_correlation import make_plots
        for path in make_plots(csv_path, args.output_dir):
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
