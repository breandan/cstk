"""Shared grammar and length-slice inputs for the standalone benchmarks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cost_counting as cnf


@dataclass(frozen=True)
class Rule:
    head: str
    tails: tuple[str, ...]
    weight: Fraction


def load_weighted_cnf(path: str | Path) -> list[Rule]:
    """Read strict CNF with explicit positive rational probabilities per rule."""
    rules, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            production, weight_text = line.rsplit(" [", 1)
            if not weight_text.endswith("]"):
                raise ValueError("missing closing weight bracket")
            weight = Fraction(weight_text[:-1])
            head, body = production.split("->")
            head, tails = head.strip(), tuple(body.split())
            if len(head.split()) != 1 or len(tails) not in (1, 2):
                raise ValueError("expected A -> token [p/q] or A -> B C [p/q]")
            if not 0 < weight <= 1:
                raise ValueError("probabilities must lie in (0, 1]")
            if (head, tails) in seen:
                raise ValueError("duplicate production")
            seen.add((head, tails))
            rules.append(Rule(head, tails, weight))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rules:
        raise ValueError("grammar has no productions")
    heads = {rule.head for rule in rules}
    totals = defaultdict(Fraction)
    for rule in rules:
        if len(rule.tails) == 1 and rule.tails[0] in heads:
            raise ValueError(f"unit production is not CNF: {rule}")
        if len(rule.tails) == 2 and any(tail not in heads for tail in rule.tails):
            raise ValueError(f"binary rule needs two nonterminals: {rule}")
        totals[rule.head] += rule.weight
    if any(total != 1 for total in totals.values()):
        raise ValueError("probabilities for each nonterminal must sum exactly to one")
    return rules


def make_grammar(rules: Sequence[Rule], start: str) -> cnf.Forest:
    symbols = {rule.head: cnf.Nonterminal(rule.head) for rule in rules}
    if start not in symbols:
        raise ValueError(f"unknown start symbol: {start}")
    grammar = cnf.Forest(symbols[start])
    for rule in rules:
        tails = tuple(symbols[tail] for tail in rule.tails) if len(rule.tails) == 2 else rule.tails
        grammar.add(cnf.Edge(symbols[rule.head], tails, rule.weight))
    return grammar


def load_grammar(path: str | Path, start: str | None = None) -> cnf.Forest:
    """Read weighted CNF or assign uniform probabilities to unweighted rules."""
    path = Path(path)
    lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    heads = dict.fromkeys(line.partition("->")[0].strip() for line in lines)
    if not heads:
        raise ValueError(f"{path}: grammar has no productions")
    start = start or next((name for name in ("START", "S") if name in heads), next(iter(heads)))
    if any(" [" in line and line.endswith("]") for line in lines):
        return make_grammar(load_weighted_cnf(path), start)
    grammar = cnf.load_cnf(path, start)
    weighted = cnf.Forest(grammar.start)
    for edge in grammar.edges:
        weight = Fraction(1, len(grammar.head_index[edge.head]))
        weighted.add(cnf.Edge(edge.head, edge.tails, weight))
    return weighted


def parse_lengths(value: str) -> tuple[int, ...]:
    """Inclusive FIRST..LAST[:STEP], a single length, or a comma-separated list."""
    try:
        if ".." in value:
            interval, _, step_text = value.partition(":")
            first, last = map(int, interval.split(".."))
            step = int(step_text) if step_text else 1
            if first < 1 or last < first or step < 1 or value.endswith(":"):
                raise ValueError
            lengths = tuple(range(first, last + 1, step))
        else:
            lengths = tuple(sorted(set(map(int, value.split(",")))))
        if not lengths or min(lengths) < 1:
            raise ValueError
        return lengths
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected positive lengths: FIRST..LAST[:STEP] (inclusive), e.g. 2..32:2 or 2,4,8"
        ) from error


def parse_inputs(description: str, default_grammar: Path, default_lengths: tuple[int, ...],
                 argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("grammar", nargs="?", type=Path, default=default_grammar,
                        help=f"CNF grammar path (default: {default_grammar.name})")
    parser.add_argument("lengths", nargs="?", type=parse_lengths, default=default_lengths,
                        metavar="FIRST..LAST[:STEP]", help="inclusive token-length slices")
    args = parser.parse_args(argv)
    if not args.grammar.is_file():
        parser.error(f"grammar file does not exist: {args.grammar}")
    return args
