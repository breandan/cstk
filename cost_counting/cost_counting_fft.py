#!/usr/bin/env python3
"""FFT cost-counting decoder; Python 3.10+ and NumPy.

The command-line benchmark reads python.cnf (or --grammar), starting at
START (or --start). Its unweighted CNF rules receive uniform probabilities
per nonterminal. By default, compile the minimum-cost derivations (one bin).
--length N generates exactly N tokens; --max-length N allows 1..N tokens;
--tokens parses a fixed sequence. Without a length/input constraint, search
the entire language. --histogram-buckets 0 enables unrestricted cost ranks.
Ranks are zero-based and index derivations, which are distinct strings only
when the grammar is unambiguous. Token names such as NAME and NEWLINE remain
literal tokens; raw Python source is not tokenized automatically.
The CLI prints live preparation timings; --quiet-timings suppresses them.
--histogram-buckets K retains the K lowest distinct exact costs per node,
including the exact count of all derivations tied at each cost. One bin is
top-1 cost counting, not a merged summary of higher costs. The limit applies
throughout construction and decoding; no uncapped refinement is performed.
Ranks outside the retained prefix require more bins (or 0 for all costs).

In the final index, H[v][c] counts derivations of v of EXACT integer cost c.
Addition is pointwise
addition; multiplication is convolution. Coefficients and ranks are stored
as Python integers. Convolution uses zero-padded NumPy real FFTs, rounding
float64 results to integers. Large coefficient bounds, very sparse spans,
or suspect numerical results fall back to exact sparse multiplication.
Compilation checks the resulting counts against exact provenance totals.
Probabilities are converted to integer costs with
c(e) = round(-scale * log(p(e))). Explicit integer costs may instead be
supplied as Edge(..., cost=...).

One-bin CNF charts compute minima first and materialize only optimal edges.
Compilation retains provenance only for the selected cost levels.
Queries never enumerate earlier derivations or extend the index. Within a
cost bucket, ties use forest edge insertion order, then lexicographic child
costs, then mixed-radix child ranks.

Supported forests: arbitrary acyclic hypergraphs and linear cyclic SCCs
(at most one tail of an edge is in its head's SCC), including the demo's
unary cycles. Zero-cost cycles and nonlinear cyclic SCCs are rejected.
For an acyclic finite forest, histogram_buckets=0 and max_cost=None include
its entire support. Positive bin caps require an acyclic forest.
For a cyclic forest, an explicit finite max_cost is required. Bounded-length
CNF generation unrolls recursive rules into an acyclic forest first. Automatic
generation with more than one bin grows that forest until every derivation
at or below the requested rank's cost is included. Generation groups equal
costs by yield length, then grammar rule order, preserving ranks as bins grow.

Cost quantization can change probability order in general. Equal rule
probabilities are a benchmark convention, not a trained Python language model.

Storage includes ONE provenance block per (edge, child-cost tuple), not one
record per derivation. This can still be large for ambiguous grammars.
No claim of a universal memory or preprocessing improvement is intended.
Convolution operates on exact integer costs. Dense FFT workspace depends on
the cost span. One-bin multiplication needs only integer addition/product.
One-bin CNF workloads bypass the general packed-forest construction.
Unrestricted counting keeps the cyclic propagation and provenance scheme
from cost_counting.py, with guarded FFT polynomial multiplication.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from itertools import accumulate, product
import json
from math import gcd, isfinite, log
from pathlib import Path
from time import perf_counter

import numpy as np


class CycleDetected(ValueError):
    pass


class ProgressTimings:
    """Flushed phase timings and throttled progress for CLI preparation."""

    def __init__(self, interval=2.0):
        self.interval = interval
        self.started = perf_counter()
        self.next_update = self.started + interval

    def note(self, message):
        now = perf_counter()
        print(f"[prepare +{now - self.started:9.3f}s] {message}", flush=True)
        self.next_update = now + self.interval

    def start(self, phase):
        self.note(f"{phase}: starting")
        return perf_counter()

    def finish(self, phase, started, detail=""):
        elapsed = perf_counter() - started
        self.note(f"{phase}: {elapsed:.3f}s" + (f"; {detail}" if detail else ""))

    def due(self):
        return perf_counter() >= self.next_update


def quantize_cost(weight, scale=1000):
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if not isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("probability must be finite and in (0, 1]")
    return round(-scale * log(weight))


# Sparse polynomials: {exponent: positive integer coefficient}.
def poly_add(a, b):
    result = dict(a)
    for cost, count in b.items():
        result[cost] = result.get(cost, 0) + count
    return result


def _poly_mul_sparse(a, b, limit=None):
    """Exact fallback for monomials, large counts, and very sparse spans."""
    result = defaultdict(int)
    for ca, na in a.items():
        for cb, nb in b.items():
            cost = ca + cb
            if limit is None or cost <= limit:
                result[cost] += na * nb
    return dict(result)


# Bound dense workspace and leave ample float64 headroom for rounding.
_FFT_MAX_SIZE = 1 << 20
_FFT_MAX_MASS = 1 << 40


def poly_mul(a, b, limit=None):
    """Multiply nonnegative-cost, positive-integer sparse polynomials.

    Remove offsets and a shared cost stride before allocating dense arrays.
    Use a guarded float64 FFT when practical, otherwise exact multiplication.
    """
    if not a or not b:
        return {}
    min_a, min_b = min(a), min(b)
    offset = min_a + min_b
    if limit is not None:
        if offset > limit:
            return {}
        a = {c: n for c, n in a.items() if c <= limit - min_b}
        b = {c: n for c, n in b.items() if c <= limit - min_a}
    if len(a) == 1 or len(b) == 1:
        return _poly_mul_sparse(a, b, limit)

    stride = 0
    for c in a:
        stride = gcd(stride, c - min_a)
    for c in b:
        stride = gcd(stride, c - min_b)
    size_a = (max(a) - min_a) // stride + 1
    size_b = (max(b) - min_b) // stride + 1
    size = size_a + size_b - 1
    fft_size = 1 << (size - 1).bit_length()
    mass = sum(a.values()) * sum(b.values())
    if (fft_size > _FFT_MAX_SIZE or fft_size > 16 * len(a) * len(b)
            or mass > _FFT_MAX_MASS):
        return _poly_mul_sparse(a, b, limit)

    dense_a = np.zeros(size_a, dtype=np.float64)
    dense_b = np.zeros(size_b, dtype=np.float64)
    for c, n in a.items():
        dense_a[(c - min_a) // stride] = n
    for c, n in b.items():
        dense_b[(c - min_b) // stride] = n
    # Pad for the FULL product, even when only a prefix is requested: shorter
    # transforms would wrap high-cost coefficients into the retained prefix.
    values = np.fft.irfft(np.fft.rfft(dense_a, n=fft_size)
                          * np.fft.rfft(dense_b, n=fft_size), n=fft_size)[:size]
    rounded = np.rint(values)
    if (not np.all(np.isfinite(values)) or np.any(rounded < 0)
            or np.any(np.abs(values - rounded) > 0.25)):
        return _poly_mul_sparse(a, b, limit)
    coefficients = rounded.astype(np.int64)
    if int(coefficients.sum()) != mass:
        return _poly_mul_sparse(a, b, limit)
    if limit is not None:
        coefficients = coefficients[:(limit - offset) // stride + 1]
    return {offset + stride * i: int(n)
            for i, n in enumerate(coefficients) if n}


def compress_histogram(hist, max_buckets):
    """Keep the lowest distinct costs and their exact counts.

    On an acyclic forest, truncating to K distinct costs after each sum or
    product preserves the first K coefficients exactly. An omitted child
    cost has K cheaper alternatives paired with the same sibling costs.
    In particular, one bin is min-plus counting, including all minimum ties.
    """
    if len(hist) <= max_buckets:
        return hist
    if max_buckets == 1:
        cost = min(hist)
        return {cost: hist[cost]}
    return {cost: hist[cost] for cost in heapq.nsmallest(max_buckets, hist)}



def poly_shift(a, cost, limit=None):
    return {c + cost: n for c, n in a.items()
            if limit is None or c + cost <= limit}


@dataclass(frozen=True, slots=True)
class Edge:
    head: object
    tails: tuple
    weight: float = 1.0
    cost: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "tails", tuple(self.tails))
        if self.cost is not None and (
            not isinstance(self.cost, int) or isinstance(self.cost, bool)
            or self.cost < 0
        ):
            raise ValueError("explicit costs must be nonnegative integers")


class Node:
    def __init__(self, label=None):
        self.label = label

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return f"{type(self).__name__}({self.label!r})"


class Nonterminal(Node):
    pass


class Forest:
    def __init__(self, start):
        self.start = start
        # Ordered dictionaries make ties reproducible; exact duplicates merge.
        self.nodes = {start: None}
        self.edges = {}
        self.head_index = defaultdict(dict)
        self.tail_index = defaultdict(dict)
        self._cost_index = None

    def add(self, edge):
        if edge in self.edges:
            return
        self.nodes[edge.head] = None
        for tail in edge.tails:
            self.nodes[tail] = None
            self.tail_index[tail][edge] = None
        self.edges[edge] = None
        self.head_index[edge.head][edge] = None
        self._cost_index = None

    def to_json(self):
        nodeindex = {v: i for i, v in enumerate(self.nodes)}
        return json.dumps({
            "root": nodeindex[self.start],
            "nodes": [{"label": str(v)} for v in self.nodes],
            "edges": [{"head": nodeindex[e.head],
                       "tails": [nodeindex[t] for t in e.tails],
                       "weight": e.weight, "cost": e.cost}
                      for e in self.edges],
        })

    @staticmethod
    def from_json(s):
        data = json.loads(s)
        nodes = [Node(v["label"]) for v in data["nodes"]]
        forest = Forest(nodes[data["root"]])
        for e in data["edges"]:
            forest.add(Edge(nodes[e["head"]],
                            tuple(nodes[t] for t in e["tails"]),
                            e.get("weight", 1.0), e.get("cost")))
        return forest

    def buckets(self):
        """Reachable SCCs, tail-first; these are NOT score buckets."""
        adjacency = {v: tuple(t for e in self.head_index.get(v, ())
                              for t in e.tails) for v in self.nodes}
        seen, finish = {self.start}, []
        exhausted = object()
        stack = [(self.start, iter(adjacency[self.start]))]
        while stack:
            node, children = stack[-1]
            tail = next(children, exhausted)
            if tail is exhausted:
                finish.append(node)
                stack.pop()
            elif tail not in seen:
                seen.add(tail)
                stack.append((tail, iter(adjacency[tail])))
        reverse = defaultdict(list)
        for node in finish:
            for tail in adjacency[node]:
                reverse[tail].append(node)
        assigned, components = set(), []
        for node in reversed(finish):
            if node in assigned:
                continue
            component, todo = [], [node]
            assigned.add(node)
            while todo:
                v = todo.pop()
                component.append(v)
                for parent in reverse[v]:
                    if parent not in assigned:
                        assigned.add(parent)
                        todo.append(parent)
            components.append(component)
        return list(reversed(components))

    def compile(self, max_cost=None, scale=1000, *, progress=None, histogram_buckets=0):
        """Compile exact costs, optionally retaining only the lowest K levels."""
        self._cost_index = CostIndex(self, max_cost, scale, progress=progress,
                                     histogram_buckets=histogram_buckets)
        return self._cost_index

    def derivation(self, rank, fun):
        """Decode a zero-based rank using a previously compiled index."""
        if self._cost_index is None:
            raise RuntimeError("call forest.compile(...) before decoding")
        return self._cost_index.derivation(rank, fun)


class Bucket:
    """One cost coefficient and its cumulative provenance block sizes."""
    __slots__ = ("count", "ends", "choices")

    def __init__(self, count):
        self.count = count
        self.ends = []
        self.choices = []

    def add(self, edge, child_costs, count):
        self.ends.append((self.ends[-1] if self.ends else 0) + count)
        self.choices.append((edge, child_costs))

    def select(self, rank):
        block = bisect_right(self.ends, rank)
        previous = self.ends[block - 1] if block else 0
        edge, child_costs = self.choices[block]
        return edge, child_costs, rank - previous


class CostIndex:
    def __init__(self, forest, max_cost=None, scale=1000, *, build_provenance=True,
                 progress=None, histogram_buckets=0, node_limits=None):
        if max_cost is not None and (
            not isinstance(max_cost, int) or isinstance(max_cost, bool)
            or max_cost < 0
        ):
            raise ValueError("max_cost must be a nonnegative integer or None")
        if (not isinstance(histogram_buckets, int) or isinstance(histogram_buckets, bool)
                or histogram_buckets < 0):
            raise ValueError("histogram_buckets must be a nonnegative integer")
        quantize_cost(1.0, scale)  # Validate scale, including explicit-cost use.
        self.start = forest.start
        self.max_cost = max_cost
        self._progress = progress
        self.histogram_buckets = histogram_buckets
        self._node_limits = node_limits or {}
        if progress:
            started = progress.start("edge costs")
        self.costs = {e: e.cost if e.cost is not None
                      else quantize_cost(e.weight, scale) for e in forest.edges}
        if progress:
            progress.finish("edge costs", started, f"edges={len(self.costs):,}")
            started = progress.start("SCC discovery")
        components = forest.buckets()
        if progress:
            progress.finish("SCC discovery", started, f"components={len(components):,}")
            started = progress.start("polynomial counting")
            if histogram_buckets:
                progress.note(f"exact lowest-cost histograms: max_buckets={histogram_buckets}")
        self.H = {}
        for number, component in enumerate(components, 1):
            if progress:
                component_started = perf_counter()
            self._component(forest, component)
            if progress and perf_counter() - component_started >= progress.interval:
                progress.finish(f"counting component {number}/{len(components)}",
                                component_started, f"head={component[0]}, nodes={len(component):,}")
        if progress:
            progress.finish("polynomial counting", started, f"nodes={len(self.H):,}")
            started = progress.start("sorting cost bins")
        self.H = {v: dict(sorted(h.items())) for v, h in self.H.items()}
        if progress:
            progress.finish("sorting cost bins", started,
                            f"bins={sum(len(h) for h in self.H.values()):,}")
        self.buckets = None
        if build_provenance:
            self._provenance(forest)
        if progress:
            started = progress.start("root cumulative counts")
        root = self.H[self.start]
        self.root_costs = tuple(root)
        self.root_ends = tuple(accumulate(root.values()))
        self.total = self.root_ends[-1] if self.root_ends else 0
        if progress:
            progress.finish("root cumulative counts", started,
                            f"bins={len(root):,}, derivations={self.total:,}")

    def _component(self, forest, component):
        """Solve H_v = B_v + sum_u K_vu * H_u in a linear SCC.

        K contains products of already computed external child polynomials.
        Positive-cost transitions advance cost. Zero-cost transitions must
        form a DAG and are processed in topological order at each cost.
        Only nonzero coefficients enter the work queue. Polynomial products
        use bounded dense FFTs when suitable; propagation itself is sparse.
        """
        members = set(component)
        constants = {v: {} for v in component}
        transitions = defaultdict(list)
        cyclic = False
        for node in component:
            for edge in forest.head_index.get(node, ()):
                if self._progress and self._progress.due():
                    sizes = [len(self.H[t]) for t in edge.tails if t not in members]
                    self._progress.note(f"counting: completed_nodes={len(self.H):,}; "
                                        f"head={node}; external_child_bins={sizes}")
                internal = [t for t in edge.tails if t in members]
                if internal and self.histogram_buckets:
                    raise CycleDetected("capped histograms require an acyclic forest")
                if len(internal) > 1:
                    raise CycleDetected("nonlinear cyclic SCC: unroll it first")
                cyclic |= bool(internal)
                limit = self._node_limits.get(node, self.max_cost)
                edge_limit = None if limit is None else limit - self.costs[edge]
                factor = {0: 1}
                for tail in edge.tails:
                    if tail not in members:
                        factor = poly_mul(factor, self.H[tail], edge_limit)
                        if self.histogram_buckets:
                            factor = compress_histogram(factor, self.histogram_buckets)
                factor = poly_shift(factor, self.costs[edge], limit)
                if internal:
                    transitions[internal[0]].append((node, factor))
                else:
                    constants[node] = poly_add(constants[node], factor)
                    if self.histogram_buckets:
                        constants[node] = compress_histogram(constants[node], self.histogram_buckets)
        if not cyclic:
            self.H.update(constants)
            return
        if self.max_cost is None:
            raise CycleDetected("cyclic forests require a finite max_cost")

        zero_next = defaultdict(list)
        indegree = dict.fromkeys(component, 0)
        for tail, arcs in transitions.items():
            for head, kernel in arcs:
                if kernel.get(0, 0):
                    zero_next[tail].append(head)
                    indegree[head] += 1
        ready = deque(v for v in component if indegree[v] == 0)
        order = {}
        while ready:
            node = ready.popleft()
            order[node] = len(order)
            for head in zero_next[node]:
                indegree[head] -= 1
                if indegree[head] == 0:
                    ready.append(head)
        if len(order) != len(component):
            raise CycleDetected("zero-cost cycle: finite coefficients not guaranteed")

        pending, agenda = {}, []

        def add(node, cost, count):
            if not count or cost > self._node_limits.get(node, self.max_cost):
                return
            key = (node, cost)
            if key not in pending:
                heapq.heappush(agenda, (cost, order[node], node))
                pending[key] = 0
            pending[key] += count

        for node in component:
            self.H[node] = {}
            for cost, count in constants[node].items():
                add(node, cost, count)
        while agenda:
            cost, _, node = heapq.heappop(agenda)
            count = pending.pop((node, cost))
            self.H[node][cost] = count
            for head, kernel in transitions[node]:
                for increment, multiplicity in kernel.items():
                    add(head, cost + increment, count * multiplicity)

    def _restrict_exact(self, max_cost, node_limits):
        """Tighten an exact count-only index before constructing provenance."""
        if self.buckets is not None:
            raise ValueError("only exact counts without provenance can be restricted")
        if self.max_cost is not None and max_cost > self.max_cost:
            raise ValueError("cannot expand a previously counted cost slice")
        self.max_cost = max_cost
        self._node_limits = node_limits
        self.H = {node: {cost: count for cost, count in h.items()
                         if cost <= node_limits.get(node, max_cost)}
                  for node, h in self.H.items()}
        root = self.H[self.start]
        self.root_costs = tuple(root)
        self.root_ends = tuple(accumulate(root.values()))
        self.total = self.root_ends[-1] if self.root_ends else 0

    def _provenance(self, forest):
        progress = self._progress
        if progress:
            started = progress.start("provenance bucket allocation")
        self.buckets = {v: {c: Bucket(n) for c, n in h.items()}
                        for v, h in self.H.items()}
        if progress:
            progress.finish("provenance bucket allocation", started)
            started = progress.start("provenance construction")
        examined = blocks = 0
        for node in self.H:
            for edge in forest.head_index.get(node, ()):
                supports = [self.H[t].items() for t in edge.tails]
                for split in product(*supports):
                    if progress:
                        examined += 1
                        if examined % 4096 == 0 and progress.due():
                            progress.note(f"provenance: examined_splits={examined:,}, "
                                          f"blocks={blocks:,}; head={node}")
                    child_costs = tuple(c for c, _ in split)
                    cost = self.costs[edge] + sum(child_costs)
                    if cost not in self.buckets[node]:
                        continue
                    count = 1
                    for _, n in split:
                        count *= n
                    self.buckets[node][cost].add(edge, child_costs, count)
                    if progress:
                        blocks += 1
            for bucket in self.buckets[node].values():
                if not bucket.ends or bucket.ends[-1] != bucket.count:
                    raise AssertionError("provenance and polynomial disagree")
        if progress:
            progress.finish("provenance construction", started,
                            f"examined_splits={examined:,}, blocks={blocks:,}")

    def locate(self, rank):
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.total:
            raise IndexError(f"rank {rank} outside [0, {self.total})")
        bucket = bisect_right(self.root_ends, rank)
        previous = self.root_ends[bucket - 1] if bucket else 0
        return self.root_costs[bucket], rank - previous

    def derivation(self, rank, fun):
        """Binary-search blocks, divmod child ranks, visit only this tree.

        O(log(root support) + sum_visited log(bucket choices)) searches,
        plus child-rank integer arithmetic and the callback's output work.
        Integer operations are NOT assumed constant time for huge ranks.
        """
        cost, local_rank = self.locate(rank)
        if self.buckets is None:
            raise RuntimeError("provenance must be built before decoding")
        todo = [(False, self.start, cost, local_rank)]
        values = []
        while todo:
            reduce, node_or_edge, cost, rank = todo.pop()
            if reduce:
                edge = node_or_edge
                arity = len(edge.tails)
                if arity:
                    children = values[-arity:]
                    del values[-arity:]
                else:
                    children = []
                values.append(fun(edge, children))
                continue
            edge, child_costs, residual = self.buckets[node_or_edge][cost].select(rank)
            ranks = [0] * len(edge.tails)
            for j in range(len(edge.tails) - 1, -1, -1):
                radix = self.H[edge.tails[j]][child_costs[j]]
                residual, ranks[j] = divmod(residual, radix)
            assert residual == 0
            todo.append((True, edge, 0, 0))
            for tail, c, r in reversed(list(zip(edge.tails, child_costs, ranks))):
                todo.append((False, tail, c, r))
        assert len(values) == 1
        return values[0]

    def statistics(self):
        return {"indexed_derivations": self.total,
                "root_cost_bins": len(self.root_costs),
                "all_cost_bins": sum(len(h) for h in self.H.values()),
                "provenance_blocks": sum(len(b.choices)
                                         for bs in (self.buckets or {}).values()
                                         for b in bs.values())}


def tree(edge, tailvalues):
    if not tailvalues:
        return str(edge.head)
    return "(%s %s)" % (edge.head, " ".join(tailvalues))


@dataclass(frozen=True, slots=True)
class EarleyNode:
    e: Edge
    d: int
    i: int
    j: int

    def __str__(self):
        return f"[{self.e}:{self.d},{self.i},{self.j}]"


@dataclass(frozen=True, slots=True)
class EarleyGoal:
    def __str__(self):
        return "[Goal]"


def earleytree(edge, tailvalues):
    if not tailvalues:
        return ""
    if isinstance(edge.head, EarleyGoal):
        return "(%s%s)" % (edge.tails[0].e.head, tailvalues[0])
    if len(tailvalues) == 1:  # shift
        return "%s %s" % (tailvalues[0], edge.tails[0].e.tails[edge.tails[0].d])
    return "%s (%s%s)" % (tailvalues[0], edge.tails[1].e.head, tailvalues[1])


def earley(grammar, words, *, progress=None):
    """Same packed Earley deductions; deterministic FIFO chart construction.

    We need every hyperedge, so no Viterbi priority is required. Costs are
    charged once at axiom/predict; structural deductions have zero cost.
    """
    words = tuple(words)
    if progress:
        started = progress.start(f"Earley chart ({len(words)} tokens)")
    agenda, scheduled, chart = deque(), set(), set()
    doneindex, nextindex = defaultdict(dict), defaultdict(dict)
    goal = EarleyGoal()
    forest = Forest(goal)

    def agendaadd(edge):
        forest.add(edge)
        if edge.head not in scheduled:
            scheduled.add(edge.head)
            agenda.append(edge.head)

    for e in grammar.head_index[grammar.start]:
        agendaadd(Edge(EarleyNode(e, 0, 0, 0), (), e.weight, e.cost))
    while agenda:
        trigger = agenda.popleft()
        chart.add(trigger)
        if progress and progress.due():
            progress.note(f"Earley chart: processed={len(chart):,}, "
                          f"pending={len(agenda):,}, edges={len(forest.edges):,}")
        if trigger == goal:
            continue
        if (trigger.e.head == grammar.start and trigger.d == len(trigger.e.tails)
                and trigger.i == 0 and trigger.j == len(words)):
            agendaadd(Edge(goal, (trigger,), 1.0, 0))
        if trigger.d == len(trigger.e.tails):
            doneindex[trigger.i, trigger.e.head][trigger] = None
            for nextnode in nextindex[trigger.e.head, trigger.i]:
                head = EarleyNode(nextnode.e, nextnode.d + 1, nextnode.i, trigger.j)
                agendaadd(Edge(head, (nextnode, trigger), 1.0, 0))
        else:
            symbol = trigger.e.tails[trigger.d]
            if isinstance(symbol, Nonterminal):
                for e in grammar.head_index[symbol]:
                    agendaadd(Edge(EarleyNode(e, 0, trigger.j, trigger.j),
                                   (), e.weight, e.cost))
                nextindex[symbol, trigger.j][trigger] = None
                for donenode in doneindex[trigger.j, symbol]:
                    head = EarleyNode(trigger.e, trigger.d + 1, trigger.i, donenode.j)
                    agendaadd(Edge(head, (trigger, donenode), 1.0, 0))
            elif trigger.j < len(words) and symbol == words[trigger.j]:
                head = EarleyNode(trigger.e, trigger.d + 1, trigger.i, trigger.j + 1)
                agendaadd(Edge(head, (trigger,), 1.0, 0))
    if progress:
        progress.finish("Earley chart", started,
                        f"states={len(chart):,}, edges={len(forest.edges):,}, "
                        f"accepted={goal in chart}")
    return forest if goal in chart else None


def load_cnf(path, start="START"):
    """Read unweighted ``A -> B C`` / ``A -> token`` rules in file order.

    Alternatives for each head have equal probability. Duplicate productions
    merge before assigning probabilities. Symbols are classified in two
    passes, so forward references and uppercase terminals such as NAME work.
    """
    path = Path(path)
    productions = {}
    heads = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, arrow, body = line.partition("->")
        head = head.strip()
        tails = tuple(body.split())
        if not arrow or len(head.split()) != 1 or len(tails) not in (1, 2):
            raise ValueError(f"{path}:{lineno}: expected A -> token or A -> B C")
        productions.setdefault((head, tails), lineno)
        heads.setdefault(head, Nonterminal(head))
    if not productions:
        raise ValueError(f"{path}: no productions found")
    if start not in heads:
        raise ValueError(f"{path}: unknown start symbol {start!r}")

    alternatives = defaultdict(int)
    for head, tails in productions:
        lineno = productions[head, tails]
        if len(tails) == 1 and tails[0] in heads:
            raise ValueError(f"{path}:{lineno}: unit production is not CNF")
        if len(tails) == 2 and any(t not in heads for t in tails):
            raise ValueError(f"{path}:{lineno}: binary rule requires two nonterminals")
        alternatives[head] += 1

    grammar = Forest(heads[start])
    for head, tails in productions:
        children = tuple(heads[t] for t in tails) if len(tails) == 2 else tails
        grammar.add(Edge(heads[head], children, 1.0 / alternatives[head]))
    return grammar


def cnf_length_forest(grammar, max_length, *, exact_length=False, progress=None):
    """Unroll a strict CNF grammar into an acyclic forest of 1..N tokens.

    Grammar production costs are preserved. A state (symbol, length) exists
    only if it can derive that many terminals, and root alternatives choose
    each productive length with zero additional cost. This counts derivations;
    an ambiguous grammar can produce the same token sequence more than once.
    """
    if (not isinstance(max_length, int) or isinstance(max_length, bool)
            or max_length < 1):
        raise ValueError("max_length must be a positive integer")

    if progress:
        started = progress.start(f"CNF forest through length {max_length}")
    lexical, binary = [], []
    for edge in grammar.edges:
        if len(edge.tails) == 1 and not isinstance(edge.tails[0], Nonterminal):
            lexical.append(edge)
        elif len(edge.tails) == 2 and all(
                isinstance(tail, Nonterminal) for tail in edge.tails):
            binary.append(edge)
        else:
            raise ValueError("length unrolling requires a strict CNF grammar")

    lengths = range(max_length, max_length + 1) if exact_length else range(1, max_length + 1)
    root = Node(f"{grammar.start} ({max_length if exact_length else f'1..{max_length}'} tokens)")
    forest = Forest(root)
    productive = set()
    for edge in lexical:
        terminal = edge.tails[0]
        state = (edge.head, 1)
        forest.add(Edge(terminal, (), cost=0))
        forest.add(Edge(state, (terminal,), edge.weight, edge.cost))
        productive.add(state)

    for length in range(2, max_length + 1):
        for number, edge in enumerate(binary, 1):
            if progress and progress.due():
                progress.note(f"CNF forest: length={length}/{max_length}, "
                              f"rule={number}/{len(binary)}, nodes={len(forest.nodes):,}, "
                              f"edges={len(forest.edges):,}")
            for left_length in range(1, length):
                left = (edge.tails[0], left_length)
                right = (edge.tails[1], length - left_length)
                if left in productive and right in productive:
                    state = (edge.head, length)
                    forest.add(Edge(state, (left, right), edge.weight, edge.cost))
                    productive.add(state)

    for length in lengths:
        state = (grammar.start, length)
        if state in productive:
            forest.add(Edge(root, (state,), cost=0))
    if progress:
        progress.finish(f"CNF forest through length {max_length}", started,
                        f"nodes={len(forest.nodes):,}, edges={len(forest.edges):,}")
    return forest


def cnf_top1_forest(grammar, max_length=None, *, words=None, exact_length=False,
                    scale=1000, max_cost=None, progress=None):
    """Compile only minimum-cost derivations for a strict CNF workload.

    Generation considers lengths 1..max_length, or only max_length when
    exact_length=True. Supplying words instead builds a fixed-input chart.
    All tied best derivations retain exact counts and deterministic ranks.
    The returned forest uses cnf_tokens for either workload.

    The first pass stores one scalar cost per chart state. Only optimal
    edges reachable from an optimal root are materialized, then counted.
    """
    quantize_cost(1.0, scale)
    if max_cost is not None and (not isinstance(max_cost, int)
                                or isinstance(max_cost, bool) or max_cost < 0):
        raise ValueError("max_cost must be a nonnegative integer or None")
    if words is not None:
        if max_length is not None or exact_length:
            raise ValueError("words cannot be combined with a length bound")
        words = tuple(words)
        size = len(words)
        if not size:
            raise ValueError("strict CNF cannot derive an empty token sequence")
    else:
        if (not isinstance(max_length, int) or isinstance(max_length, bool)
                or max_length < 1):
            raise ValueError("max_length must be a positive integer")
        size = max_length

    symbols = tuple(n for n in grammar.nodes if isinstance(n, Nonterminal))
    number = {symbol: i for i, symbol in enumerate(symbols)}
    lexical = defaultdict(list)
    binary = []
    rule_costs = {}
    for edge in grammar.edges:
        cost = edge.cost if edge.cost is not None else quantize_cost(edge.weight, scale)
        rule_costs[edge] = cost
        if (len(edge.tails) == 1
                and not isinstance(edge.tails[0], Nonterminal)):
            lexical[edge.tails[0]].append((edge.head, cost))
        elif (len(edge.tails) == 2
              and all(isinstance(t, Nonterminal) for t in edge.tails)):
            binary.append((edge.head, *edge.tails, cost))
        else:
            raise ValueError("top-1 chart requires a strict CNF grammar")

    phase = "top-1 token chart" if words is not None else "top-1 length chart"
    if progress:
        started = progress.start(phase)
    if words is None:
        # Integer minima remain exact; exceptionally large user-supplied
        # costs use Python integers instead of overflowing fixed-width sums.
        infinity = (2 * size - 1) * max(rule_costs.values(), default=0) + 1
        dtype = np.int64 if 3 * infinity <= np.iinfo(np.int64).max else object
        chart = np.full((size + 1, len(symbols)), infinity, dtype=dtype)
        for alternatives in lexical.values():
            for head, cost in alternatives:
                h = number[head]
                chart[1, h] = min(chart[1, h], cost)
        if binary:
            heads = np.asarray([number[r[0]] for r in binary], dtype=np.intp)
            left = np.asarray([number[r[1]] for r in binary], dtype=np.intp)
            right = np.asarray([number[r[2]] for r in binary], dtype=np.intp)
            costs = np.asarray([r[3] for r in binary], dtype=dtype)
            for length in range(2, size + 1):
                # One vectorized min-plus product across rules and splits.
                candidates = chart[1:length, left] + chart[length-1:0:-1, right]
                best = candidates.min(axis=0) + costs
                np.minimum.at(chart[length], heads, best)
                if progress and progress.due():
                    progress.note(f"top-1 length chart: length={length}/{size}")
        start_id = number[grammar.start]
        lengths = range(size, size + 1) if exact_length else range(1, size + 1)
        best_cost = min(int(chart[length, start_id]) for length in lengths)
        roots = [(grammar.start, length) for length in lengths
                 if chart[length, start_id] == best_cost and best_cost < infinity]

        def state_cost(state):
            return int(chart[state[1], number[state[0]]])

        def children(state, edge):
            _, length = state
            if len(edge.tails) == 1:
                if length == 1:
                    yield edge.tails
                return
            a, b = edge.tails
            wanted = state_cost(state) - rule_costs[edge]
            sums = chart[1:length, number[a]] + chart[length-1:0:-1, number[b]]
            for split in np.flatnonzero(sums == wanted):
                left_length = int(split) + 1
                yield ((a, left_length), (b, length - left_length))

        root_label = (f"{grammar.start} ({size} tokens, top-1)" if exact_length
                      else f"{grammar.start} (1..{size} tokens, top-1)")
        state_count = int(np.count_nonzero(chart < infinity))
    else:
        # Sparse CKY: the fixed tokens usually permit only a small fraction
        # of nonterminals at any span. Index rules by their left child.
        chart = [[{} for _ in range(size + 1)] for _ in range(size)]
        by_left = defaultdict(list)
        for head, left, right, cost in binary:
            by_left[left].append((right, head, cost))
        for i, token in enumerate(words):
            cell = chart[i][i + 1]
            for head, cost in lexical.get(token, ()):
                old = cell.get(head)
                if old is None or cost < old:
                    cell[head] = cost
        for length in range(2, size + 1):
            for i in range(size - length + 1):
                j = i + length
                cell = chart[i][j]
                for split in range(i + 1, j):
                    lhs, rhs = chart[i][split], chart[split][j]
                    if not lhs or not rhs:
                        continue
                    for left, a in lhs.items():
                        for right, head, cost in by_left[left]:
                            b = rhs.get(right)
                            if b is None:
                                continue
                            candidate = cost + a + b
                            old = cell.get(head)
                            if old is None or candidate < old:
                                cell[head] = candidate
            if progress and progress.due():
                progress.note(f"top-1 token chart: length={length}/{size}")
        best_cost = chart[0][size].get(grammar.start)
        roots = [(grammar.start, 0, size)] if best_cost is not None else []

        def state_cost(state):
            return chart[state[1]][state[2]][state[0]]

        def children(state, edge):
            _, i, j = state
            if len(edge.tails) == 1:
                if j == i + 1 and edge.tails[0] == words[i]:
                    yield edge.tails
                return
            a, b = edge.tails
            wanted = state_cost(state) - rule_costs[edge]
            for split in range(i + 1, j):
                left = chart[i][split].get(a)
                right = chart[split][j].get(b)
                if left is not None and right is not None and left + right == wanted:
                    yield ((a, i, split), (b, split, j))

        root_label = f"{grammar.start} ({size} fixed tokens, top-1)"
        state_count = sum(len(cell) for row in chart for cell in row)
    if progress:
        progress.finish(phase, started, f"states={state_count:,}")
    if not roots:
        raise ValueError("no derivations for the requested tokens or length")
    if max_cost is not None and best_cost > max_cost:
        raise ValueError("no derivations within the requested cost bound")

    if progress:
        started = progress.start("top-1 optimal forest")
    forest = Forest(Node(root_label))
    pending = list(roots)
    seen = set()
    for state in roots:
        forest.add(Edge(forest.start, (state,), cost=0))
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        target = state_cost(state)
        # Grammar rule order, then ascending split, matches the length
        # forest's deterministic tie order without materializing other edges.
        for edge in grammar.head_index.get(state[0], ()):
            cost = rule_costs[edge]
            if len(edge.tails) == 1 and cost != target:
                continue
            for tails in children(state, edge):
                forest.add(Edge(state, tails, edge.weight, edge.cost))
                if len(edge.tails) == 1:
                    forest.add(Edge(tails[0], (), cost=0))
                else:
                    pending.extend(t for t in tails if t not in seen)
    if progress:
        progress.finish("top-1 optimal forest", started,
                        f"nodes={len(forest.nodes):,}, edges={len(forest.edges):,}")
    index = CostIndex(forest, best_cost, scale, progress=progress, histogram_buckets=1)
    forest._cost_index = index
    return forest, index


def cnf_tokens(edge, tailvalues):
    """Render a length-unrolled derivation as whitespace-separated terminals."""
    return " ".join(tailvalues) if tailvalues else str(edge.head)


def cnf_tree(edge, tailvalues):
    """Render a CNF span forest as a bracketed parse tree."""
    if not tailvalues:
        return str(edge.head)
    if isinstance(edge.head, Node):
        return tailvalues[0]
    return f"({edge.head[0]} {' '.join(tailvalues)})"


class CNFLengthBounds:
    """Exact least cost beyond a CNF yield-length bound.

    All rule costs must be nonnegative. Productive zero-cost recursion is
    rejected because it would put infinitely many derivations in a cost
    bucket, preventing a cost-ordered bijection with the natural numbers.
    """

    def __init__(self, grammar, scale=1000):
        infinity = float("inf")
        lexical, binary = [], []
        for edge in grammar.edges:
            cost = (edge.cost if edge.cost is not None
                    else quantize_cost(edge.weight, scale))
            if len(edge.tails) == 1 and not isinstance(edge.tails[0], Nonterminal):
                lexical.append((edge.head, cost))
            elif len(edge.tails) == 2 and all(
                    isinstance(tail, Nonterminal) for tail in edge.tails):
                binary.append((edge.head, *edge.tails, cost))
            else:
                raise ValueError("length bounds require a strict CNF grammar")

        # Dijkstra's algorithm for AND/OR rules: a binary rule can offer a
        # candidate only after both child minima have been finalized. Repeated
        # children occur twice in the incidence lists and contribute twice.
        incidence = defaultdict(list)
        remaining, accumulated = [], []
        for index, (_, left, right, cost) in enumerate(binary):
            incidence[left].append(index)
            incidence[right].append(index)
            remaining.append(2)
            accumulated.append(cost)
        queue = [(cost, index, head)
                 for index, (head, cost) in enumerate(lexical)]
        heapq.heapify(queue)
        serial = len(queue)
        self.minimum = {}
        while queue:
            cost, _, node = heapq.heappop(queue)
            if node in self.minimum:
                continue
            self.minimum[node] = cost
            for index in incidence[node]:
                remaining[index] -= 1
                accumulated[index] += cost
                if not remaining[index]:
                    heapq.heappush(queue, (accumulated[index], serial,
                                           binary[index][0]))
                    serial += 1

        # A context is a finite root derivation with one nonterminal hole.
        # Following one child pays the rule and the cheapest other child.
        adjacency = defaultdict(list)
        for head, left, right, cost in binary:
            if left in self.minimum and right in self.minimum:
                adjacency[head].append((left, cost + self.minimum[right]))
                adjacency[head].append((right, cost + self.minimum[left]))
        self.context = {}
        queue = [(0, 0, grammar.start)] if grammar.start in self.minimum else []
        serial = 1
        while queue:
            cost, _, node = heapq.heappop(queue)
            if node in self.context:
                continue
            self.context[node] = cost
            for child, extra in adjacency[node]:
                if child not in self.context:
                    heapq.heappush(queue, (cost + extra, serial, child))
                    serial += 1

        # A reachable cycle of zero-cost contexts can be repeated without
        # increasing cost. Unproductive or unreachable cycles do not matter.
        indegree = {node: 0 for node in self.context}
        for node in self.context:
            for child, cost in adjacency[node]:
                if cost == 0:
                    indegree[child] += 1
        pending = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            for child, cost in adjacency[node]:
                if cost == 0:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        pending.append(child)
        if visited != len(indegree):
            raise CycleDetected("productive zero-cost CNF cycle has infinitely "
                                "many derivations in a cost bucket")

        self._lexical = {}
        for head, cost in lexical:
            if head in self.context:
                self._lexical[head] = min(self._lexical.get(head, infinity), cost)
        self._binary = [rule for rule in binary
                        if rule[0] in self.context
                        and rule[1] in self.minimum
                        and rule[2] in self.minimum]
        self._exact = {node: [infinity] for node in self.context}
        self._max_length = 0

    def node_limits(self, forest, max_cost):
        """Prune costs that cannot fit alongside even the cheapest root context."""
        if max_cost is None:
            return {}
        return {node: max_cost - self.context[node[0]]
                for node in forest.nodes
                if isinstance(node, tuple) and node[0] in self.context}

    def beyond(self, max_length):
        """Return the minimum root cost for more than max_length tokens.

        Return infinity if no such derivation exists. Exact minima for shorter
        lengths are retained and extended when a larger bound is requested.
        """
        if (not isinstance(max_length, int) or isinstance(max_length, bool)
                or max_length < 1):
            raise ValueError("max_length must be a positive integer")
        infinity = float("inf")
        if not self.context:
            return infinity
        while self._max_length < max_length:
            length = self._max_length + 1
            for costs in self._exact.values():
                costs.append(infinity)
            if length == 1:
                for head, cost in self._lexical.items():
                    self._exact[head][length] = cost
            else:
                for head, left, right, cost in self._binary:
                    left_costs, right_costs = self._exact[left], self._exact[right]
                    best = self._exact[head][length]
                    for split in range(1, length):
                        a, b = left_costs[split], right_costs[length - split]
                        if a != infinity and b != infinity:
                            best = min(best, cost + a + b)
                    self._exact[head][length] = best
            self._max_length = length

        suffixes = {}
        for node, costs in self._exact.items():
            suffix = [infinity] * (max_length + 2)
            for length in range(max_length, 0, -1):
                suffix[length] = min(costs[length], suffix[length + 1])
            suffixes[node] = suffix

        # Every too-long tree has a bottommost subtree crossing the bound:
        # both children have <= max_length tokens, but together they exceed
        # it. Attach their minimum-cost trees to a minimum-cost root context.
        # This both bounds every omitted tree and constructs a tree attaining
        # the bound, so the certificate is exact rather than an estimate.
        best = infinity
        for head, left, right, cost in self._binary:
            base = self.context[head] + cost
            left_costs, right_suffix = self._exact[left], suffixes[right]
            for length in range(1, max_length + 1):
                a, b = left_costs[length], right_suffix[max_length + 1 - length]
                if a != infinity and b != infinity:
                    best = min(best, base + a + b)
        return best


def compile_cnf_top1(grammar, scale=1000, max_cost=None, *, progress=None):
    """Compile every globally minimum-cost CNF derivation without length trials.

    CNFLengthBounds computes exact inside minima and rejects productive
    zero-cost context cycles. Keep only rules attaining their head's minimum,
    starting at the grammar root. Every reachable node in this forest has one
    exact cost coefficient, including the complete multiplicity of ties.

    Ties use yield length first, then grammar rule insertion order, ascending
    child lengths, and mixed-radix child ranks. This matches length-unrolled
    generation when the bin cap is increased. Only attainable lengths of
    optimal derivations are tracked; no intervening length range is swept.
    The caller should reject requested ranks outside this retained cost bin;
    such a rank does not imply exhaustion of the grammar's entire language.
    """
    quantize_cost(1.0, scale)
    if max_cost is not None and (
            not isinstance(max_cost, int) or isinstance(max_cost, bool)
            or max_cost < 0):
        raise ValueError("max_cost must be a nonnegative integer or None")
    if progress:
        started = progress.start("global minimum-cost CNF derivations")
    bounds = CNFLengthBounds(grammar, scale)
    optimal = Forest(grammar.start)
    minimum = bounds.minimum.get(grammar.start)
    if minimum is not None and (max_cost is None or minimum <= max_cost):
        pending = [grammar.start]
        visited = set()
        while pending:
            head = pending.pop()
            if head in visited:
                continue
            visited.add(head)
            for edge in grammar.head_index.get(head, ()):
                rule_cost = (edge.cost if edge.cost is not None
                             else quantize_cost(edge.weight, scale))
                children = tuple(t for t in edge.tails if isinstance(t, Nonterminal))
                if any(t not in bounds.minimum for t in children):
                    continue
                total_cost = rule_cost + sum(bounds.minimum[t] for t in children)
                if total_cost != bounds.minimum[head]:
                    continue
                optimal.add(edge)
                for tail in edge.tails:
                    if isinstance(tail, Nonterminal):
                        pending.append(tail)
                    else:
                        optimal.add(Edge(tail, (), cost=0))
    # The optimal dependency graph is acyclic: a productive cycle among
    # optimal rules would be a zero-cost context cycle rejected above.
    # Sparse length sets avoid constructing states for impossible lengths.
    lengths = {}
    for component in optimal.buckets():
        if len(component) != 1:
            raise AssertionError("optimal CNF rules unexpectedly contain a cycle")
        head = component[0]
        attainable = set()
        for edge in optimal.head_index.get(head, ()):
            if not edge.tails:
                attainable.add(1)  # A terminal leaf contributes one token.
            elif len(edge.tails) == 1:
                attainable.update(lengths[edge.tails[0]])
            else:
                left, right = edge.tails
                attainable.update(a + b for a in lengths[left] for b in lengths[right])
        lengths[head] = attainable

    forest = Forest(Node(f"{grammar.start} (global minimum cost, top-1)"))
    roots = [(grammar.start, length) for length in sorted(lengths[grammar.start])]
    pending = list(roots)
    for state in roots:
        forest.add(Edge(forest.start, (state,), cost=0))
    visited = set()
    while pending:
        state = pending.pop()
        if state in visited:
            continue
        visited.add(state)
        head, length = state
        for edge in optimal.head_index.get(head, ()):
            if len(edge.tails) == 1:
                if length == 1:
                    forest.add(Edge(state, edge.tails, edge.weight, edge.cost))
                    forest.add(Edge(edge.tails[0], (), cost=0))
                continue
            left, right = edge.tails
            for left_length in sorted(lengths[left]):
                right_length = length - left_length
                if right_length not in lengths[right]:
                    continue
                tails = ((left, left_length), (right, right_length))
                forest.add(Edge(state, tails, edge.weight, edge.cost))
                pending.extend(tail for tail in tails if tail not in visited)
    if progress:
        progress.finish("global minimum-cost CNF derivations", started,
                        f"minimum_cost={minimum}, nodes={len(forest.nodes):,}, "
                        f"edges={len(forest.edges):,}")
    # Filtering optimal rules makes every support a singleton intrinsically;
    # this exact compilation cannot reconstruct any discarded cost levels.
    index = CostIndex(forest, max_cost, scale, progress=progress, histogram_buckets=1)
    forest._cost_index = index
    return forest, index


def compile_cnf_rank(grammar, rank, scale=1000, max_cost=None, *, progress=None,
                     histogram_buckets=0):
    """Compile a complete cost prefix covering any requested finite rank.

    Positive histogram_buckets retains only that many distinct lowest costs
    per node, with exact counts and provenance. A requested rank outside this
    prefix raises IndexError; the cap is never silently removed. Zero keeps
    all cost levels needed for the requested rank. One bin uses the grammar's
    global minimum directly, without sweeping through token lengths.
    Returns (forest, index, largest included token length); the last item is
    None for global one-bin generation, which does not sweep a length bound.
    """
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError("rank must be an integer")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if (not isinstance(histogram_buckets, int) or isinstance(histogram_buckets, bool)
            or histogram_buckets < 0):
        raise ValueError("histogram_buckets must be a nonnegative integer")
    if max_cost is not None and (
            not isinstance(max_cost, int) or isinstance(max_cost, bool) or max_cost < 0):
        raise ValueError("max_cost must be a nonnegative integer or None")
    if histogram_buckets == 1:
        forest, index = compile_cnf_top1(grammar, scale, max_cost, progress=progress)
        if index.total <= rank:
            raise IndexError(f"rank {rank} outside top-1 cost bin [0, {index.total}); "
                             "increase --histogram-buckets or use 0 for unrestricted costs")
        return forest, index, None
    if progress:
        started = progress.start("minimum costs and root contexts")
    bounds = CNFLengthBounds(grammar, scale)
    if progress:
        progress.finish("minimum costs and root contexts", started,
                        f"productive={len(bounds.minimum):,}, reachable={len(bounds.context):,}")
    length = 1
    limit = max_cost
    while True:
        if progress:
            trial_started = progress.start(f"trial length={length}, max_cost={limit}")
        forest = cnf_length_forest(grammar, length, progress=progress)
        node_limits = bounds.node_limits(forest, limit)
        index = CostIndex(forest, limit, scale, build_provenance=False, progress=progress,
                          histogram_buckets=histogram_buckets, node_limits=node_limits)
        if progress:
            started = progress.start("longer-derivation cost bound")
        unseen_cost = bounds.beyond(length)
        if progress:
            progress.finish("longer-derivation cost bound", started,
                            f"minimum_unseen_cost={unseen_cost}")
            progress.finish(f"trial length={length}", trial_started,
                            f"retained_derivations={index.total:,}, requested_rank={rank:,}, "
                            f"root_bins={len(index.root_costs):,}")
        if index.total > rank:
            target_cost = index.locate(rank)[0]
            limit = target_cost if limit is None else min(limit, target_cost)
            if progress:
                progress.note(f"rank_cost_upper_bound={limit}, "
                              f"minimum_unseen_cost={unseen_cost}; complete={unseen_cost > limit}")
        elif histogram_buckets and len(index.root_costs) == histogram_buckets:
            # Certify the entire retained prefix before reporting that its
            # exact count is too small. Longer trees may improve that prefix.
            target_cost = index.root_costs[-1]
            limit = target_cost if limit is None else min(limit, target_cost)
        if unseen_cost == float("inf") or (limit is not None and unseen_cost > limit):
            if progress:
                started = progress.start("retained-cost provenance")
            if index.total <= rank:
                if histogram_buckets:
                    raise IndexError(f"rank {rank} outside retained cost bins [0, {index.total}); "
                                     "increase --histogram-buckets or use 0 for unrestricted costs")
                raise IndexError(f"rank {rank} outside [0, {index.total}); "
                                 "the grammar or requested cost slice is finite")
            target_cost = index.locate(rank)[0]
            index._restrict_exact(target_cost, bounds.node_limits(forest, target_cost))
            index._provenance(forest)
            forest._cost_index = index
            if progress:
                progress.finish("retained-cost provenance", started,
                                f"max_length={length}, max_cost={target_cost}")
            return forest, index, length
        if progress:
            progress.note(f"extending to length={length + 1}, max_cost={limit}")
        length += 1


def example_grammar():
    s, np, npx, vp, n, v, vv = (Nonterminal(x) for x in "S NP NPX VP N V VV".split())
    f = Forest(s)
    f.add(Edge(s, (np, vp), 1.0))
    f.add(Edge(vp, (v, np), 1.0))
    f.add(Edge(np, (n,), 0.7))
    f.add(Edge(np, (npx,), 0.3))
    f.add(Edge(npx, (np,), 1.0))
    f.add(Edge(n, ("John",), 0.5))
    f.add(Edge(n, ("Mary",), 0.5))
    f.add(Edge(v, ("saw",), 1.0))
    return f


def self_test():
    # Compare FFT products with exact arithmetic, including truncated tails
    # that would alias into low costs without full linear-convolution padding.
    from random import Random

    rng = Random(0)
    # Compare the scalar top-1 charts against independently compiled full
    # forests, including ties and reconstruction for fixed input sequences.
    s, a = Nonterminal("S"), Nonterminal("A")
    for _ in range(12):
        grammar = Forest(s)
        for head in (s, a):
            for token in ("x", "y"):
                grammar.add(Edge(head, (token,), cost=rng.randrange(8)))
            for left in (s, a):
                for right in (s, a):
                    grammar.add(Edge(head, (left, right), cost=rng.randrange(8)))
        for length in range(1, 5):
            full = cnf_length_forest(grammar, length, exact_length=True)
            expected = full.compile()
            top, actual = cnf_top1_forest(grammar, length, exact_length=True)
            cost = expected.root_costs[0]
            assert actual.H[top.start] == {cost: expected.H[full.start][cost]}
            assert all(len(h) <= 1 for h in actual.H.values())
            for rank in sorted({0, actual.total // 2, actual.total - 1}):
                sample = full.derivation(rank, cnf_tokens)
                assert top.derivation(rank, cnf_tokens) == sample
                parsed = earley(grammar, sample.split())
                reference = parsed.compile()
                pf, pi = cnf_top1_forest(grammar, words=sample.split())
                pc = reference.root_costs[0]
                assert pi.H[pf.start] == {pc: reference.H[parsed.start][pc]}
                assert pf.derivation(0, cnf_tokens) == sample

    for _ in range(100):
        a = {c: rng.randrange(1, 100) for c in rng.sample(range(30), 12)}
        b = {c: rng.randrange(1, 100) for c in rng.sample(range(30), 10)}
        for limit in (None, -1, 0, 15, 30, 60):
            assert poly_mul(a, b, limit) == _poly_mul_sparse(a, b, limit)
    for a, b in (
        ({10**12 + 1204 * i: i + 1 for i in range(8)},
         {10**12 + 7 + 1204 * i: 9 - i for i in range(8)}),
        ({0: 2**80, 1: 2**81}, {0: 2**90, 1: 3}),
        ({0: 1, 10**12: 2}, {0: 3, 10**12 + 1: 4}),
        ({5: 2**100}, {2: 3, 7: 4}),
    ):
        for limit in (None, min(a) + min(b), max(a) + max(b) - 1):
            actual = poly_mul(a, b, limit)
            assert actual == _poly_mul_sparse(a, b, limit)
            assert all(type(n) is int and n > 0 for n in actual.values())

    from math import comb
    from tempfile import TemporaryDirectory

    # Fixed-length recursion may have zero-cost rules: ties remain finite
    # and integer ranks can greatly exceed machine precision.
    grammar = Forest(s)
    grammar.add(Edge(s, (s, s), cost=0))
    grammar.add(Edge(s, ("x",), cost=0))
    grammar.add(Edge(s, ("y",), cost=0))
    top, index = cnf_top1_forest(grammar, 90, exact_length=True)
    assert index.total == comb(178, 89) // 90 * 2**90
    assert top.derivation(index.total - 1, cnf_tokens).split() == ["y"] * 90
    huge_cost = Forest(s)
    huge_cost.add(Edge(s, (s, s), cost=10**100))
    huge_cost.add(Edge(s, ("x",), cost=10**100))
    _, index = cnf_top1_forest(huge_cost, 5, exact_length=True)
    assert index.root_costs == (9 * 10**100,) and index.total == 14

    # Top-K keeps exact minimum coefficients, without merging higher costs.
    dense = dict.fromkeys(range(100), 1)
    assert compress_histogram({1: 1, 100: 1}, 1) == {1: 1}
    for cap in (1, 4, 16, 256):
        compressed = compress_histogram(dense, cap)
        assert compressed == dict.fromkeys(range(min(cap, 100)), 1)

    # Discarded higher costs do not contribute to counts in the first bin.
    s, a = Nonterminal("S"), Nonterminal("A")
    grammar = Forest(s)
    grammar.add(Edge(s, (a, a), cost=0))
    grammar.add(Edge(a, ("x",), cost=0))
    grammar.add(Edge(a, ("y",), cost=1))
    trial = CostIndex(cnf_length_forest(grammar, 2), 1,
                      build_provenance=False, histogram_buckets=1)
    assert trial.total == 1
    assert all(len(h) <= 1 for h in trial.H.values())
    exact_forest = cnf_length_forest(grammar, 2)
    exact = exact_forest.compile(1)
    for cap in (0, 2, 256):
        forest, index, _ = compile_cnf_rank(grammar, 2, max_cost=1, histogram_buckets=cap)
        assert index.total == exact.total == 3
        for rank in range(3):
            assert index.locate(rank) == exact.locate(rank)
            assert forest.derivation(rank, cnf_tokens) == exact_forest.derivation(rank, cnf_tokens)
    try:
        compile_cnf_rank(grammar, 1, histogram_buckets=1)
    except IndexError as exc:
        assert "top-1" in str(exc)
    else:
        raise AssertionError("top-1 silently expanded to higher costs")

    # Equal-cost yields of different lengths retain their rank when the bin
    # cap grows. Put the longer rule first to distinguish the tie orders.
    s, a, b = (Nonterminal(name) for name in ("S", "A", "B"))
    grammar = Forest(s)
    grammar.add(Edge(s, (a, b), cost=0))
    grammar.add(Edge(s, ("x",), cost=2))
    grammar.add(Edge(s, ("y",), cost=2))
    grammar.add(Edge(s, (s, s), cost=3))
    grammar.add(Edge(a, ("a",), cost=0))
    grammar.add(Edge(b, ("b",), cost=2))
    grammar.add(Edge(b, ("c",), cost=2))
    for cap in (0, 1, 2):
        forest, index, _ = compile_cnf_rank(grammar, 3, histogram_buckets=cap)
        assert index.total == 4 and index.root_costs == (2,)
        assert [forest.derivation(rank, cnf_tokens) for rank in range(4)] == [
            "x", "y", "a b", "a c"]

    with TemporaryDirectory() as directory:
        path = Path(directory) / "grammar.cnf"
        # Forward references, uppercase terminals, punctuation, and deduplication.
        path.write_text("# fixture\nSTART -> A B\nA -> NAME\nB -> |\n"
                        "B -> NUMBER\nB -> NUMBER\n", encoding="utf-8")
        grammar = load_cnf(path)
        assert len(grammar.edges) == 4
        assert len(grammar.head_index) == 3
        parsed = earley(grammar, ["NAME", "|"])
        assert parsed.compile().total == 1
        forest = cnf_length_forest(grammar, 2)
        assert forest.compile().total == 2
        assert {forest.derivation(i, cnf_tokens) for i in range(2)} == {
            "NAME |", "NAME NUMBER"}
        assert cnf_length_forest(grammar, 1).compile().total == 0
        assert str(load_cnf(path, "A").start) == "A"
        try:
            load_cnf(path, "missing")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown start symbol accepted")

        # Nonlinear recursion must become finite via length unrolling. For
        # S -> SS | a | b there are Catalan(n-1) * 2**n derivations of length n.
        path.write_text("START -> START START\nSTART -> a\nSTART -> b\n",
                        encoding="utf-8")
        forest = cnf_length_forest(load_cnf(path), 4)
        index = forest.compile()
        rule_cost = quantize_cost(1 / 3)
        assert index.H[forest.start] == {
            rule_cost: 2, 3 * rule_cost: 4, 5 * rule_cost: 16, 7 * rule_cost: 80}
        samples = [forest.derivation(i, cnf_tokens) for i in range(index.total)]
        assert len(samples) == 102
        assert len(set(samples)) == 30
        assert all(1 <= len(sample.split()) <= 4 for sample in samples)
        assert forest.compile(3 * rule_cost).total == 6

        # A large requested rank is covered without a fixed length ceiling.
        grammar = load_cnf(path)
        forest, index, length = compile_cnf_rank(grammar, 1_000_000_000)
        totals = [comb(2 * (n - 1), n - 1) // n * 2**n
                  for n in range(1, length + 1)]
        assert sum(totals[:-1]) <= 1_000_000_000 < sum(totals) == index.total
        assert index.locate(1_000_000_000)[0] == (2 * length - 1) * rule_cost
        assert len(forest.derivation(1_000_000_000, cnf_tokens).split()) == length
        small, small_index, _ = compile_cnf_rank(grammar, 10)
        for rank in range(11):
            assert small_index.locate(rank) == index.locate(rank)
            assert small.derivation(rank, cnf_tokens) == forest.derivation(rank, cnf_tokens)

        for bad in ("", "START a", "START ->", "START -> A B C",
                    "START -> A\nA -> a", "START -> A Missing\nA -> a"):
            path.write_text(bad, encoding="utf-8")
            try:
                load_cnf(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid CNF accepted: {bad!r}")

    # Enough short trees does not certify a cost rank: longer can be cheaper.
    s, a, b = (Nonterminal(name) for name in ("S", "A", "B"))
    grammar = Forest(s)
    grammar.add(Edge(s, ("expensive",), cost=10))
    grammar.add(Edge(s, (a, b), cost=0))
    grammar.add(Edge(a, ("a",), cost=0))
    grammar.add(Edge(b, ("b",), cost=0))
    bounded = cnf_length_forest(grammar, 1)
    assert bounded.compile().locate(0)[0] == 10
    forest, index, length = compile_cnf_rank(grammar, 0)
    assert length == 2 and index.locate(0)[0] == 0
    assert forest.derivation(0, cnf_tokens) == "a b"
    for rank, max_cost in ((2, None), (1, 0)):
        try:
            compile_cnf_rank(grammar, rank, max_cost=max_cost)
        except IndexError:
            pass
        else:
            raise AssertionError("finite language/cost-slice exhaustion not detected")

    # Zero-cost contexts would give an infinite bucket and no cost ordering.
    grammar.add(Edge(s, (s, b), cost=0))
    try:
        compile_cnf_rank(grammar, 0)
    except CycleDetected:
        pass
    else:
        raise AssertionError("productive zero-cost recursion accepted")

    # Independent exhaustive enumeration on a finite ambiguous-cost DAG.
    f = Forest("S")
    f.add(Edge("S", ("B", "C"), cost=2))
    for head, choices in (("B", (("b0", 0), ("b1", 0), ("b2", 1))),
                          ("C", (("c0", 0), ("c1", 0), ("c2", 0),
                                 ("c3", 1), ("c4", 1), ("c5", 1), ("c6", 1)))):
        for label, cost in choices:
            f.add(Edge(head, (label,), cost=cost))
            f.add(Edge(label, (), cost=0))

    def enumerate_all(node):
        result = []
        for e in f.head_index[node]:
            for children in product(*(enumerate_all(t) for t in e.tails)):
                result.append((e.cost + sum(c for c, _ in children),
                               tree(e, [value for _, value in children])))
        return result

    expected = enumerate_all(f.start)
    index = f.compile()
    actual = [(index.locate(i)[0], f.derivation(i, tree)) for i in range(index.total)]
    assert index.H["S"] == {2: 6, 3: 11, 4: 4}
    assert len(actual) == len(set(actual)) == 21
    assert sorted(actual) == sorted(expected)
    assert [c for c, _ in actual] == sorted(c for c, _ in actual)
    for cap in (1, 2, 3):
        limited = f.compile(histogram_buckets=cap)
        assert limited.H["S"] == dict(list(index.H["S"].items())[:cap])
        assert all(len(h) <= cap for h in limited.H.values())
        assert [(limited.locate(i)[0], f.derivation(i, tree))
                for i in range(limited.total)] == actual[:limited.total]
    assert poly_add({0: 2, 1: 1}, {0: 3, 1: 4}) == {0: 5, 1: 5}
    assert poly_mul({0: 2, 1: 1}, {0: 3, 1: 4}) == {0: 6, 1: 11, 2: 4}
    assert poly_mul({}, {0: 1}) == {}
    for bad in (-1, index.total):
        try:
            index.locate(bad)
        except IndexError:
            pass
        else:
            raise AssertionError("out-of-range rank accepted")
    f.compile(3)
    assert [f.derivation(i, tree) for i in range(17)] == [s for _, s in actual[:17]]
    assert Forest.from_json(f.to_json()).compile().total == 21

    # Huge exact counts with few blocks and no recursion-limit dependence.
    big = Forest(80)
    big.add(Edge(0, (), cost=0))
    for bit in ("0", "1"):
        big.add(Edge(bit, (), cost=0))
    big.add(Edge("bit", ("0",), cost=0))
    big.add(Edge("bit", ("1",), cost=0))
    for i in range(1, 81):
        big.add(Edge(i, ("bit", i - 1), cost=0))
    huge = big.compile()
    assert huge.total == 2**80

    def bits(e, children):
        return "".join(children) if children else (e.head if e.head in ("0", "1") else "")

    assert big.derivation(2**80 - 1, bits) == "1" * 80
    assert big.derivation(2**79, bits) == "1" + "0" * 79

    # Unary SCC with external multiplicities: H = 2 + 2*z*H.
    cyclic = Forest("A")
    cyclic.add(Edge("X", ("x",), cost=0))
    cyclic.add(Edge("X", ("y",), cost=0))
    cyclic.add(Edge("x", (), cost=0))
    cyclic.add(Edge("y", (), cost=0))
    cyclic.add(Edge("A", ("X",), cost=0))
    cyclic.add(Edge("A", ("A", "X"), cost=1))
    ci = cyclic.compile(4)
    assert ci.H["A"] == {i: 2**(i + 1) for i in range(5)}
    assert len({cyclic.derivation(i, tree) for i in range(ci.total)}) == ci.total

    for tails, cost in ((("A",), 0), (("A", "A"), 1)):
        invalid = Forest("A")
        invalid.add(Edge("A", (), cost=0))
        invalid.add(Edge("A", tails, cost=cost))
        try:
            invalid.compile(3)
        except CycleDetected:
            pass
        else:
            raise AssertionError("unsupported cyclic grammar accepted")

    parsed = earley(example_grammar(), "John saw Mary".split())
    base = 2 * quantize_cost(.7) + 2 * quantize_cost(.5)
    step = quantize_cost(.3)
    demo = parsed.compile(base + 20 * step)
    assert demo.H[parsed.start] == {base + k * step: k + 1 for k in range(21)}
    samples = [parsed.derivation(i, earleytree) for i in range(demo.total)]
    assert len(set(samples)) == demo.total == 231
    assert [s.count("(NPX ") for s in samples] == [k for k in range(21) for _ in range(k + 1)]
    assert earley(example_grammar(), "John Mary".split()) is None
    assert Forest("empty").compile().total == 0
    nullable_node = Forest("root")
    nullable_node.add(Edge("root", (None,), cost=0))
    nullable_node.add(Edge(None, (), cost=0))
    assert nullable_node.compile().total == 1
    print("All self-tests passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path,
                        default=Path(__file__).with_name("python.cnf"),
                        help="CNF grammar file (default: python.cnf beside this script)")
    parser.add_argument("--start", default="START", help="grammar start symbol")
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--max-length", type=int,
                          help="restrict generation to 1..N tokens (default: grow for --rank)")
    workload.add_argument("--length", type=int, help="generate exactly N tokens")
    workload.add_argument("--tokens", help="parse a quoted, whitespace-separated token sequence")
    parser.add_argument("--rank", type=int,
                        help="zero-based final sample rank (default: last retained with a "
                             "length/input constraint, otherwise 0; unrestricted costs: 1,000,000,000)")
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--max-cost", type=int,
                        help="explicit integer-cost ceiling (default: grow for --rank)")
    parser.add_argument("--histogram-buckets", type=int, default=1,
                        help="retain K lowest exact cost levels per node (default: 1, top-1; "
                             "0 for unrestricted costs); includes every tie within a level")
    parser.add_argument("--quiet-timings", action="store_true",
                        help="suppress live preparation timings and progress")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.rank is not None and args.rank < 0:
        parser.error("rank must be nonnegative")
    if (args.scale <= 0 or (args.max_length is not None and args.max_length <= 0)
            or (args.length is not None and args.length <= 0)):
        parser.error("scale and length bounds must be positive")
    if args.max_cost is not None and args.max_cost < 0:
        parser.error("max-cost must be nonnegative")
    if args.histogram_buckets < 0:
        parser.error("histogram-buckets must be nonnegative")

    start = perf_counter()
    try:
        grammar = load_cnf(args.grammar, args.start)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    load_seconds = perf_counter() - start
    print(f"*** grammar: {args.grammar}, start={args.start}, "
          f"rules={len(grammar.edges)}")
    print("*** uniform rule probabilities per nonterminal; "
          "integer costs: round(-scale * ln(probability))")

    start = perf_counter()
    progress = None if args.quiet_timings else ProgressTimings()
    automatic = args.tokens is None and args.max_length is None and args.length is None
    final_rank = args.rank
    length = args.length if args.length is not None else args.max_length
    try:
        if automatic:
            if final_rank is None:
                final_rank = 0 if args.histogram_buckets else 1_000_000_000
            print(f"*** preparing cost prefix through rank {final_rank:,}; "
                  f"bins={args.histogram_buckets or 'unrestricted'}", flush=True)
            result, index, length = compile_cnf_rank(
                grammar, final_rank, args.scale, args.max_cost, progress=progress,
                histogram_buckets=args.histogram_buckets)
            callback = cnf_tokens
            mode = "generate"
        elif args.histogram_buckets == 1:
            result, index = cnf_top1_forest(
                grammar, length, words=None if args.tokens is None else args.tokens.split(),
                exact_length=args.length is not None, scale=args.scale,
                max_cost=args.max_cost, progress=progress)
            callback = cnf_tokens if args.tokens is None else cnf_tree
            mode = "generate" if args.tokens is None else "parse"
        else:
            if args.tokens is None:
                result = cnf_length_forest(grammar, length,
                                           exact_length=args.length is not None, progress=progress)
                callback = cnf_tokens
                mode = "generate"
            else:
                result = earley(grammar, args.tokens.split(), progress=progress)
                callback = earleytree
                mode = "parse"
                if result is None:
                    parser.error("token sequence is not accepted by this grammar/start symbol")
            index = result.compile(args.max_cost, args.scale, progress=progress,
                                   histogram_buckets=args.histogram_buckets)
    except (ValueError, IndexError) as exc:
        parser.error(str(exc))
    prepare_seconds = perf_counter() - start
    print("*** FFT cost-counting index:")
    print(json.dumps({"mode": mode,
                      "rank_scope": "complete cost prefix" if automatic else "bounded forest",
                      "max_length": length if mode == "generate" else None,
                      "exact_length": args.length,
                      "token_count": len(args.tokens.split()) if mode == "parse" else None,
                      "max_cost": index.max_cost, "scale": args.scale,
                      "histogram_buckets": args.histogram_buckets,
                      **index.statistics(),
                      "load_seconds": load_seconds,
                      "prepare_seconds": prepare_seconds}, indent=2))
    if not index.total:
        parser.error("no derivations within the requested length/cost bounds")
    final_rank = index.total - 1 if final_rank is None else final_rank
    if final_rank >= index.total:
        parser.error(f"rank {final_rank} outside [0, {index.total}); "
                     "increase --histogram-buckets or use 0 for unrestricted costs")
    print("*** indexing is zero-based")
    for rank in sorted({0, final_rank}):
        print(f"*** sample (rank={rank:,}):")
        start = perf_counter()
        sample = result.derivation(rank, callback)
        decode_seconds = perf_counter() - start
        print(sample)
        print(f"cost={index.locate(rank)[0]}, "
              f"decode_seconds={decode_seconds:.6f}")

    ranks = sorted({0, final_rank} | {1 << bit for bit in range(final_rank.bit_length())})
    for rank in ranks:
        start = perf_counter()
        result.derivation(rank, callback)
        us = (perf_counter() - start) * 1e6
        print(f"rank {rank:>10,}: {us:8.1f} μs")


if __name__ == "__main__":
    main()
