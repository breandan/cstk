#!/usr/bin/env python3
"""Cost-indexed decoding of acyclic packed forests (standard library only).

    index = forest.compile(scale=1_000_000_000, top_k=256_000)
    word = index.derivation(rank, render)

Each node stores counts and provenance by integer cost, not individual trees.
Costs are round(-scale * log(weight)), unless Edge.cost is supplied. A heap
merges child-cost combinations on demand; top_k limits every node's prefix.
Decode any retained rank by binary search and mixed-radix child unranking.

Ties use edge insertion order, then child costs, then child ranks. Omitting
top_k exhausts all cost buckets. A finite prefix may end inside a cost tie;
its H coefficients count only retained derivations. Cycles must be unrolled
first. Compilation and decoding both use explicit stacks for deep forests.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import heapq
from itertools import accumulate
from math import isfinite, log
from typing import TypeVar, cast

T = TypeVar("T")


class CycleDetected(ValueError):
    pass


def quantize_cost(weight: float | Fraction, scale: int = 1000) -> int:
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if not isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("probability must be finite and in (0, 1]")
    return round(-scale * log(weight))


@dataclass(frozen=True, slots=True)
class Edge:
    head: Hashable
    tails: tuple[Hashable, ...]
    weight: float | Fraction = 1.0
    cost: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tails", tuple(self.tails))
        if self.cost is not None and (
            not isinstance(self.cost, int) or isinstance(self.cost, bool)
            or self.cost < 0
        ):
            raise ValueError("explicit costs must be nonnegative integers")


class Node:
    def __init__(self, label: object = None) -> None:
        self.label = label

    def __str__(self) -> str:
        return str(self.label)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label!r})"


class Nonterminal(Node):
    pass


class Forest:
    def __init__(self, start: Hashable) -> None:
        self.start = start
        # Ordered dictionaries make ties reproducible; exact duplicates merge.
        self.nodes = {start: None}
        self.edges: dict[Edge, None] = {}
        self.head_index: defaultdict[Hashable, dict[Edge, None]] = defaultdict(dict)
        self._cost_index: CostIndex | None = None

    def add(self, edge: Edge) -> None:
        if edge in self.edges:
            return
        self.nodes[edge.head] = None
        for tail in edge.tails:
            self.nodes[tail] = None
        self.edges[edge] = None
        self.head_index[edge.head][edge] = None
        self._cost_index = None

    def compile(self, max_cost: int | None = None, scale: int = 1000,
                *, top_k: int | None = None) -> CostIndex:
        """Compile a cost slice, or its first top_k derivations on a DAG."""
        self._cost_index = CostIndex(self, max_cost, scale, top_k=top_k)
        return self._cost_index

    def derivation(self, rank: int, fun: Callable[[Edge, list[T]], T]) -> T:
        """Decode a zero-based rank using a previously compiled index."""
        if self._cost_index is None:
            raise RuntimeError("call forest.compile(...) before decoding")
        return self._cost_index.derivation(rank, fun)


def _minimum_costs(forest: Forest, costs: Mapping[Edge, int]) -> dict[Hashable, int | None]:
    """Reachable DAG postorder and minimum costs; None means unproductive."""
    minimum: dict[Hashable, int | None] = {}
    active: set[Hashable] = set()
    stack = [(forest.start, False)]
    while stack:
        node, ready = stack.pop()
        if node in minimum:
            continue
        edges = forest.head_index.get(node, ())
        if not ready:
            if node in active:
                raise CycleDetected("acyclic forest required; unroll cycles first")
            active.add(node)
            stack.append((node, True))
            stack.extend((tail, False) for edge in edges for tail in edge.tails)
            continue
        active.remove(node)
        best: int | None = None
        for edge in edges:
            children = [minimum[tail] for tail in edge.tails]
            if all(cost is not None for cost in children):
                cost = costs[edge] + sum(cast(list[int], children))
                best = cost if best is None else min(best, cost)
        minimum[node] = best
    return minimum


class Bucket:
    """One cost coefficient and its cumulative provenance block sizes."""
    __slots__ = ("count", "ends", "choices")

    def __init__(self) -> None:
        self.count = 0
        self.ends: list[int] = []
        self.choices: list[tuple[Edge, tuple[int, ...]]] = []

    def add(self, edge: Edge, child_costs: tuple[int, ...], count: int) -> None:
        self.count += count
        self.ends.append(self.count)
        self.choices.append((edge, child_costs))

    def select(self, rank: int) -> tuple[Edge, tuple[int, ...], int]:
        block = bisect_right(self.ends, rank)
        previous = self.ends[block - 1] if block else 0
        edge, child_costs = self.choices[block]
        return edge, child_costs, rank - previous


class _CostStream:
    """A node's finalized cost buckets and its temporary Cartesian frontier."""
    __slots__ = ("node", "costs", "total", "heap", "pending", "done")

    def __init__(self, node: Hashable) -> None:
        self.node = node
        self.costs: list[int] = []
        self.total = 0
        self.heap: list[tuple[int, int, tuple[int, ...]]] | None = None
        self.pending: list[tuple[int, int, tuple[int, ...]]] = []
        self.done = False


class _CostBuilder:
    """Demand-driven cost convolution; never enumerate individual trees.

    Each node needs at most its own first k derivations. A parent using a
    later child has at least k earlier parents, obtained by substituting the
    child's earlier derivations. This holds for both costs and native ties.
    Thus partial child buckets preserve the root prefix and mixed-radix order.
    """

    def __init__(self, index: CostIndex, forest: Forest, k: int | None) -> None:
        self.index, self.forest, self.k = index, forest, k
        self.minimum = _minimum_costs(forest, index.costs)
        self.edges = tuple(forest.edges)
        self.edge_order = {edge: i for i, edge in enumerate(self.edges)}
        self.streams: dict[Hashable, _CostStream] = {}

    def stream(self, node: Hashable) -> _CostStream:
        if node not in self.streams:
            self.streams[node] = _CostStream(node)
            self.index.H[node] = {}
            self.index.buckets[node] = {}
        return self.streams[node]

    def seed(self, state: _CostStream) -> None:
        # Minimum costs suffice to seed an edge: losing edges do not force
        # construction of their children's counts or provenance.
        state.heap = []
        for edge in self.forest.head_index.get(state.node, ()):
            children = [self.minimum[t] for t in edge.tails]
            if any(c is None for c in children):
                continue
            cost = self.index.costs[edge] + sum(cast(list[int], children))
            if self.index.max_cost is None or cost <= self.index.max_cost:
                state.heap.append((cost, self.edge_order[edge], (0,) * len(edge.tails)))
        heapq.heapify(state.heap)

    def advance(self, state: _CostStream) -> Iterator[tuple[Hashable, int]]:
        """Finalize one bucket, yielding child-bucket requests to a trampoline."""
        if state.heap is None:
            self.seed(state)
        assert state.heap is not None
        pending, state.pending = state.pending, []
        for cost, edge_id, indices in pending:
            edge = self.edges[edge_id]
            # Unique-parent lattice traversal needs no permanent visited set.
            # Increment coordinate j only if all earlier coordinates are zero.
            for j, tail in enumerate(edge.tails):
                child = self.stream(tail)
                next_index = indices[j] + 1
                yield tail, next_index
                if next_index < len(child.costs):
                    next_cost = cost + child.costs[next_index] - child.costs[indices[j]]
                    if self.index.max_cost is None or next_cost <= self.index.max_cost:
                        next_indices = indices[:j] + (next_index,) + indices[j+1:]
                        heapq.heappush(state.heap, (next_cost, edge_id, next_indices))
                if indices[j]:
                    break
        if not state.heap:
            state.done = True
            return
        cost = state.heap[0][0]
        bucket = Bucket()
        while state.heap and state.heap[0][0] == cost:
            candidate = heapq.heappop(state.heap)
            _, edge_id, indices = candidate
            edge = self.edges[edge_id]
            child_costs, count = [], 1
            for tail, bucket_index in zip(edge.tails, indices):
                yield tail, bucket_index
                child_cost = self.streams[tail].costs[bucket_index]
                child_costs.append(child_cost)
                count *= self.index.H[tail][child_cost]
                if self.k is not None:
                    count = min(self.k, count)
            kept = count if self.k is None else min(count, self.k - state.total)
            bucket.add(edge, tuple(child_costs), kept)
            state.total += kept
            if state.total == self.k:
                state.done = True
                state.heap.clear()
                state.pending.clear()
                break
            # Every successor has strictly higher cost, so none is needed to
            # finish this bucket. Expand only if another bucket is requested.
            state.pending.append(candidate)
        state.costs.append(cost)
        self.index.H[state.node][cost] = bucket.count
        self.index.buckets[state.node][cost] = bucket

    def ensure(self, node: Hashable, bucket_index: int) -> None:
        state = self.stream(node)
        if state.done or bucket_index < len(state.costs):
            return
        # Requests advance a stream by at most one bucket. An explicit stack
        # keeps deep acyclic forests independent of Python's recursion limit.
        assert bucket_index == len(state.costs)
        stack = [self.advance(state)]
        while stack:
            try:
                child, child_index = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            child_state = self.stream(child)
            if not child_state.done and child_index >= len(child_state.costs):
                assert child_index == len(child_state.costs)
                stack.append(self.advance(child_state))

    def build(self) -> None:
        # Full mode exhausts every reachable node, preserving local counts
        # even when a root cost cap excludes some child buckets.
        nodes = self.minimum if self.k is None else (self.index.start,)
        for node in nodes:
            state = self.stream(node)
            while not state.done:
                self.ensure(node, len(state.costs))
        if self.k is None:
            return
        root = self.stream(self.index.start)
        # Child lookahead can discover buckets that no retained root choice
        # uses. Keep only provenance reachable from the requested prefix.
        needed: defaultdict[Hashable, set[int]] = defaultdict(set)
        todo = [(root.node, cost) for cost in root.costs]
        edges = set()
        while todo:
            node, cost = todo.pop()
            if cost in needed[node]:
                continue
            needed[node].add(cost)
            for edge, child_costs in self.index.buckets[node][cost].choices:
                edges.add(edge)
                todo.extend(zip(edge.tails, child_costs))
        needed[root.node]  # Preserve an empty root for an unproductive forest.
        self.index.H = {node: {c: n for c, n in self.index.H[node].items() if c in costs}
                        for node, costs in needed.items()}
        self.index.buckets = {node: {c: b for c, b in self.index.buckets[node].items() if c in costs}
                              for node, costs in needed.items()}
        self.index.costs = {e: self.index.costs[e] for e in edges}


class CostIndex:
    """Compile a full cost index or a fixed top-k prefix; queries do not grow it."""

    def __init__(self, forest: Forest, max_cost: int | None = None, scale: int = 1000,
                 *, top_k: int | None = None) -> None:
        for name, value in (("max_cost", max_cost), ("top_k", top_k)):
            if value is not None and (not isinstance(value, int)
                                      or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        quantize_cost(1.0, scale)
        self.start, self.max_cost, self.top_k = forest.start, max_cost, top_k
        self.costs = {e: e.cost if e.cost is not None else quantize_cost(e.weight, scale)
                      for e in forest.edges}
        self.H: dict[Hashable, dict[int, int]] = {self.start: {}}
        self.buckets: dict[Hashable, dict[int, Bucket]] = {self.start: {}}
        if top_k == 0:
            self.costs = {}
        else:
            _CostBuilder(self, forest, top_k).build()
        root = self.H[self.start]
        self.root_costs = tuple(root)
        self.root_ends = tuple(accumulate(root.values()))
        self.total = self.root_ends[-1] if self.root_ends else 0

    def locate(self, rank: int) -> tuple[int, int]:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.total:
            raise IndexError(f"rank {rank} outside [0, {self.total})")
        bucket = bisect_right(self.root_ends, rank)
        previous = self.root_ends[bucket - 1] if bucket else 0
        return self.root_costs[bucket], rank - previous

    def derivation(self, rank: int, fun: Callable[[Edge, list[T]], T]) -> T:
        """Binary-search blocks, divmod child ranks, visit only this tree.

        O(log(root support) + sum_visited log(bucket choices)) searches,
        plus child-rank integer arithmetic and the callback's output work.
        Integer operations are NOT assumed constant time for huge ranks.
        """
        cost, local_rank = self.locate(rank)
        todo = [(False, self.start, cost, local_rank)]
        values: list[T] = []
        while todo:
            reduce, node_or_edge, cost, rank = todo.pop()
            if reduce:
                edge = cast(Edge, node_or_edge)
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

    def statistics(self) -> dict[str, int]:
        return {"indexed_derivations": self.total,
                "root_cost_bins": len(self.root_costs),
                "all_cost_bins": sum(len(h) for h in self.H.values()),
                "provenance_blocks": sum(len(b.choices)
                                         for bs in self.buckets.values()
                                         for b in bs.values())}


def tree(edge: Edge, tailvalues: Sequence[str]) -> str:
    if not tailvalues:
        return str(edge.head)
    return "(%s %s)" % (edge.head, " ".join(tailvalues))
