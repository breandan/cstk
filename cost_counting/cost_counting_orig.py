#!/usr/bin/env python3
"""Cost-counting reference decoder; Python 3.10+, standard library only.

Run:
    python cost_counting.py
    python cost_counting.py --self-test
    python cost_counting.py --rank 1000000 --scale 1000

The grammar, probabilities, input, callbacks, and zero-based sample ranks
match the supplied Huang--Chiang demonstration. That grammar has infinitely
many parses of ONE word sequence. We index a finite cost slice, so this demo
is a bijection onto derivations, not distinct strings. On an unambiguous
forest the same index is a bijection onto strings.

H[v][c] counts derivations of v of EXACT integer cost c. Addition is pointwise
addition; multiplication is convolution. All coefficients and ranks are
Python integers. Only the one-time conversion of probabilities to integer
costs uses floating point: c(e) = round(-scale * log(p(e))). Explicit integer
costs may instead be supplied as Edge(..., cost=...).

Compilation is eager and bounded by max_cost, including provenance blocks.
Queries never enumerate earlier derivations or extend the index. Within a
cost bucket, ties use forest edge insertion order, then lexicographic child
costs, then mixed-radix child ranks. This differs from the supplied decoder's
process-dependent id(edge) tie order.

Supported forests: arbitrary acyclic hypergraphs and linear cyclic SCCs
(at most one tail of an edge is in its head's SCC), including the demo's
unary cycles. Zero-cost cycles and nonlinear cyclic SCCs are rejected.
For an acyclic finite forest, use max_cost=None to include its entire support.
For a cyclic forest, an explicit finite max_cost is required.

Cost quantization can change probability order in general. For this exact
demo every parse with k unary cycles has probability .1225 * .3**k and
cost 2*c(.7) + 2*c(.5) + k*c(.3), so its probability ordering is preserved
whenever c(.3)>0. The supplied floating-point product decoder underflows in
the tail; use log scores or exact arithmetic for a full baseline comparison.

Storage includes ONE provenance block per (edge, child-cost tuple), not one
record per derivation. This can still be large: in this particular demo each
split of the cycle count at the top binary edge identifies a single parse.
No claim of a universal memory or preprocessing improvement is intended.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import time
from itertools import accumulate, product
import json
from math import isfinite, isqrt, log
from time import perf_counter


class CycleDetected(ValueError):
    pass


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


def poly_mul(a, b, limit=None):
    result = defaultdict(int)
    for ca, na in a.items():
        for cb, nb in b.items():
            cost = ca + cb
            if limit is None or cost <= limit:
                result[cost] += na * nb
    return dict(result)


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
        """Reachable SCCs, tail-first; these are NOT score buckets.

        Iterative Kosaraju traversal avoids recursion limits in large DAGs.
        """
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

    def compile(self, max_cost=None, scale=1000):
        """Build the ENTIRE requested slice before any rank is decoded."""
        self._cost_index = CostIndex(self, max_cost, scale)
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
    def __init__(self, forest, max_cost=None, scale=1000):
        if max_cost is not None and (
            not isinstance(max_cost, int) or isinstance(max_cost, bool)
            or max_cost < 0
        ):
            raise ValueError("max_cost must be a nonnegative integer or None")
        quantize_cost(1.0, scale)  # Validate scale, including explicit-cost use.
        self.start = forest.start
        self.max_cost = max_cost
        self.costs = {e: e.cost if e.cost is not None
                      else quantize_cost(e.weight, scale) for e in forest.edges}
        self.H = {}
        for component in forest.buckets():
            self._component(forest, component)
        self.H = {v: dict(sorted(h.items())) for v, h in self.H.items()}
        self.buckets = {v: {c: Bucket(n) for c, n in h.items()}
                        for v, h in self.H.items()}
        self._provenance(forest)
        root = self.H[self.start]
        self.root_costs = tuple(root)
        self.root_ends = tuple(accumulate(root.values()))
        self.total = self.root_ends[-1] if self.root_ends else 0

    def _component(self, forest, component):
        """Solve H_v = B_v + sum_u K_vu * H_u in a linear SCC.

        K contains products of already computed external child polynomials.
        Positive-cost transitions advance cost. Zero-cost transitions must
        form a DAG and are processed in topological order at each cost.
        Only nonzero coefficients enter the work queue, so a large numeric
        cost range by itself does not cause dense allocation or iteration.
        """
        members = set(component)
        constants = {v: {} for v in component}
        transitions = defaultdict(list)
        cyclic = False
        for node in component:
            for edge in forest.head_index.get(node, ()):
                internal = [t for t in edge.tails if t in members]
                if len(internal) > 1:
                    raise CycleDetected("nonlinear cyclic SCC: unroll it first")
                cyclic |= bool(internal)
                factor = {0: 1}
                for tail in edge.tails:
                    if tail not in members:
                        factor = poly_mul(factor, self.H[tail], self.max_cost)
                factor = poly_shift(factor, self.costs[edge], self.max_cost)
                if internal:
                    transitions[internal[0]].append((node, factor))
                else:
                    constants[node] = poly_add(constants[node], factor)
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
            if not count or cost > self.max_cost:
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

    def _provenance(self, forest):
        for node in self.H:
            for edge in forest.head_index.get(node, ()):
                supports = [self.H[t].items() for t in edge.tails]
                for split in product(*supports):
                    child_costs = tuple(c for c, _ in split)
                    cost = self.costs[edge] + sum(child_costs)
                    if self.max_cost is not None and cost > self.max_cost:
                        continue
                    count = 1
                    for _, n in split:
                        count *= n
                    self.buckets[node][cost].add(edge, child_costs, count)
            for bucket in self.buckets[node].values():
                if not bucket.ends or bucket.ends[-1] != bucket.count:
                    raise AssertionError("provenance and polynomial disagree")

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
                                         for bs in self.buckets.values()
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


def earley(grammar, words):
    """Same packed Earley deductions; deterministic FIFO chart construction.

    We need every hyperedge, so no Viterbi priority is required. Costs are
    charged once at axiom/predict; structural deductions have zero cost.
    """
    words = tuple(words)
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
    return forest if goal in chart else None


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
    parser.add_argument("--rank", type=int, default=1_000_000,
                        help="zero-based final rank, matching the supplied demo")
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--max-cost", type=int,
                        help="override the finite demo slice's cost bound")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.rank < 0 or args.scale <= 0:
        parser.error("rank must be nonnegative and scale must be positive")

    grammar = example_grammar()
    print("*** grammar:")
    print(grammar.to_json())
    print("*** integer costs: round(-scale * ln(probability))")
    for e in grammar.edges:
        print(f"  {e.head} -> {' '.join(map(str, e.tails))}: "
              f"p={e.weight:g}, cost={quantize_cost(e.weight, args.scale)}")

    start = perf_counter()
    result = earley(grammar, "John saw Mary".split())
    parse_seconds = perf_counter() - start
    assert result is not None
    ranks = [1_000, args.rank]  # Both calls in the supplied demo are zero-based.
    base = 2 * quantize_cost(.7, args.scale) + 2 * quantize_cost(.5, args.scale)
    step = quantize_cost(.3, args.scale)
    # Demo-specific closed form chooses the bound, NOT the decoded tree.
    # There are k+1 ways to distribute k loops over the two NP occurrences.
    cycles = (isqrt(8 * max(ranks) + 1) - 1) // 2
    limit = args.max_cost if args.max_cost is not None else base + cycles * step
    if limit < 0:
        parser.error("max-cost must be nonnegative")
    start = perf_counter()
    index = result.compile(limit, args.scale)
    compile_seconds = perf_counter() - start
    if index.total <= max(ranks):
        parser.error(f"slice contains only {index.total} derivations; increase --max-cost")
    expected_max_cycles = (limit - base) // step
    assert index.H[result.start] == {
        base + k * step: k + 1 for k in range(expected_max_cycles + 1)}
    print("*** finite cost-counting index:")
    print(json.dumps({"max_cost": limit, **index.statistics(),
                      "parse_seconds": parse_seconds,
                      "compile_seconds": compile_seconds}, indent=2))
    print("*** indexing is zero-based, as in the supplied implementation")
    for rank in ranks:
        print(f"*** {rank:,}th best sample (rank={rank}):")
        start = perf_counter()
        sample = result.derivation(rank, earleytree)
        decode_seconds = perf_counter() - start
        k = (isqrt(8 * rank + 1) - 1) // 2
        assert sample.count("(NPX ") == k
        print(sample)
        print(f"cost={index.locate(rank)[0]}, unary_cycles={k}, "
              f"log_probability={log(.1225) + k * log(.3):.12f}, "
              f"decode_seconds={decode_seconds:.6f}")

    ranks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    for k in ranks:
        t0 = time.perf_counter()
        deriv = result.derivation(k, earleytree)
        us = (time.perf_counter() - t0) * 1e6
        print(f"rank {k:>10,}: {us:8.1f} μs")


if __name__ == "__main__":
    main()
