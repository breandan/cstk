"""Benchmark-only CNF preparation: exact first-K costs, then counted forest.

Each backend still builds its own CostIndex and uses its own poly_mul.
This helper avoids creating grammar edges that cannot enter retained bins.
"""

from collections import defaultdict
import numpy as np


class CNFTopKChart:
    """Incremental cost chart shared by both benchmark decoder backends.

    ``extend(N)`` prepares the first ``bins`` exact costs at every length
    through N. ``compile_length(N)`` counts the relevant packed forest using
    the backend's own polynomial multiplication and provenance machinery.
    No strings or derivations are enumerated during preparation.
    """

    def __init__(self, module, grammar, bins, scale=1000):
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise ValueError('bins must be a positive integer')
        module.quantize_cost(1.0, scale)
        self.module, self.grammar = module, grammar
        self.max_length, self.max_bins, self.scale = 1, bins, scale
        symbols = tuple(n for n in grammar.nodes if isinstance(n, module.Nonterminal))
        self.number = number = {symbol: i for i, symbol in enumerate(symbols)}
        grouped = defaultdict(list)
        lexical = defaultdict(list)
        self.costs = costs = {}
        for edge in grammar.edges:
            cost = edge.cost if edge.cost is not None else module.quantize_cost(edge.weight, scale)
            costs[edge] = cost
            if len(edge.tails) == 2:
                grouped[number[edge.head]].append((number[edge.tails[0]], number[edge.tails[1]], cost))
            elif len(edge.tails) == 1 and not isinstance(edge.tails[0], module.Nonterminal):
                lexical[number[edge.head]].append(cost)
            else:
                raise ValueError('strict CNF required')
        self.max_rule_cost = max(costs.values(), default=0)
        self.infinity = infinity = max(1 << 60, self.max_rule_cost + 1)
        dtype = np.int64 if 3 * infinity <= np.iinfo(np.int64).max else object
        self.chart = chart = np.full((2, len(symbols), bins), infinity, dtype=dtype)
        for head, alternatives in lexical.items():
            values = sorted(set(alternatives))[:bins]
            chart[1, head, :len(values)] = values
        indexed = []
        for head, alternatives in grouped.items():
            indexed.append((head,
                            np.asarray([r[0] for r in alternatives], dtype=np.intp),
                            np.asarray([r[1] for r in alternatives], dtype=np.intp),
                            np.asarray([r[2] for r in alternatives], dtype=dtype)))
        self.indexed = indexed

    def extend(self, length):
        """Extend only newly requested chart layers; earlier costs are reused."""
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError('length must be a positive integer')
        if length <= self.max_length:
            return
        infinity = max(self.infinity, (2 * length - 1) * self.max_rule_cost + 1)
        dtype = np.int64 if 3 * infinity <= np.iinfo(np.int64).max else object
        chart = np.full((length + 1, len(self.number), self.max_bins), infinity, dtype=dtype)
        prefix = chart[:self.max_length + 1]
        prefix[:] = self.chart
        if infinity != self.infinity:
            prefix[prefix == self.infinity] = infinity
        self.chart, self.infinity = chart, infinity
        for size in range(self.max_length + 1, length + 1):
            for head, left, right, rule_costs in self.indexed:
                a = chart[1:size, left]
                b = chart[size-1:0:-1, right]
                active = a[..., 0] + b[..., 0] < infinity
                if not np.any(active):
                    continue
                increments = np.broadcast_to(rule_costs, active.shape)[active]
                values = (a[active, :, None] + b[active, None, :]
                          + increments[:, None, None])
                values = np.unique(values)[:self.max_bins]
                values = values[values < infinity]
                chart[size, head, :len(values)] = values
        self.max_length = length

    def compile_length(self, length, bins=None):
        """Return a normal backend forest/index for exactly this token length."""
        self.extend(length)
        bins = self.max_bins if bins is None else bins
        if not isinstance(bins, int) or isinstance(bins, bool) or not 1 <= bins <= self.max_bins:
            raise ValueError('query outside prepared dimensions')
        module, grammar = self.module, self.grammar
        chart, infinity, number = self.chart, self.infinity, self.number
        costs = chart[length, number[grammar.start], :bins]
        costs = costs[costs < infinity]
        if not len(costs):
            raise ValueError('unproductive token length')
        forest = module.Forest(module.Node(f'{grammar.start} ({length} tokens)'))
        start = (grammar.start, length)
        forest.add(module.Edge(forest.start, (start,), cost=0))
        seen = set()
        pending = [start]
        while pending:
            state = pending.pop()
            if state in seen:
                continue
            seen.add(state)
            symbol, size = state
            values = chart[size, number[symbol], :bins]
            ceiling = int(values[values < infinity][-1])
            for edge in grammar.head_index.get(symbol, ()):
                cost = self.costs[edge]
                if len(edge.tails) == 1:
                    if size == 1 and cost <= ceiling:
                        forest.add(module.Edge(state, edge.tails, edge.weight, edge.cost))
                        forest.add(module.Edge(edge.tails[0], (), cost=0))
                    continue
                left, right = edge.tails
                minima = chart[1:size, number[left], 0] + chart[size-1:0:-1, number[right], 0] + cost
                for split in np.flatnonzero(minima <= ceiling):
                    left_size = int(split) + 1
                    tails = ((left, left_size), (right, size - left_size))
                    forest.add(module.Edge(state, tails, edge.weight, edge.cost))
                    pending.extend(t for t in tails if t not in seen)
        index = module.CostIndex(forest, None, self.scale, histogram_buckets=bins)
        forest._cost_index = index
        assert tuple(int(v) for v in costs) == index.root_costs
        return forest, index
