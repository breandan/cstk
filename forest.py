import numpy
import heapq, itertools, collections, operator
import json
from functools import reduce


class CycleDetected(Exception):
    pass

def product(iterable, initializer=1):
    return reduce(operator.mul, iterable, initializer)

def py2_max(x, y):
    if x is None:
        return y
    elif y is None:
        return x
    else:
        return max(x, y)

class Edge(object):
    """An Edge has a head, zero or more tails, and an optional weight.

    The weight, if provided, should support multiplication and
    addition. If you want to avoid underflow, you need to make the
    weights able to take on very small values (see, e.g., the bigfloat
    module). Moreover, the inside/outside routines for Forests convert
    weights inside cycles to floats. So your weights should be
    convertible to float and multipliable with float. Since this
    conversion happens only inside cycles, it should be safe from
    underflow unless you have extraordinarily long cycles.
    """
    def __init__(self, head, tails, weight=None):
        self.head = head
        self.tails = tails
        self.weight = weight

    def __eq__(self, other):
        return isinstance(other, Edge) and self.head == other.head and self.tails == other.tails

    def __lt__(self, other):
        if not isinstance(other, Edge):
            return False
        if self.head != other.head:
            return self.head < other.head
        return self.tails < other.tails

    def __hash__(self):
        return hash((self.head, self.tails))

    def __repr__(self):
        return "Edge(%s,%s,%s)" % (repr(self.head), repr(self.tails), repr(self.weight))

class Node(object):
    """Forest nodes aren't required to be Node instances. This is the
    vanilla implementation that is used by the JSON reader. At
    minimum, a node just has to be convertible to a str, and be
    comparable using ==. Node objects that are == are considered to be
    the same hypergraph node."""

    def __init__(self, label=None):
        self.label = label

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return "Node(%s)" % repr(self.label)

class Forest(object):
    def __init__(self, start, viterbi=False):
        self.nodes = set([start])
        self.start = start
        self.edges = set()
        self.head_index = collections.defaultdict(set)
        self.tail_index = collections.defaultdict(set)
        self.viterbi = collections.defaultdict(lambda: None) if viterbi else None

    @staticmethod
    def from_json(s):
        o = json.loads(s)
        nodes = [Node(node['label']) for node in o['nodes']]
        f = Forest(start=nodes[o['root']])
        for edge in o['edges']:
            f.add(Edge(nodes[edge['head']], tuple(nodes[tail] for tail in edge['tails'])))
        return f

    def to_json(self):
        o = {}
        nodes = []
        nodeindex = {}
        for ni,node in enumerate(self.nodes):
            nodes.append({"label" : str(node)})
            nodeindex[node] = ni
        edges = []
        for edge in self.edges:
            edges.append({"head" : nodeindex[edge.head],
                          "tails" : [nodeindex[tail] for tail in edge.tails]})
        return json.dumps({"root" : nodeindex[self.start],
                           "nodes" : nodes,
                           "edges" : edges})

    def add(self, e):
        self.nodes.add(e.head)
        self.nodes.update(e.tails)
        self.edges.add(e)
        self.head_index[e.head].add(e)
        for tail in e.tails:
            self.tail_index[tail].add(e)
        if self.viterbi is not None:
            self.viterbi[e.head] = py2_max(self.viterbi[e.head], (product(self.viterbi[tail][0] for tail in e.tails)*e.weight, e))

        # clean up memos
        # any way to make this smarter?
        if hasattr(self, "_buckets"): del self._buckets
        if hasattr(self, "_inside"): del self._inside
        if hasattr(self, "_outside"): del self._outside
        if hasattr(self, "_kbest"): del self._kbest

    ### Inside/outside

    def buckets(self):
        """Tarjan's algorithm
        http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

        Return a list of strongly connected components, tail-first (= bottom-up).
        """

        if hasattr(self, "_buckets"):
            return self._buckets

        stack = [] # contains all visited nodes not yet in a component
        visited = set()
        lowlink = {}
        components = []

        # do a DFS of the graph, forming a subtree of the graph
        def visit(v):
            dfs = lowlink[v] = len(visited)
            stack.append(v)
            visited.add(id(v))

            for edge in self.head_index[v]:
                for v1 in edge.tails:
                    if id(v1) not in visited:
                        visit(v1)

                    # lowlink = lowest-numbered node that v can reach and can reach v
                    if v1 in lowlink:
                        # i.e., if v1 in stack
                        # That means v1's component root is an ancestor of v
                        # or else it would have been removed from the stack.
                        # Therefore v can reach any node v1 can,
                        # and any node that can reach v1, can reach v.
                        lowlink[v] = min(lowlink[v], lowlink[v1])

            if lowlink[v] == dfs:
                # v is a component root
                component = []
                v1 = None
                while v1 is not v:
                    v1 = stack.pop()
                    del lowlink[v1]
                    component.append(v1)
                components.append(component)

        visit(self.start)
        self._buckets = components
        return components

    def _compute_inside(self):
        """Andreas Stolcke. 1995. An efficient probabilistic
        context-free parsing algorithm that computes prefix
        probabilities. Computational Linguistics, 21(2):165-201."""

        self._inside = {}
        buckets = self.buckets()

        if not hasattr(self, "_bucket_matrices"):
            self._bucket_matrices = [None]*len(self.buckets())

        for (bi,bucket) in enumerate(buckets):
            """
            Let A be the vector of inside probabilities of nodes in bucket.
            A can be expressed in terms of itself as:
              A = WA+C
            where W is a matrix and C is a vector.
            Then the solution for A is (I-W)^-1*C."""

            weights = numpy.zeros((len(bucket),len(bucket)))
            constants = numpy.zeros((len(bucket),))
            bucket_index = {}
            for (i,node) in enumerate(bucket):
                bucket_index[id(node)] = i
            for (i,node) in enumerate(bucket):
                for edge in self.head_index[node]:
                    cycle_j = None
                    weight = edge.weight
                    for tail in edge.tails:
                        j = bucket_index.get(id(tail), None)
                        if j is not None:
                            # we can't handle cycles in multiple tails
                            if cycle_j is not None:
                                raise CycleDetected
                            cycle_j = j
                        else:
                            weight *= self._inside[tail]

                    if cycle_j is not None:
                        # we found exactly one j with a cycle
                        weights[i][cycle_j] += weight
                    else:
                        # no cycle, fold this part of the inside prob
                        # into the constant
                        constants[i] += weight

            weights = numpy.linalg.inv(numpy.identity(len(bucket))-weights)
            # save this -- outside() will need it
            self._bucket_matrices[bi] = weights

            insides = numpy.dot(weights, constants)
            for (i,node) in enumerate(bucket):
                self._inside[node] = insides[i]

    def _compute_outside(self):
        self._outside = {}

        buckets = self.buckets()

        # initialize
        for bucket in buckets:
            for node in bucket:
                self._outside[node] = 0.

        self._outside[self.start] = 1.

        for bi in range(len(buckets)-1, -1, -1):
            bucket = buckets[bi]
            constants = numpy.empty((len(bucket),))
            for (i,node) in enumerate(bucket):
                constants[i] = self._outside[node]

            weights = self._bucket_matrices[bi]

            outsides = numpy.dot(constants, weights)
            for (j,node) in enumerate(bucket):
                self._outside[node] = outsides[j]

            # finally, add in this bucket's contribution to the lower buckets
            # this is repeated work :/
            bucket_index = {}
            for (i,node) in enumerate(bucket):
                bucket_index[id(node)] = i

            for node in bucket:
                for edge in self.head_index[node]:
                    weight = self._outside[node]*edge.weight*product(self._inside[tail] for tail in edge.tails)
                    for tail in edge.tails:
                        if id(tail) not in bucket_index:
                            self._outside[tail] += weight/self._inside[tail]

    def inside(self, node):
        """Calculate inside probability allowing for unary cycles"""
        if not hasattr(self, "_inside"):
            self._compute_inside()
        return self._inside[node]

    def outside(self, node):
        """Calculate outside probability allowing for unary cycles"""
        if not hasattr(self, "_inside"):
            self._compute_inside()
        if not hasattr(self, "_outside"):
            self._compute_outside()
        return self._outside[node]

    ### k-best derivations

    def derivation(self, rank, fun):
        """Calculate the rank'th best derivation"""
        if not hasattr(self, "_kbest"):
            # bootstrap the k-best lists to handle cycles
            self._kbest = {}
            for node in self.nodes:
                self._kbest[node] = [(self.viterbi[node][0], self.viterbi[node][1], (0,)*len(self.viterbi[node][1].tails))]
            kbest = dict((node, KBest(self, node)) for node in self.nodes)
            for node in self.nodes:
                for edge in self.head_index[node]:
                    kbest[node].push(edge)
            self._kbest = kbest
        return self.derivation_helper(self.start, rank, fun)

    def derivation_helper(self, node, rank, fun):
        (weight, edge, tailranks) = self._kbest[node][rank]
        tailvalues = [self.derivation_helper(tail, tailrank, fun) for (tail, tailrank) in zip(edge.tails, tailranks)]
        return fun(edge, tailvalues)

# Functions to plug into Forest.derivation

def tree(edge, tailvalues):
    if len(tailvalues) == 0:
        return str(edge.head)
    else:
        return "(%s %s)" % (edge.head, " ".join(tailvalues))

def earleytree(edge, tailvalues):
    if len(tailvalues) == 0:
        return ""
    elif isinstance(edge.head, EarleyGoal):
        return "(%s%s)" % (edge.tails[0].e.head, tailvalues[0])
    elif len(tailvalues) == 1: # shift
        return "%s %s" % (tailvalues[0], edge.tails[0].e.tails[edge.tails[0].d])
    elif len(tailvalues) == 2:
        return "%s (%s%s)" % (tailvalues[0], edge.tails[1].e.head, tailvalues[1])

class KBest(object):
    def __init__(self, f, v):
        self.f = f
        self.v = v
        self.cand = []
        self.index = set()
        self.result = []
        self.prevedge = self.prevranks = None

    def push(self, edge, ranks=None):
        if ranks is None:
            ranks = (0,)*len(edge.tails)
        self.index.add((edge, ranks))
        weight = edge.weight * product(self.f._kbest[tail][rank][0] for (tail,rank) in zip(edge.tails, ranks))
        heapq.heappush(self.cand, (-weight, edge, ranks))

    def __getitem__(self, rank):
        while len(self.result)-1 < rank:
            # push successors of previous node onto heap if not already there
            if self.prevedge is not None:
                edge = self.prevedge
                for (i,tail) in enumerate(edge.tails):
                    ranks = tuple(r+1 if j == i else r for (j,r) in enumerate(self.prevranks))
                    if (edge, ranks) not in self.index:
                        try:
                            self.push(edge, ranks)
                        except IndexError:
                            pass # successor doesn't exist
            if len(self.cand) > 0:
                x, self.prevedge, self.prevranks = heapq.heappop(self.cand)
                self.result.append((x,self.prevedge, self.prevranks))
            else:
                raise IndexError
        return (-self.result[rank][0], self.result[rank][1], self.result[rank][2])

class Nonterminal(object):
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return str(self.x)

    def __lt__(self, other):
        return self.x < other.x

    def __repr__(self):
        return "Nonterminal(%s)" % repr(self.x)

class EarleyNode(object):
    def __init__(self, e, d, i, j):
        self.e = e
        self.d = d
        self.i = i
        self.j = j

    def __eq__(self, other):
        return isinstance(other, EarleyNode) and self.e is other.e and (self.d, self.i, self.j) == (other.d, other.i, other.j)

    def __lt__(self, other):
        if not isinstance(other, EarleyNode):
            return False
        if self.e is not other.e:
            return self.e < other.e
        return (self.d, self.i, self.j) < (other.d, other.i, other.j)

    def __hash__(self):
        return hash((id(self.e), self.d, self.i, self.j))

    def __str__(self):
        return "[%s:%s,%s,%s]" % (self.e, self.d, self.i, self.j)

class EarleyGoal(object):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, EarleyGoal)

    def __hash__(self):
        return 0

    def __str__(self):
        return "[Goal]"

def earley(g, w):
    agenda = []

    chart = set()
    doneindex = dict(((i,x),set()) for x in g.nodes for i in range(len(w)+1))
    nextindex = dict(((x,i),set()) for x in g.nodes for i in range(len(w)+1))

    goal = EarleyGoal()
    f = Forest(goal, viterbi=True)

    def agendaadd(e):
        f.add(e)
        heapq.heappush(agenda, (-f.viterbi[e.head][0], e.head))

    for e in g.head_index[g.start]:
        axiom = EarleyNode(e, 0, 0, 0)
        agendaadd(Edge(axiom, (), e.weight))

    while len(agenda) > 0:
        (priority, trigger) = heapq.heappop(agenda)

        if trigger in chart:
            continue
        chart.add(trigger)

        if trigger == goal:
            continue

        if trigger.e.head == g.start and trigger.d == len(trigger.e.tails) and trigger.i == 0 and trigger.j == len(w):
            agendaadd(Edge(goal, (trigger,), 1.))

        if trigger.d == len(trigger.e.tails):
            doneindex[trigger.i,trigger.e.head].add(trigger)
            # complete
            for nextnode in nextindex[trigger.e.head,trigger.i]:
                head = EarleyNode(nextnode.e, nextnode.d+1, nextnode.i, trigger.j)
                agendaadd(Edge(head, (nextnode, trigger), 1.))
        else:
            next = trigger.e.tails[trigger.d]
            if isinstance(next, Nonterminal):
                # predict
                for e in g.head_index[next]:
                    head = EarleyNode(e, 0, trigger.j, trigger.j)
                    agendaadd(Edge(head, (), e.weight)) # pretend no tails
                # complete
                nextindex[next,trigger.j].add(trigger)
                for donenode in doneindex[trigger.j,next]:
                    head = EarleyNode(trigger.e, trigger.d+1, trigger.i, donenode.j)
                    agendaadd(Edge(head, (trigger, donenode), 1.))
            elif trigger.j < len(w) and next == w[trigger.j]:
                # shift
                head = EarleyNode(trigger.e, trigger.d+1, trigger.i, trigger.j+1)
                agendaadd(Edge(head, (trigger,), 1.))

    if goal in chart:
        return f
    else:
        return None

if __name__ == "__main__":
    s, np, npx, vp, n, v, vv = (Nonterminal(x) for x in "S NP NPX VP N V VV".split())

    f = Forest(s)

    f.add(Edge(s, (np, vp), 1.))
    f.add(Edge(vp, (v, np), 1.))
    f.add(Edge(np, (n,), 0.7))
    f.add(Edge(np, (npx,), 0.3))
    f.add(Edge(npx, (np,), 1.))
    f.add(Edge(n, ("John",), 0.5))
    f.add(Edge(n, ("Mary",), 0.5))
    f.add(Edge(v, ("saw",), 1.))

    print("*** grammar:")
    print(f.to_json())
    s = f.to_json()
    f1 = Forest.from_json(s)
    print("*** grammar (again):")
    print(f1.to_json())

    result = earley(f, "John saw Mary".split())
    if result:
        print("*** forest:")
        print(result.to_json())
        print("*** buckets:")
        b = result.buckets()
        for nodes in b:
            print(" ".join(str(node) for node in nodes))

    print("*** probabilities")

    for node in result.nodes:
        try:
            print("  %s inside=%s outside=%s viterbi=%s" % (node, result.inside(node), result.outside(node), result.viterbi[node][0]))
        except KeyError:
            pass

    print("*** 1000-best")

    for i in range(1000):
        print(result.derivation(i, earleytree))

