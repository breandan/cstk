# Cost-counting benchmarks

Run from the repository root:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_latency.py cost_counting/python.cnf 10..100:10
```

`benchmark_latency.py` (formerly `benchmark_cost_counting.py`) and
`benchmark_memory.py` accept only two optional positional inputs: **grammar**
and **slice range**. Ranges are inclusive: `2..32:2` means lengths 2, 4, …, 32;
`1..23` means every length from 1 through 23. A comma-separated list such as
`18,24,32` is also accepted. With no arguments, each script uses its
`GRAMMAR_PATH` and `LENGTHS` globals.

All tuning settings live together at the top of the respective script:
sample count, repeats, cost resolution, binning policy, resource limits,
worker count, output retention, and output directory. There are no tuning
CLI flags. Both scripts read weighted CNF rules with `[p/q]` probabilities or
unweighted CNF rules with uniform probabilities per head. `START = None`
selects `START` when present, otherwise `S`, otherwise the first rule's head.
Set `START` explicitly for another entry symbol.

The latency defaults measure `python.cnf` at lengths 10, 20, …, 100 using
the four maps in Section 2 of the paper:

| Setting | Map | Default parameters |
| --- | --- | --- |
| Identity | `t` | Exact costs |
| Uniform | `floor(t / Δ)` | `Δ = 1000` |
| Truncated | `floor(t / Δ)` below `mΔ`, overflow otherwise | `Δ = 1000`, `m = 4` |
| Geometric | `floor(log_ρ(1 + (ρ−1)t / Δ))` | `Δ = 1000`, `ρ = 2` |

Rule costs remain `round(-1000 * ln(p))`; unweighted grammars use uniform
probabilities per nonterminal. Binned multiplication need not be associative:
this experiment uses `x^c ⊗φ (H_B ⊗φ H_C)`, so child bins `a,b` contribute to
`φ(c + φ(a + b))`. Terminal monomials keep their raw rule costs and infinity
is absorbing. All four maps retain every derivation; identity retains every
exact cost, and truncated binning retains overflow. Counts are checked against
an independent unbinned inside recurrence.

All four settings use the original `CostIndex.locate` and
`CostIndex.derivation` methods unchanged. Parsing builds the complete
provenance index for the counted histogram: each bin stores its ordered
rule/split choices and cumulative block totals in native `Bucket` objects.
Decoding binary-searches these stored blocks and divides child ranks; it
does not scan grammar rules, enumerate bin combinations, or build provenance.

Construction is measured once per point, including a fresh chart through
the requested length, all counts, provenance construction, and cumulative
root-bin totals. Decoding is the median of three
warmed batches, each rendering the first 1,000 derivations (or the full slice
if smaller). Grammar loading, warm-up, validation and explicit garbage
collection run outside the timed regions. No answers are cached, ranks are
never repeated to fill a batch, and times are never extrapolated. Ambiguity
counts derivations rather than distinct rendered strings.

Every preparation runs in a fresh process with the same 120-second and
16-GiB budgets. An unsuccessful attempt records its status, elapsed time,
peak memory and error; it has no successful parse or decode timing. No
partial histogram or index substitutes for a failed complete construction.

New results go to `benchmark_results/latency/latency.csv` and `latency.json`.
They record complete and indexed counts, actual batch sizes, preparation and
decode times, provenance time and block count, repetitions, output checksums,
source hashes and environment. `prepare_seconds` includes both
`chart_count_seconds` and `provenance_seconds`; the latter includes the root
cumulative totals. Superseded query-scanning and adaptive-prefix experiments
are archived in `four_binnings/previous_query_scan/` and
`four_binnings/previous_adaptive_prefix/`; prior measured results stay there.

The benchmark saves standalone `latency.png`, `.svg`, and `.pdf` plots.
Four colors identify the maps; solid and dashed curves distinguish
construction and decoding on a logarithmic y-axis. Failed measurements leave
gaps. Benchmarking uses the standard library; rendering uses Matplotlib.
It no longer patches the paper's figure or caption automatically.

## Exact word-order comparison

From the repository root, run:

```sh
python3 -m venv cost_counting/.venv
cost_counting/.venv/bin/python -m pip install numpy matplotlib
cost_counting/.venv/bin/python cost_counting/benchmark_rank_correlation.py
```

`dyck.cnf` supplies the same rational, locally normalized PCFG to three
experiments: exhaustive exact scoring, `cost_counting_orig.py` polynomial decoding,
and `forest_py3.py` k-best decoding. Its productions use explicit `[p/q]`
weights, read by the experiment's loader. Both decoders use their native
cost/probability ordering. `cost_counting.py` supplies only
the common CNF forest builder and terminal rendering callback; its counting
and decoding implementation is not used in this experiment.

The grammar generates all nonempty Dyck words with round and square brackets.
It is unambiguous: the first matching bracket pair uniquely determines the
interior and the remaining suffix. Thus a word's PCFG score is its unique
derivation's product of rational rule weights. Every rule has either two
nonterminal children or one terminal, with no epsilon or unit productions.
Probabilities sum to one for each nonterminal. Omitting the empty word keeps
the grammar in strict CNF and does not affect the positive-length slices.

The default run enumerates **every word** at token lengths 2, 4, ..., 18:
2, 8, 40, 224, 1,344, 8,448, 54,912, 366,080, and 2,489,344 words respectively.
For even n the slice size is Catalan(n/2) times 2^(n/2). An independent
string dynamic program computes exact `Fraction` probabilities and sorts
them descending. Both decoders receive the same length-unrolled forest.
The forest decoder uses `Fraction` weights, including synthetic unit edges;
the polynomial decoder uses its native `round(-scale * log(p))` per-rule
costs at a fixed scale of 1000. It is instantiated explicitly as
`cost_counting_orig.CostIndex(packed, max_cost=None, scale=1000)`.
Every exact integer-exponent coefficient is retained. Addition is exact
pointwise integer addition and multiplication is exact sparse convolution;
no coarsening, merging of distinct exponents, support truncation, or cost cap
is applied. The integer conversion of rule weights is inherent in the
original decoder and is separate from polynomial coarsening.

All three rankings are exhaustive. The reference scores and sorts all
distinct words; forest decoding queries every rank from 0 through N-1 and
checks that rank N is exhausted; polynomial decoding queries every rank in
the full index. Both decoded word sets must equal the complete reference
set, with no duplicates, pruning, or sampling. `decoded_count` in the CSV
records the number of words ranked by each method.

Two comparisons handle the many equal PCFG scores explicitly:

- `rank_correlation`: Spearman correlation of exact-score average ranks
  against average ranks of each decoder's ordering key (rational probability
  for forest, integer cost for polynomial decoding). This measures agreement of the
  score orderings independently of arbitrary tie-breaking.
- `emission_rank_correlation`: Spearman correlation of exact-score average
  ranks against the actual one-based positions of words emitted by each
  decoder. Exact score ties make even a perfectly score-sorted enumeration
  fall below one. The exact curve gives this attainable ceiling; words tied
  in exact score may be emitted in any order without changing the statistic.

The polynomial decoder's native order among equal-cost derivations is preserved. Forest tie order
uses object IDs and may differ across processes, but both forest correlation
statistics are unaffected. Reference display positions break exact score
ties lexicographically. Increasing cost scale need not improve correlation
at every length: rounded rule costs can merge distinct scores or separate
exactly equal products.

Outputs are written to `cost_counting/rank_correlation/`:

- Two `.png` plots and corresponding self-contained, editable PGFPlots `.tex`
  sources, which compile with `pdflatex`.
- `summary.csv`: correlations, full slice sizes, numbers of score levels, and
  counts of adjacent emitted pairs that violate descending exact score.
- `word_ranks.csv.gz`: every word's exact rational probability, midranks,
  native emission positions, and polynomial cost at each tested scale.
- `experiment.json`: settings, metric definitions, source hashes, environment,
  and completion status.

The runner checks local normalization, strict CNF, unambiguity, balanced
brackets, Catalan-times-bracket-colors cardinalities, identical complete
word sets, each decoded score/cost, and monotone decoder keys. To run the
focused independent checks or regenerate plots from saved data:

```sh
cost_counting/.venv/bin/python -m unittest discover -s cost_counting -p 'test_rank_correlation.py'
cost_counting/.venv/bin/python cost_counting/plot_rank_correlation.py
```

Use `--lengths 2,4,6,8,10,12` or `--scales 10,100,1000` to vary one factor.
The scale option changes the initial integer rule-cost conversion; it never
coarsens the polynomial or limits its support. CSV field names retaining
`histogram` refer to the original exact polynomial decoder.
Word counts grow exponentially; the defaults intentionally keep exhaustive
comparison tractable. `--grammar` accepts another unambiguous Dyck-two PCFG
with the same terminal alphabet and explicit normalized rational weights.

Forest decoding has score-rank correlation 1 at every tested length, since
its ordering uses exact rational probabilities. At fixed cost scale 1000,
the original polynomial decoder's score-rank correlation is not monotone
over all lengths: it starts at 1 for lengths 2 and 4, falls to 0.999483 at
length 6, then rises over the larger measured slices. This is an observation
for this grammar, not a general convergence result.

The polynomial decoder respects descending exact score through length 14,
but its first genuine ordering inversions occur at length 16, despite the
higher Spearman correlation. There are two adjacent inversions at length 16.
For example, integer costs 14482 and 14483 order words with probabilities
1/1953125 and 6561/12800000000 in the wrong direction (the latter probability
is larger). Earlier score-rank discrepancies come from splitting exact
probability ties into distinct integer costs. The emission-order plot
separately shows the loss relative to the attainable exact-score tie ceiling.

Exact polynomial coefficients do not remove errors from converting rule
weights to integer costs. Each rounded rule cost has error at most 1/(2*scale)
in negative-log units. A strict CNF derivation of n terminals has 2n-1 rules,
so its total error is at most (2n-1)/(2*scale); increasing length can therefore
expose more close-score reversals. Spearman correlation depends on the whole
score distribution and need not deteriorate monotonically with this bound.

Measured full-slice results at scale 1000 (forest score-rank rho is 1 throughout):

| Length n | Words decoded by each method | Polynomial score-rank rho | Adjacent exact-score inversions |
| ---: | ---: | ---: | ---: |
| 2 | 2 | 1.000000000 | 0 |
| 4 | 8 | 1.000000000 | 0 |
| 6 | 40 | 0.999482706 | 0 |
| 8 | 224 | 0.999783592 | 0 |
| 10 | 1,344 | 0.999840562 | 0 |
| 12 | 8,448 | 0.999907449 | 0 |
| 14 | 54,912 | 0.999942686 | 0 |
| 16 | 366,080 | 0.999963317 | 2 |
| 18 | 2,489,344 | 0.999975442 | 15 |

At length 18, actual emission-position correlation is 0.999965787747 for
the polynomial decoder versus the forest/exact ceiling of 0.999965788468,
a deficit of about 7.21e-10. The emission plot's lower panel resolves this
small difference. Adjacent inversions count local violations, not all
inverted pairs. These full-slice measurements do not establish behavior at
arbitrarily large lengths.

## Lazy top-k cost counting on Dyck length slices

[`cost_counting_orig.py`](cost_counting_orig.py) is a compact, standard-library
implementation for acyclic packed forests. One cost-bucket builder handles
both full indexing and a known output prefix; decoding uses binary search and
mixed-radix child ranks. On a forest unrolled to an exact Dyck token length:

```python
index = forest.compile(scale=1000, top_k=500_000)
for rank in range(index.total):
    word = index.derivation(rank, cnf_tokens)
```

The equivalent constructor is
`cost_counting_orig.CostIndex(forest, scale=1000, top_k=500_000)`.
Omitting `top_k`, or passing `None`, builds the full cost index for the DAG.
`index.total` is `min(top_k, available_derivations)`; rank `index.total` is
out of range. `top_k=0` creates an empty index. An optional `max_cost` can
further restrict the available slice.

The prefix builder computes minimum costs, then merges cost combinations from
per-node heaps only when their buckets are requested. Each block counts many
derivations; construction does not enumerate the individual outputs. Every
node retains at most its first `top_k` derivations, clipping the final cost
bucket and provenance block when needed. Costs and the original deterministic
tie order are preserved, including when the cutoff falls inside a tie.
In this mode `H` contains **retained prefix counts**, so its final bucket is
not necessarily the complete coefficient for that cost. This is output-prefix
truncation, not cost binning or a change to integer-cost ordering.

The ordinary binary-search/mixed-radix decoder is unchanged. After compilation,
temporary heaps and child buckets unused by retained root provenance are
discarded. Memory depends on the relevant graph and cost combinations across
nodes; a per-node prefix bound does not imply a globally minimal-memory index.
Both compilation and decoding use explicit stacks for deep forests.

Compilation requires an acyclic graph in both modes. For Dyck, use the same
exact-length unrolling as the memory benchmark. A reachable cycle raises
`CycleDetected` with an unrolling hint. The standalone Earley parser, cyclic
solver, JSON helpers, and embedded demo were removed to keep this file focused
on indexing and decoding. The measured implementation is preserved in each
benchmark's `sources/` snapshot, including the
[retained-output experiment](benchmark_results/dyck_lazy_memory_256k_highres_retained/sources/cost_counting_orig.py).

To compare this mode with forest rescoring, set `SAMPLES = 500_000`,
`COST_SCALE = 1000`, `COST_TOP_K = True`, and `RETAIN_OUTPUTS = False` at the
top of `benchmark_memory.py`, then run:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py cost_counting/dyck.cnf 18,24,32
```

Set `COST_TOP_K = False` to measure the full cost index. The historical
three-repeat run is in
[`benchmark_results/dyck_lazy_memory_500k/`](benchmark_results/dyck_lazy_memory_500k/).
At length 32, median peak RSS was 40.22 MiB for top-k cost counting,
1,396.73 MiB for a fresh full-index baseline, and 1,528.30 MiB for forest
rescoring. All decoded 500,000 words. Top-k retained 45 root cost buckets and
7,706 provenance blocks, versus 8,144 buckets and 7,568,432 blocks for the
full index. All three top-k output checksums match the full-index outputs.
`full_index_n32.json` contains the fresh baseline; `validation.json` records
the comparisons, and `sources/` preserves the measured code. These results
include interpreter/import overhead and apply to this Dyck workload.

The complete even-length sweep from 2 through 32 is saved as
[`dyck_lazy_memory_500k_sweep/memory.png`](benchmark_results/dyck_lazy_memory_500k_sweep/memory.png),
with SVG/PDF versions, raw measurements, and aggregated CSV alongside it.
Lengths 2–16 exhaust their smaller slices; lengths 18–32 decode exactly
500,000 words. The plot marks this transition at length 17. Three fresh
processes contribute to each median and min–max band. Measurements at 18,
24, and 32 are reused from the matching-source run above; remaining lengths
were measured with six concurrent workers. `source_batches` in the merged
manifest records both batches, and `plot_memory.py` validates the completed
96-row dataset before regenerating the plot.

For the top-256,000 variant with a one-token length step, set
`SAMPLES = 256_000`, `COST_SCALE = 1000`, and `RETAIN_OUTPUTS = False`, then run:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py cost_counting/dyck.cnf 1..32
```

The saved [256k PNG](benchmark_results/dyck_lazy_memory_256k/memory.png) uses
a logarithmic peak-resident-memory axis and ticks at every integer length.
Odd lengths contain no words and are marked in an empty-slice strip, without
assigning them decoder memory values. Even lengths through 14 are exhaustive;
even lengths from 16 onward decode exactly 256,000 words. The standalone
`dyck_lazy_memory_256k/plot_memory.py` validates all 192 worker records and
regenerates the PNG, SVG, and PDF from the saved data.

For finer cost resolution, use `COST_SCALE = 1_000_000_000` and the
nonempty slice range `2..32:2`. The historical streaming experiment also used
`RETAIN_OUTPUTS = False` and `JOBS = 6`:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py cost_counting/dyck.cnf 2..32:2
```

This changes the per-rule integer cost to `round(-1_000_000_000 * log(p))`,
a millionfold finer resolution than the earlier scale of 1,000. The lazy
index still stores sparse exact integer-cost buckets; increasing the scale
does not allocate a dense array spanning the enlarged cost range. Forest
rescoring continues to use exact rational probabilities. Equal-probability
words can retain different tie orders between the two methods.

The [high-resolution PNG](benchmark_results/dyck_lazy_memory_256k_highres/memory.png)
uses the same logarithmic resident-memory axis, with ticks at lengths
2, 4, …, 32 and no empty-slice strip. Both methods are measured afresh, with
three repetitions per point. The standalone `plot_memory.py` in that
directory validates all 96 measurements before rendering PNG, SVG, and PDF.
Separate `ordering_audit.py` and `ordering_validation.json` artifacts check
the finer-cost ranking against exact Dyck probabilities without adding
ranking-audit memory to the measured decoder processes.

All 96 trials completed. At length 32, median peak RSS was 39.52 MiB for
lazy cost counting and 901.50 MiB for forest rescoring. The independent audit
checked all 2,368,978 high-resolution outputs across the 16 lengths, finding
zero strict probability inversions and the exact top-k probability counts
at every length. All 48 measured cost-counting output hashes match the audit.
The old scale of 1,000 reproduces the probability reversal at positions
1,087–1,088 of the length-16 prefix. These checks establish matching score
order for the measured workload; equal-probability word order and word
identity at a partial boundary tie remain unconstrained.

To include the returned top-k word list, set `RETAIN_OUTPUTS = True`.
This is now the default, together with `SAMPLES = 256_000`,
`COST_SCALE = 1_000_000_000`, and `COST_TOP_K = True`:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py cost_counting/dyck.cnf 2..32:2
```

The runs below are historical measurements made with the source snapshots
in their result directories; new runs use `OUTPUT_DIR` and the current code.

The [retained-output PNG](benchmark_results/dyck_lazy_memory_256k_highres_retained/memory.png)
measures peak RSS with both the native decoder state and a preallocated Python
list of emitted strings alive. These are the decoder's native space-separated
token strings, without extra per-output scores or tree copies. Earlier memory
curves streamed and discarded these strings; their native decoder caches were
included. The new curves come from fresh workers, rather than adding estimated
output sizes to the earlier RSS measurements.

Raw rows additionally record `retained_count`, `output_list_shallow_bytes`,
`output_strings_bytes`, and their sum `retained_output_bytes`. For this
64-bit CPython 3.13 build, a Dyck output of n tokens has 2n−1 ASCII characters
and its string object occupies 2n+40 bytes. With m=min(256000, slice size),
the preallocated list and its strings occupy exactly `56 + m * (2*n + 48)`
object bytes. This is 19.53 MiB at n=16 and 27.34 MiB at n=32. Once m=256000,
the explicit output storage grows by 512,000 bytes (0.48828125 MiB) per
additional token. Allocator overhead and resident pages are accounted for
separately by the directly measured peak RSS; object bytes are not an RSS
breakdown. Storing an explicit list of length-n strings therefore adds
Θ(mn) storage to the cost/provenance index.

All 96 retained-output trials completed. Median total peak RSS for lazy cost
counting was 63.33, 65.38, and 69.31 MiB at lengths 16, 24, and 32; forest
rescoring used 391.89, 653.66, and 932.48 MiB. The logical output-list sizes
were identical for both methods at every length. All 48 cost-counting output
hashes match the previous exact-probability audit. `analyze_storage.py`
validates source snapshots, raw measurements, and exact object-byte formulas,
then writes `storage_summary.csv` and `validation.json`. RSS varies with
allocator size classes and the changing cost/provenance index, so its growth
need not equal the linear output-object growth at every individual step.

Run the focused prefix tests with:

```sh
cost_counting/.venv/bin/python -m unittest discover -s cost_counting -p test_cost_counting_lazy.py
```

## Memory comparison on Dyck and BIFI grammars

`benchmark_memory.py` compares `forest_py3.py` forest rescoring with the
lazy `cost_counting_orig.py` decoder. Its default grammar is `dyck.cnf`:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py
```

Defaults are exact lengths 2, 4, …, 32, top-256,000 decoding, three fresh
processes per point, cost scale 1,000,000,000, and retained output strings.
Smaller slices are exhausted. Forest rescoring uses native lazy k-best
with rational probabilities. Cost counting compiles only the requested
prefix. `COST_TOP_K = False` enables full indexing; `RETAIN_OUTPUTS = False`
streams and discards outputs.

Both methods use the same exact-length forest builder, convert its result
to their native representation, and release the temporary graph. The plot
shows median peak resident memory (MiB), with min/max shading and a logarithmic
y-axis. Peak RSS includes interpreter/import overhead, construction,
preparation, decoding, and retained outputs. Each worker has a 120-second and
4-GiB guard. Failed runs appear in a status strip; only fully completed groups
contribute to the memory curves.

For BIFI lengths 1 through 23, run:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py cost_counting/data/bifi_pcfg.txt 1..23
```

To reproduce the historical 500k **full-index** configuration described below,
set `SAMPLES = 500_000`, `COST_SCALE = 1000`, `COST_TOP_K = False`,
`RETAIN_OUTPUTS = False`, `MAX_SECONDS = 600`, `MAX_MEMORY_GIB = 8`, and
`JOBS = 4` at the top of the script. `START` is inferred automatically.

BIFI slices at lengths 1–5 contain 1, 14, 130, 3,301, and 49,873 derivations,
respectively; these slices are decoded exhaustively. Larger slices request
the first 500,000 derivations. An independent integer inside recurrence counts
each complete slice. Ambiguous grammars can have multiple derivations of the
same terminal string, so the sample count measures ranked derivations, not
distinct strings. The two methods use their native ordering: exact rational
probabilities for forest rescoring and quantized integer costs for cost
counting.

`JOBS` controls concurrent fresh workers (default 1); RSS and resource limits
apply separately to each worker. `OUTPUT_DIR` defaults to
`cost_counting/benchmark_results/memory_comparison_current/`. Each run writes:

- `memory.png`, `memory.svg`, and `memory.pdf`: length versus peak memory.
- `measurements.csv`: raw measurements from each worker process.
- `summary.csv`: measurements aggregated by method and length.
- `experiment.json`: configuration, environment, source and grammar hashes,
  and experiment records.

The saved BIFI plot requested through length 23 is
[`benchmark_results/bifi_memory_500k/lengths_1_23/memory.png`](benchmark_results/bifi_memory_500k/lengths_1_23/memory.png).
Forest rescoring completed all three repetitions at every displayed length.
Cost counting completed through length 7, exceeded 8 GiB at lengths 8–9,
and timed out after 600 seconds at length 10. The sweep was stopped on request;
cost-counting lengths 11–23 have no completed measurements. Gray dashes mark
these missing measurements. The original checkpoint retains additional forest
measurements through length 25. Scheduling began with four workers and resumed
with eight; the per-worker limits and measured code stayed unchanged. Exact
measured sources are preserved under `bifi_memory_500k/sources/`; the plot subset
records the separate rendering-source hash for subsequent plotting fixes.

For a short run, choose a small slice range:

```sh
cost_counting/.venv/bin/python cost_counting/benchmark_memory.py cost_counting/dyck.cnf 2..6:2
```

To render saved memory results without benchmarking, call
`plot_results(json.loads((output / "experiment.json").read_text()), output)`
from `benchmark_memory`, where `output` is the result directory's `Path`.
