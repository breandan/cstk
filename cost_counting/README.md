# Cost-counting decoder benchmark

Run from the project directory:

```sh
.venv/bin/python benchmark_cost_counting.py
```

The default run measures `python.cnf` at exact token lengths 10, 20, …, 100,
with 1–10 retained cost bins, for both `cost_counting_fft.py` and
`cost_counting.py`. NumPy and Matplotlib are required for this benchmark.

The two `*_decode.png` and `*_decode.svg` plots show **batch decoding time
after parsing**, in milliseconds. Each point is the median of three batches.
The x-axis is the token length of a generated grammar slice, not a prefix of
a supplied source file. Each cost bin is one distinct exact integer cost;
all tied derivations are counted.

The additional `slice_construction_times_fft` and
`slice_construction_times_ordinary` plots (PNG and SVG) show slice construction
time in milliseconds against token length, with separate lines for bins 1–10.
They use the recorded `prepare_seconds` measurements: cumulative chart work
through the requested length plus that slice's forest, counting, and
provenance/index construction. Grammar loading, decoding, and the indexes
built for shorter slices are excluded. Construction is measured once per
point; the three-run median applies only to decoding. Both construction
plots have the same axis scales and bin colors.

These plots are generated automatically alongside the decoding plots on
future benchmark runs. Generate them directly from the existing measurements:

```sh
.venv/bin/python plot_cost_counting.py --metric construction
```

A batch decodes ranks `0` through `min(999, retained_count - 1)` and renders
their token strings. When fewer than 1,000 derivations are retained, it
decodes only those available. Hollow markers identify these smaller batches.
Ranks are never repeated to fill a batch and timings are never extrapolated.
Ambiguous derivations can render the same string; this experiment counts
ranked derivations rather than distinct rendered strings.

Each backend uses the same incremental first-K cost chart to identify
relevant forest edges, then its own polynomial counting and provenance
implementation. This avoids constructing millions of irrelevant edges.
The benchmark checks all retained root counts and every decoded cost/string
against the other backend. Decoding code is the same in both implementations;
FFT multiplication, when selected by its guards, runs during preparation.

`cost_counting.csv` contains the actual decoded count, retained count, median
decode time, minimum/maximum repeated times, preparation time, FFT transform
count, and output checksum for every point. `cost_counting.json` also stores
every repetition, source hashes, the environment, and measurement definitions.
Preparation time excludes grammar loading and includes cumulative chart work
through that length plus its own index/provenance construction; indexes for
shorter slices are excluded. The charts are independent across bin limits
and backends. Decoding warm-up, garbage collection, and correctness checks
run outside the timed decoding region.

Replot the recorded data without rerunning the experiment:

```sh
.venv/bin/python plot_cost_counting.py
```

To plot the recorded preparation-plus-decoding times instead:

```sh
.venv/bin/python plot_cost_counting.py --metric total
```

For a different experiment, use `--lengths 10,30,50,70,90`, `--bins 1,2,5,10`,
`--samples 1000`, `--repeats 5`, or `--grammar path/to/grammar.cnf`.
