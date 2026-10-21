package edu.mcgill.cstk.experiments.repair


import ai.hypergraph.kaliningraph.ANSI_GREEN_BACKGROUND
import ai.hypergraph.kaliningraph.ANSI_ORANGE_BACKGROUND
import ai.hypergraph.kaliningraph.ANSI_RED_BACKGROUND
import ai.hypergraph.kaliningraph.automata.latestLangEditDistance
import ai.hypergraph.kaliningraph.automata.repairWithSparseGRE
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.approximations.WFA
import ai.hypergraph.kaliningraph.parsing.approximations.toNederhofNFA
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.visualization.alsoCopy
import ai.hypergraph.markovian.concurrency.removeEpsilonsParallel
import ai.hypergraph.markovian.concurrency.determinizeParallel
import ai.hypergraph.markovian.concurrency.trainDFAParallel
import ai.hypergraph.markovian.concurrency.writeWDFA
import ai.hypergraph.markovian.mcmc.toMarkovChain
import com.beust.klaxon.Klaxon
import com.google.common.util.concurrent.AtomicLongMap
import edu.mcgill.cstk.experiments.probing.MakeMore
import edu.mcgill.cstk.experiments.probing.charify
import edu.mcgill.cstk.experiments.probing.encodeToMakemore
import edu.mcgill.cstk.experiments.probing.uncharify
import edu.mcgill.cstk.utils.*
import org.apache.datasketches.frequencies.ErrorType
import java.io.File
import java.io.OutputStream
import java.io.PrintStream
import java.math.BigInteger
import java.math.MathContext
import java.nio.file.Files
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.function.Function
import java.util.stream.Collectors
import java.util.stream.Stream
import kotlin.collections.filter
import kotlin.math.absoluteValue
import kotlin.math.floor
import kotlin.streams.asStream
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource
import kotlin.time.measureTimedValue

@Volatile
private var dfaDecodeChecksum = 0L

/*
./gradlew collectSummaryStats
 */
fun main(args: Array<String>) {
  addBifiNgramRanks()
//  addBifiStateRefinementRanks()
//  addBifiTokenHistoryRanks()
//  prepareLaterRerankerDataset()
//  addBifiPcfgRanks()
//  addBifiHistory3Ranks()
//  compareWfaTrainingCorpora()
//  runHistogramRankExperiment(costQuantum = 1.0, maxTokens = 80)
//  runPrefixRankExperiment()
//  runLengthSliceRankExperiment()
//  runRankExperiment()
//  trainPythonStatementPCFG()
//  trainMarkovizedPythonStatementPCFG()
//  measureSingleLineStatementCoverage()
//  measurePythonDFAUnrankingDelay()
//  measurePythonDFAEnumerationDelay()
//  paginatePythonDFASlices()
//  measurePythonDFASlices()
//  evaluateChatGPTRepairPrecision()
//  LangCache.prepopPythonLangCache()
//  stackOverflowSnips().computeLengthDistributionStats()
//  stackOverflowSnips().computeRawTokenFrequencies()
//  seq2ParseSnips().computeBigramFrequencies()
//  computeErrorSizeFreq()
//  computePatchStats()
//  readPy150()
//  paperExample()
//  computeSnippetLengthDistribution()
//  computeLevDistDistribution()
//  fetchLevenshteinAlignment()
//  collectPCFGQuintuples()
//  collectNaturallySmallRepairs()
//  prepareUnsupervisedMakemoreDataset()
//  checkSemanticAdmissibility()
//  collectShortRepairSpecimens()
//  collectSyntheticRepairs()
//  collectPairwisePythonRepairs()
//    println(naturallySmallRepairs.map { it.second }.joinToString("\n").parseAndCountActiveSymbols().alsoCopy())
//  estimateLevenshteinDistanceDistribution()
//  computePatchTrigramStats()
//  corruptedTrigramStats()
//  readBIFI().toList()
//  computeEditLocationFrequency()
//  computeRelativeIntraEditDistance()
//  totalCharacterEditDistance()
//  mostCommonSubstitutions()
//  testContextEditIssue()
//  measureThroughput()
//  MakeMore.checkSamples()
//  prepareBreakerDataset()
//  MakeMore.checkPairwiseSamples()
//  MakeMore.measureRankOfTrueNextTokenWithoutConstraints()
//  MakeMore.measureRankOfTrueNextTokenWithLBHConstraints()
//  MakeMore.previewSamples()
//  prepareFixerDataset()
//  collectSingleLineExamples()
//  measureIntersection()
//  prepareRerankerDataset()
//  evaluateRerankerMRR()
//  trainPDFA(s2pg, 2).writeWDFA("wdfa.bin", s2pg, pythonStatementCNFAllProds)
//  trainPDFA()
//  correctNames()
//  errorPredDiscrepancy()
//  reconstructAndRebalance()
//  testTokenIndexing()
//  println(s2pg.terminals)
//  println(pythonStatementCNFAllProds.terminals)
//  File("python.cnf").writeText(s2pg.joinToString("\n") { it.pretty() })

//  sizeAndDistBalancedRepairsUnminimized.forEach { (broke, fixed) ->
//    parallelPythonRepair(broke).take(10).forEach { println(it) }
//    println()
//  }
}

/** Compares exact tree ranks on the same saved fixes, with all histogram work outside decoding. */
fun runHistogramRankExperiment(
  samples: Int = 1_000, seed: Int = 0, costQuantum: Double = 1.0, maxTokens: Int = 80,
  inputFile: File = File("repair_rank_comparison.csv"),
  csvFile: File = File("histogram_rank_comparison.csv"),
  texFile: File = File("histogram_rank_cdf.tex"), svgFile: File = File("histogram_rank_cdf.svg")
) {
  require(samples > 0 && maxTokens > 0)
  val lines = inputFile.readLines()
  require(lines.first().substringAfterLast(',') in setOf("fixed", "tokens"))
  val statements = lines.drop(1).filter { it.isNotBlank() }.map { line ->
    require(line.endsWith('"') && ",\"" in line)
    line.substringAfterLast(",\"").dropLast(1).replace("\"\"", "\"").tokenizeByWhitespace()
      .map { if (it == "->") "arrow" else it }
  }.filter { it.size in 1..maxTokens }.take(samples)
  require(statements.isNotEmpty())
  println("Comparing ${statements.size} saved fixes; cost quantum=$costQuantum, seed=$seed")
  val clock = TimeSource.Monotonic.markNow()
  val decoder = HistogramPCFGDecoder(pythonStatementCNFAllProds, statementPCFGWeights,
    statements.maxOf { it.size }, costQuantum) { length ->
    if (length % 5 == 0) println("Counted cost histograms through length=$length in ${clock.elapsedNow()}")
  }
  val greedy = SeededPCFGSampler(pythonStatementCNFAllProds, statementPCFGWeights, decoder.maxLength, seed)
  statements.map { it.size }.distinct().forEach { check(decoder.size(it) == greedy.size(it)) }
  val targets = statements.distinct().parallelStream().map { it to decoder.minimumTree(it) }.toList().toMap()
  val buckets = targets.values.filterNotNull().map { it.tokens().size to decoder.cost(it) }
    .distinct().sortedWith(compareBy({ it.first }, { it.second }))
  val buildTime = clock.elapsedNow()
  val compileClock = TimeSource.Monotonic.markNow()
  buckets.forEachIndexed { i, (length, cost) ->
    decoder.compile(length, cost)
    if ((i + 1) % 10 == 0 || i == buckets.lastIndex)
      println("Compiled ${i + 1}/${buckets.size} root cost buckets in ${compileClock.elapsedNow()}")
  }
  val compileTime = compileClock.elapsedNow()
  val ranks = targets.entries.parallelStream().map { (tokens, tree) ->
    val rank = tree?.let {
      decoder.rank(it).also { rank ->
        val decoded = decoder.unrankTree(tokens.size, rank)
        check(decoded == it && decoded.tokens() == tokens && decoder.rank(decoded) == rank)
        check(decoder.costAtRank(tokens.size, rank) == decoder.cost(it))
      }.plus(BigInteger.ONE)
    }
    tokens to (rank to greedy.rank(tokens)?.plus(BigInteger.ONE))
  }.toList().toMap()
  csvFile.bufferedWriter().use { writer ->
    writer.appendLine("instance,length,slice_size,cost_quantum,cost,cost_bucket_size,cost_rank,cost_log10_rank,greedy_rank,greedy_log10_rank,seed,status,tokens")
    statements.forEachIndexed { i, tokens ->
      val (rank, baseline) = ranks.getValue(tokens)
      val cost = targets[tokens]?.let(decoder::cost)
      val size = decoder.size(tokens.size)
      println("length=${tokens.size} trees=$size cost=$cost rank=$rank greedy=$baseline")
      writer.appendLine(listOf(i + 1, tokens.size, size, costQuantum, cost ?: "",
        cost?.let { decoder.histogram(tokens.size)[it] } ?: "", rank ?: "", rank?.log10Rank() ?: "",
        baseline ?: "", baseline?.log10Rank() ?: "", seed, if (rank == null) "unparseable" else "ranked",
        "\"${tokens.joinToString(" ").replace("\"", "\"\"")}\"").joinToString(","))
    }
  }
  writeHistogramRankComparison(statements.map { ranks.getValue(it).first },
    statements.map { ranks.getValue(it).second }, costQuantum, seed, texFile, svgFile)
  println("Histogram/target construction: $buildTime; cost-index compilation: $compileTime; total: ${clock.elapsedNow()}")
  println("Wrote ${csvFile.absolutePath}, ${texFile.absolutePath}, and ${svgFile.absolutePath}")
}

/** Reuses recorded fixes so every prefix experiment follows the same examples as the rank CDFs. */
fun runPrefixRankExperiment(
  inputFile: File = File("repair_rank_comparison.csv"), seed: Int = 0,
  csvFile: File = File("prefix_ranks.csv"), texFile: File = File("prefix_rank_violin.tex"),
  panelFile: File = File("prefix_rank_panel_c.tex"), summaryFile: File = File("prefix_rank_summary.csv")
) {
  // Both rank CSV writers store the quoted fixed-token sequence in the final column.
  val lines = inputFile.readLines()
  require(lines.first().substringAfterLast(',') in setOf("fixed", "tokens"))
  val statements = lines.drop(1).filter { it.isNotBlank() }.map { line ->
    require(line.endsWith('"') && ",\"" in line)
    line.substringAfterLast(",\"").dropLast(1).replace("\"\"", "\"").tokenizeByWhitespace()
      .map { if (it == "->") "arrow" else it }
  }
  require(statements.isNotEmpty() && statements.all { it.size in 1..80 })
  val sampler = SeededPCFGSampler(pythonStatementCNFAllProds, statementPCFGWeights, statements.maxOf { it.size }, seed)
  val done = AtomicInteger()
  val distinct = statements.distinct()
  println("Ranking all prefixes of ${statements.size} repairs (${distinct.size} distinct statements), seed=$seed")
  val ranks = distinct.parallelStream().map { tokens ->
    Pair(tokens, sampler.prefixRanks(tokens)).also {
      if (done.incrementAndGet() % 25 == 0) println("Ranked prefixes for ${done.get()}/${distinct.size} statements")
    }
  }.toList().toMap()
  val buckets = sortedMapOf<Int, MutableList<Double>>()
  csvFile.bufferedWriter().use { writer ->
    writer.appendLine("instance,length,prefix_length,unrevealed,slice_size,rank,log10_rank,seed,status,tokens")
    statements.forEachIndexed { i, tokens ->
      ranks.getValue(tokens).forEachIndexed { prefix, (size, zeroRank) ->
        val rank = zeroRank?.plus(BigInteger.ONE)
        val logRank = rank?.log10Rank()
        logRank?.let { buckets.getOrPut(tokens.size - prefix) { mutableListOf() }.add(it) }
        writer.appendLine(listOf(i + 1, tokens.size, prefix, tokens.size - prefix, size, rank ?: "", logRank ?: "", seed,
          if (rank == null) "unparseable" else "ranked", "\"${tokens.joinToString(" ").replace("\"", "\"\"")}\"")
          .joinToString(","))
      }
    }
  }
  summaryFile.writeText("unrevealed,samples,mean_log10_rank,population_sd_log10_rank,lower,upper,min_log10_rank,max_log10_rank\n" +
    buckets.entries.reversed().joinToString("\n") { (remaining, values) ->
      val mean = values.average()
      val sd = kotlin.math.sqrt(values.sumOf { (it - mean) * (it - mean) } / values.size)
      listOf(remaining, values.size, mean, sd, mean - sd, mean + sd, values.min(), values.max()).joinToString(",")
    } + "\n")
  texFile.writeText(prefixRankPlotTikz(buckets) + "\n")
  panelFile.writeText(prefixRankPlotTikz(buckets, groupSlot = true) + "\n")
  println("Wrote ${csvFile.absolutePath}, ${summaryFile.absolutePath}, ${texFile.absolutePath}, and ${panelFile.absolutePath}")
}

/** Exact first-hit tree ranks for single-statement fixes; n includes the final NEWLINE. */
fun runLengthSliceRankExperiment(
  samples: Int = 1_000, seed: Int = 0,
  csvFile: File = File("length_slice_ranks.csv"), texFile: File = File("length_slice_rank_cdf.tex")
) {
  require(samples > 0)
  MIN_TOKENS = 3; MAX_TOKENS = 80; MAX_RADIUS = 3
  val statements = sizeAndDistBalancedRepairsUnminimized.map { it.π2.tokenizeByWhitespace() }
    .filter { it.count { token -> token == "NEWLINE" } == 1 }.take(samples)
    .map { it.map { token -> if (token == "->") "arrow" else token } }.toList()
  check(statements.isNotEmpty()) { "No single-statement repairs in the test set" }
  if (statements.size < samples) println("Only ${statements.size}/$samples single-statement repairs available after filtering")
  println("Ranking ${statements.size} length-slice repairs with seed=$seed (length includes NEWLINE)")
  val sampler = SeededPCFGSampler(pythonStatementCNFAllProds, statementPCFGWeights, statements.maxOf { it.size }, seed)
  val ranks = statements.distinct().parallelStream().map { tokens ->
    Pair(tokens, sampler.rank(tokens)?.also { check(sampler.unrank(tokens.size, it) == tokens) }?.plus(BigInteger.ONE))
  }.toList().toMap()
  csvFile.bufferedWriter().use { writer ->
    writer.appendLine("instance,length,slice_size,rank,log10_rank,seed,status,tokens")
    statements.forEachIndexed { i, tokens ->
      val rank = ranks[tokens]
      val size = sampler.size(tokens.size)
      println("length=${tokens.size} trees=$size rank=${rank ?: "unparseable"}")
      writer.appendLine(listOf(i + 1, tokens.size, size, rank ?: "", rank?.log10Rank() ?: "", seed,
        if (rank == null) "unparseable" else "ranked",
        "\"${tokens.joinToString(" ").replace("\"", "\"\"")}\"").joinToString(","))
    }
  }
  texFile.writeText(rankCDFTikz(statements.map { ranks[it] }, "${'$'}L(G)\\cap\\Sigma^{n}${'$'}", seed) + "\n")
  println("Wrote ${statements.size} samples to ${csvFile.absolutePath} and ${texFile.absolutePath}")
}

/** First-hit ranks in the seeded PCFG derivation sampler; lengths include the final NEWLINE. */
fun runRankExperiment(
  seed: Int = 0,
  pcfgFile: File = File("python_statement_pcfg.txt"),
  csvFile: File = File("rank_experiment.csv"),
  plotFile: File = File("rank_cdf.svg")
) {
  val weights = pcfgFile.readLines().filter { it.isNotBlank() }.associate { line ->
    val (lhs, rhs) = line.substringBeforeLast(" [").split(" -> ")
    val (numerator, denominator) = line.substringAfterLast('[').removeSuffix("]").split('/').map(String::toDouble)
    require(numerator > 0 && denominator >= numerator) { "Invalid PCFG weight: $line" }
    Pair(lhs to rhs.tokenizeByWhitespace(), numerator / denominator)
  }
  val statements = sizeAndDistBalancedRepairsUnminimized.take(100).toList().flatMapIndexed { snippet, (_, fixed) ->
    fixed.tokenizeByWhitespace()
      .fold(mutableListOf(mutableListOf<String>())) { statements, token ->
        when (token) {
          "NEWLINE" -> statements.add(mutableListOf())
          "INDENT", "DEDENT" -> Unit
          else -> statements.last().add(token)
        }
        statements
      }.filter { it.isNotEmpty() }.mapIndexed { statement, tokens ->
        Triple(snippet + 1, statement + 1, tokens.map { if (it == "->") "arrow" else it } + "NEWLINE")
      }
  }
  check(statements.isNotEmpty()) { "No fixed statements in the test set" }
  println("Ranking ${statements.size} statements with seed=$seed; lengths include NEWLINE")
  val sampler = SeededPCFGSampler(pythonStatementCNFAllProds, weights, statements.maxOf { it.third.size }, seed)
  val ranks = statements.map { it.third }.distinct().associateWith { tokens ->
    sampler.rank(tokens)?.also { check(sampler.unrank(tokens.size, it) == tokens) }
      ?.plus(BigInteger.ONE)
  }
  fun BigInteger.log10() = maxOf(0, bitLength() - 53).let {
    kotlin.math.log10(shiftRight(it).toDouble()) + it * kotlin.math.log10(2.0)
  }
  csvFile.bufferedWriter().use { writer ->
    writer.appendLine("snippet,statement,length,rank,log10_rank,slice_size,seed,status,tokens")
    statements.forEach { (snippet, statement, tokens) ->
      println(tokens.joinToString(" "))
      val rank = ranks[tokens]
      val sliceSize = sampler.size(tokens.size)
      val rankPercent = rank?.toBigDecimal()?.scaleByPowerOfTen(2)
        ?.divide(sliceSize.toBigDecimal(), MathContext(4))
      println("length=${tokens.size} rank=${rank ?: "unparseable"}" +
        (rankPercent?.let { " ($it% of parse trees)" } ?: ""))
      writer.appendLine(listOf(snippet, statement, tokens.size, rank ?: "", rank?.log10() ?: "",
        sliceSize, seed, if (rank == null) "unparseable" else "ranked",
        "\"${tokens.joinToString(" ").replace("\"", "\"\"")}\"").joinToString(","))
    }
  }
  val logRanks = statements.mapNotNull { ranks[it.third]?.log10() }
  plotRankCDF(logRanks, statements.size, seed, plotFile)
  println("Ranked ${logRanks.size}/${statements.size} statements (seed=$seed); " +
    "wrote ${csvFile.absolutePath} and ${plotFile.absolutePath}")
}

/**
 * Tokenizes BIFI's original good snippets and parses their statements in parallel, using the
 * coverage measurement's extraction. Uses one CNF parse per statement and add-one smoothing.
 */
fun trainPythonStatementPCFG(
  datasetFile: File = File("src/main/resources/datasets/python/bifi/data/orig_good_code/orig.good.json"),
  outputFile: File = File("python_statement_pcfg.txt")
): File {
  val grammar = pythonStatementCNFAllProds.sortedBy { it.pretty() }.toSet().freeze()
  fun String.statements() = tokenizeByWhitespace()
    .fold(mutableListOf(mutableListOf<String>())) { statements, token ->
      when (token) {
        "NEWLINE" -> statements.add(mutableListOf())
        "INDENT", "DEDENT" -> Unit
        else -> statements.last().add(token)
      }
      statements
    }.filter { it.isNotEmpty() }

  fun Tree.productions(): Sequence<Production> =
    sequenceOf(root to if (children.isEmpty()) listOf(requireNotNull(terminal)) else children.map { it.root }) +
      children.asSequence().flatMap { it.productions() }

  val snippets = AtomicInteger(0)
  val frequencies = Files.lines(datasetFile.toPath()).use { lines ->
    lines.parallel().filter { it.trimStart().startsWith("\"code_string\":") }
      .map { requireNotNull(Klaxon().parseJsonObject("{${it.trim().removeSuffix(",")}}".reader()).string("code_string")) }
      .map { it.mapToUnquotedPythonTokens() }
      .peek { if (snippets.incrementAndGet() % 10_000 == 0) println("Tokenized $snippets BIFI snippets") }
      .flatMap { it.statements().stream() }
      .collect(Collectors.groupingByConcurrent(Function.identity(), Collectors.counting()))
  }
  // Initialize the grammar's shared lazy indices before parallel parsing.
  grammar.parseForest("NAME NEWLINE")
  val parsed = AtomicLong(0)
  val processed = AtomicInteger(0)
  val counts = frequencies.entries.parallelStream()
    .flatMap { (tokens, frequency) ->
      // Extraction leaves tokens unchanged; adapt arrow spelling and restore NEWLINE for the grammar.
      tokens.map { if (it == "->") "arrow" else it }.plus("NEWLINE").joinToString(" ")
        .let { grammar.parseForest(it).firstOrNull { it.root == START_SYMBOL } }
        .also { if (processed.incrementAndGet() % 1_000 == 0) println("Parsed $processed/${frequencies.size} distinct statements") }
        ?.also { parsed.addAndGet(frequency) }
        ?.productions()?.map { Pair(it, frequency) }?.asStream() ?: Stream.empty()
    }.collect(Collectors.groupingByConcurrent({ it.first }, Collectors.summingLong { it.second }))
  check(parsed.get() > 0L) { "No BIFI statements parsed; PCFG output was not written" }

  val smoothedCounts = grammar.associateWith { counts.getOrDefault(it, 0L) + 1L }
  val totals = smoothedCounts.entries.groupBy { it.key.LHS }
    .mapValues { (_, rules) -> rules.sumOf { it.value } }
  tailrec fun gcd(a: Long, b: Long): Long = if (b == 0L) a else gcd(b, a % b)

  return smoothedCounts.entries.joinToString("\n", postfix = "\n") { (production, numerator) ->
    val denominator = totals.getValue(production.LHS)
    gcd(numerator, denominator).let { "${production.pretty()} [${numerator / it}/${denominator / it}]" }
  }.let { outputFile.apply { writeText(it) } }
    .also { println("Wrote ${grammar.size} productions to ${it.absolutePath}; " +
      "accepted=$parsed skipped=${frequencies.values.sum() - parsed.get()} (add-one smoothing)") }
}

/** Fit full BIFI snippets with one deterministic CNF parse each and add-one production counts. */
fun trainBifiPCFG(): Map<Production, Double> {
  val corpus = File("bifi_good_code.txt")
  val output = File("bifi_pcfg.txt")
  val productions = vanillaS2PCFG.sortedBy { it.pretty() }
  val alphabet = MakeMore.PyTokMap.mt.toSortedMap()
  fun fingerprint(bytes: ByteArray) = java.security.MessageDigest.getInstance("SHA-256")
    .digest(bytes).joinToString("") { "%02x".format(it) }
  val corpusDigest = java.security.MessageDigest.getInstance("SHA-256")
  corpus.inputStream().buffered().use { input ->
    val buffer = ByteArray(1024 * 1024)
    while (true) {
      val count = input.read(buffer)
      if (count < 0) break
      corpusDigest.update(buffer, 0, count)
    }
  }
  val provenance = "# bifi-pcfg-first-cnf-v1 corpus-sha256=" +
    corpusDigest.digest().joinToString("") { "%02x".format(it) } +
    " grammar-sha256=" + fingerprint(productions.joinToString("\n") { it.pretty() }.toByteArray()) +
    " alphabet-sha256=" + fingerprint(alphabet.entries.joinToString("\n") { "${it.key.code}=${it.value}" }.toByteArray())
  if (output.isFile) {
    val lines = output.readLines()
    if (lines.firstOrNull() == provenance) {
      val weights = lines.filterNot { it.startsWith('#') || it.isBlank() }.associate { line ->
        val (lhs, rhs) = line.substringBeforeLast(" [").split(" -> ")
        val (num, den) = line.substringAfterLast('[').removeSuffix("]").split('/').map(String::toDouble)
        require(num > 0 && den >= num && den.isFinite()) { "Invalid cached PCFG weight: $line" }
        Pair(lhs to rhs.tokenizeByWhitespace(), num / den)
      }
      require(weights.keys == productions.toSet()) { "Cached BIFI PCFG has different productions" }
      require(weights.entries.groupBy { it.key.LHS }.values.all { rules ->
        kotlin.math.abs(rules.sumOf { it.value } - 1.0) < 1e-9
      }) { "Cached BIFI PCFG weights are not normalized" }
      println("Loaded ${output.absolutePath}: ${lines.getOrNull(1)?.removePrefix("# ")}")
      return weights
    }
  }

  // Keep the compact character encoding in memory; repeated snippets retain their multiplicity.
  val frequencies = HashMap<String, Long>()
  var total = 0L
  var maxTokens = 0
  corpus.useLines { lines -> lines.filter(String::isNotBlank).forEach { text ->
    require(text.startsWith('|') && text.endsWith('}')) { "Missing BIFI snippet delimiters at line ${total + 1}" }
    require(text.substring(1, text.lastIndex).all { it in alphabet }) { "Unknown BIFI token at line ${total + 1}" }
    frequencies[text] = (frequencies[text] ?: 0L) + 1L
    total++
    maxTokens = maxOf(maxTokens, text.length - 2)
  } }
  require(total > 0) { "Empty BIFI corpus" }
  val nts = productions.map { it.LHS }.distinct().sorted()
  val ntIds = nts.withIndex().associate { it.value to it.index }
  val start = ntIds.getValue(START_SYMBOL)
  val parents = IntArray(productions.size) { ntIds.getValue(productions[it].LHS) }
  val lefts = IntArray(productions.size) { -1 }
  val rights = IntArray(productions.size) { -1 }
  val byLeft = Array(nts.size) { mutableListOf<Int>() }
  val byToken = mutableMapOf<String, MutableList<Int>>()
  productions.forEachIndexed { id, (_, rhs) ->
    require(rhs.size == 1 && rhs[0] !in ntIds || rhs.size == 2 && rhs.all { it in ntIds }) {
      "Expected CNF production: ${productions[id]}"
    }
    if (rhs.size == 1) byToken.getOrPut(rhs[0]) { mutableListOf() }.add(id)
    else {
      lefts[id] = ntIds.getValue(rhs[0]); rights[id] = ntIds.getValue(rhs[1])
      byLeft[lefts[id]].add(id)
    }
  }
  val binary = byLeft.map { it.toIntArray() }.toTypedArray()
  val lexical = alphabet.mapValues { (_, token) -> byToken[token]?.toIntArray() ?: IntArray(0) }
  val stride = maxTokens + 1
  require(stride.toLong() * productions.size < Int.MAX_VALUE)
  val slots = stride.toLong() * stride * nts.size
  require(slots <= Int.MAX_VALUE) { "BIFI chart is too large: $maxTokens tokens, ${nts.size} nonterminals" }

  class Parser {
    // Intrusive active lists make joins/reset sparse; arrays are reused across snippets.
    val heads = IntArray(stride * stride)
    val choices = IntArray(slots.toInt())
    val next = IntArray(slots.toInt())
    val counts = LongArray(productions.size)
    var accepted = 0L
    var acceptedDistinct = 0
    fun add(cell: Int, rule: Int, split: Int) {
      val nt = parents[rule]
      val slot = cell * nts.size + nt
      if (choices[slot] != 0) return
      choices[slot] = split * productions.size + rule + 1
      next[slot] = heads[cell]
      heads[cell] = nt + 1
    }
    fun fit(text: String, frequency: Long) {
      val n = text.length - 2
      for (i in 0 until n) for (rule in lexical.getValue(text[i + 1])) add(i * stride + i + 1, rule, 0)
      for (length in 2..n) for (begin in 0..n - length) {
        val end = begin + length
        val cell = begin * stride + end
        for (split in begin + 1 until end) {
          val leftCell = begin * stride + split
          val rightCell = split * stride + end
          if (heads[rightCell] == 0) continue
          var active = heads[leftCell]
          while (active != 0) {
            val nt = active - 1
            for (rule in binary[nt]) {
              if (choices[rightCell * nts.size + rights[rule]] != 0) add(cell, rule, split)
            }
            active = next[leftCell * nts.size + nt]
          }
        }
      }
      fun count(begin: Int, end: Int, nt: Int) {
        val choice = choices[(begin * stride + end) * nts.size + nt] - 1
        val rule = choice % productions.size
        counts[rule] += frequency
        if (lefts[rule] >= 0) {
          val split = choice / productions.size
          count(begin, split, lefts[rule]); count(split, end, rights[rule])
        }
      }
      if (n > 0 && choices[n * nts.size + start] != 0) {
        count(0, n, start)
        accepted += frequency
        acceptedDistinct++
      }
      for (begin in 0 until n) for (end in begin + 1..n) {
        val cell = begin * stride + end
        var active = heads[cell]
        while (active != 0) {
          val slot = cell * nts.size + active - 1
          choices[slot] = 0
          active = next[slot]
        }
        heads[cell] = 0
      }
    }
  }
  val workers = minOf(Runtime.getRuntime().availableProcessors(), 8)
  val parsers = java.util.concurrent.ConcurrentLinkedQueue<Parser>()
  val local = ThreadLocal.withInitial { Parser().also { parsers.add(it) } }
  val processed = AtomicInteger(0)
  val started = TimeSource.Monotonic.markNow()
  println("Fitting BIFI PCFG: $total snippets, ${frequencies.size} distinct, max_tokens=$maxTokens, workers=$workers")
  val pool = java.util.concurrent.ForkJoinPool(workers)
  try {
    pool.submit { frequencies.entries.parallelStream().forEach { (text, frequency) ->
      local.get().fit(text, frequency)
      val done = processed.incrementAndGet()
      if (done % 10_000 == 0) println("Parsed $done/${frequencies.size} distinct BIFI snippets in ${started.elapsedNow()}")
    } }.get()
  } finally { pool.shutdown() }
  val accepted = parsers.sumOf { it.accepted }
  check(accepted > 0) { "No full BIFI snippets parsed; PCFG output was not written" }
  val counts = LongArray(productions.size) { id -> 1L + parsers.sumOf { it.counts[id] } }
  val totals = LongArray(nts.size)
  counts.indices.forEach { totals[parents[it]] += counts[it] }
  val stats = "snippets=$total distinct=${frequencies.size} accepted=$accepted skipped=${total - accepted}" +
    " accepted_distinct=${parsers.sumOf { it.acceptedDistinct }} max_tokens=$maxTokens add_one=true"
  tailrec fun gcd(a: Long, b: Long): Long = if (b == 0L) a else gcd(b, a % b)
  val temporary = File(output.path + ".tmp")
  temporary.bufferedWriter().use { writer ->
    writer.appendLine(provenance); writer.appendLine("# $stats")
    productions.forEachIndexed { id, production ->
      val num = counts[id]; val den = totals[parents[id]]; val divisor = gcd(num, den)
      writer.appendLine("${production.pretty()} [${num / divisor}/${den / divisor}]")
    }
  }
  Files.move(temporary.toPath(), output.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
    java.nio.file.StandardCopyOption.ATOMIC_MOVE)
  println("Wrote ${output.absolutePath}: $stats in ${started.elapsedNow()}")
  return productions.indices.associate { Pair(productions[it], counts[it].toDouble() / totals[parents[it]]) }
}

fun measureSingleLineStatementCoverage(tokenLimit: Int = 80, editLimits: IntRange = 2..6) {
  val filename = "datasets/python/stack_overflow/so_err_rep.txt"
  fun String.statements() = tokenizeByWhitespace()
    .fold(mutableListOf(mutableListOf<String>())) { statements, token ->
      when (token) {
        "NEWLINE" -> statements.add(mutableListOf())
        "INDENT", "DEDENT" -> Unit
        else -> statements.last().add(token)
      }
      statements
    }.filter { it.isNotEmpty() }

  fun countEditedStatements(broke: List<List<String>>, fixed: List<List<String>>): Pair<IntArray, Int> {
    val costs = Array(broke.size + 1) { IntArray(fixed.size + 1) }
    for (i in 1..broke.size) costs[i][0] = costs[i - 1][0] + broke[i - 1].size
    for (j in 1..fixed.size) costs[0][j] = costs[0][j - 1] + fixed[j - 1].size
    for (i in 1..broke.size) for (j in 1..fixed.size) {
      costs[i][j] = minOf(
        costs[i - 1][j - 1] + levenshtein(broke[i - 1], fixed[j - 1]),
        costs[i - 1][j] + broke[i - 1].size,
        costs[i][j - 1] + fixed[j - 1].size
      )
    }

    var (i, j, edited) = listOf(broke.size, fixed.size, 0)
    val underLimits = IntArray(editLimits.last + 1)
    while (i > 0 || j > 0) {
      val editDistance = if (i > 0 && j > 0) levenshtein(broke[i - 1], fixed[j - 1]) else Int.MAX_VALUE
      when {
        i > 0 && j > 0 && costs[i][j] == costs[i - 1][j - 1] + editDistance -> {
          if (editDistance > 0) {
            edited++
            editLimits.filter { editDistance < it }.forEach { underLimits[it]++ }
          }
          i--; j--
        }
        i > 0 && costs[i][j] == costs[i - 1][j] + broke[i - 1].size -> {
          edited++
          editLimits.filter { broke[i - 1].size < it }.forEach { underLimits[it]++ }
          i--
        }
        else -> j--
      }
    }
    return underLimits to edited
  }

  var totalStatements = 0
  var statementsUnderTokenLimit = 0
  var editedStatements = 0
  val statementsUnderEditLimits = IntArray(editLimits.last + 1)

  object {}.javaClass.classLoader.getResource(filename)!!.readText()
    .lines().asSequence().windowed(4, 4).forEach { (broke, fixed) ->
      val statements = broke.statements()
      totalStatements += statements.size
      statementsUnderTokenLimit += statements.count { it.size < tokenLimit }
      countEditedStatements(statements, fixed.statements()).let { (underLimits, edited) ->
        editLimits.forEach { statementsUnderEditLimits[it] += underLimits[it] }
        editedStatements += edited
      }
    }

  fun percentage(count: Int, total: Int) = String.format(Locale.US, "%.2f", 100.0 * count / total)
  println("Statements under $tokenLimit tokens: $statementsUnderTokenLimit / $totalStatements " +
    "(${percentage(statementsUnderTokenLimit, totalStatements)}%)")
  editLimits.forEach { limit ->
    println("Edited statements with <$limit same-line lexical edits: " +
      "${statementsUnderEditLimits[limit]} / $editedStatements " +
      "(${percentage(statementsUnderEditLimits[limit], editedStatements)}%)")
  }
}

/**
 * Samples global shortlex ranks in the windows 1, 2..10, 11..100, ... and separately reports
 * DFA preparation, exact BigInteger unranking, and total non-output time for each bucket.
 */
fun measurePythonDFAUnrankingDelay(
  maxSamplesPerBucket: Int = 1_000,
  samplesToPrint: Int = 10,
  randomSeed: Long = 0L,
  output: PrintStream = System.out
) {
  require(maxSamplesPerBucket > 0)
  require(samplesToPrint >= 0)
  val index = pythonStatementCNFAllProds.shortlexDFAIndex()
  val random = Random(randomSeed)
  var lowerBound = BigInteger.ZERO
  var upperBound = BigInteger.ONE

  output.println(
    "Measuring DFA shortlex BigInteger unranking " +
      "(up to $maxSamplesPerBucket uniformly sampled solutions/bucket, seed=$randomSeed); " +
      "interrupt to stop."
  )
  while (true) {
    val firstRank = lowerBound + BigInteger.ONE
    output.println("preparing bucketRanks=$firstRank..$upperBound")
    output.flush()

    val bucketStarted = TimeSource.Monotonic.markNow()
    val dfaPrepStarted = TimeSource.Monotonic.markNow()
    index.releaseBefore(lowerBound)
    index.ensureRank(upperBound - BigInteger.ONE)
    val dfaPrepTime = dfaPrepStarted.elapsedNow()
    val ranks = sampleBigIntegerRange(
      from = lowerBound,
      until = upperBound,
      limit = maxSamplesPerBucket,
      random = random
    )
    val decoded = arrayOfNulls<List<String>>(ranks.size)
    val decodeStarted = TimeSource.Monotonic.markNow()
    for (sample in ranks.indices) decoded[sample] = index.unrank(ranks[sample])
    val decodeTime = decodeStarted.elapsedNow()
    var checksum = 1L
    decoded.forEach { word -> checksum = checksum * 31 + word.hashCode() }
    dfaDecodeChecksum = checksum
    val totalTime = bucketStarted.elapsedNow()

    output.println("bucketRanks=$firstRank..$upperBound samples=${ranks.size}")
    repeat(minOf(samplesToPrint, ranks.size)) { sample ->
      val rank = ranks[sample]
      val word = decoded[sample]!!
      output.println(
        "rank=${rank + BigInteger.ONE} length=${word.size} ${word.joinToString(" ")}"
      )
    }
    output.println("dfaPrepTime=$dfaPrepTime decodeTime=$decodeTime totalTime=$totalTime")
    output.println()
    output.flush()

    lowerBound = upperBound
    upperBound *= BigInteger.TEN
  }
}

private fun sampleBigIntegerRange(from: BigInteger, until: BigInteger, limit: Int, random: Random): List<BigInteger> {
  require(from.signum() >= 0 && from < until)
  require(limit > 0)
  val population = until - from
  if (population <= BigInteger.valueOf(limit.toLong())) {
    val result = ArrayList<BigInteger>(population.intValueExact())
    var rank = from
    while (rank < until) {
      result += rank
      rank += BigInteger.ONE
    }
    Collections.shuffle(result, random)
    return result
  }

  // Rejection of duplicate draws gives a uniform ordered sample without replacement.
  val result = LinkedHashSet<BigInteger>(limit * 4 / 3 + 1)
  while (result.size < limit)
    result += from + random.nextBigInteger(population)
  return result.toMutableList().also { Collections.shuffle(it, random) }
}

private fun Random.nextBigInteger(bound: BigInteger): BigInteger {
  require(bound.signum() > 0)
  while (true) {
    val candidate = BigInteger(bound.bitLength(), this)
    if (candidate < bound) return candidate
  }
}

fun paginatePythonDFASlices(
  maxLength: Int = 20,
  pageSize: Int = 20,
  readCommand: () -> String? = { readlnOrNull() },
  output: PrintStream = System.out
) {
  require(maxLength >= 1)
  require(pageSize > 0)
  var rank = BigInteger.ONE
  var pageNumber = 1

  output.println(
    "Python statement solutions in DFA shortlex order " +
      "(token lengths 1..$maxLength; ε is shown explicitly)"
  )
  for (page in pythonStatementCNFAllProds.wordsInShortlexOrder(maxLength).chunked(pageSize)) {
    output.println("\nPage $pageNumber")
    page.forEach { tokens ->
      output.println("$rank. ${tokens.joinToString(" ")}")
      rank += BigInteger.ONE
    }

    output.print("\n[Enter] next page, q [Enter] quit > ")
    output.flush()
    val command = readCommand()?.trim()
    if (command == null || command.equals("q", ignoreCase = true) || command.equals("quit", ignoreCase = true)) {
      output.println()
      return
    }
    pageNumber++
  }
  output.println("\nNo more solutions.")
}

fun measurePythonDFASlices(): PackedDFA {
  var t = TimeSource.Monotonic.markNow()
  return pythonStatementCNFAllProds.minimalSliceDFA(20) { n, (q, delta, languageSize) ->
    println("n=$n |Q|=$q |δ|=$delta |L|=$languageSize, t=${t.elapsedNow()}")
    t = TimeSource.Monotonic.markNow()
  }.also { println("packed ${it.summarize()}") }
}

fun testTokenIndexing() = """
    x = 1 + 1
    [ i for i in j ]
    stripped_lines = lambda f : (l.rstrip("\n") for l in f)
    newlist = [word for word in words if len(word) == 9]
    Keys = [x for x in d if d[x] == 'a']
    gen_fun = lambda num: (x for u in range(num) for x in (u*2, u*10, u*u))
    [2 * x if x > 2 else add_nothing_to_list for x in some_list]
    lines = [[float(x) for x in line] for line in csv.reader(f)]
    filtered = [x for x in common if x in words]
    () + (1, 'a') + (2, 'b') + (3, 'c')
  """.lines().filter { it.isNotBlank() }.sortedBy {
    val tks = it.lexToStrTypesAsPython()
      .map { it.replace("'", "") }.filter { it in s2pg.terminals }
    pythonWDFA.scoreTokens(tks)
  }.also { println(it.joinToString("\n")) }

fun reconstructAndRebalance() {
  val markovRepairs = "datasets/python/stack_overflow/so_ds_markov.txt"
  val errRepFile = "datasets/python/stack_overflow/so_err_rep.txt"

  // Step 1: Read and map the error-repair tuples from so_err_rep.txt
  val errRepMap = object {}.javaClass.classLoader.getResource(errRepFile)!!.readText().lines().windowed(4, 4)
    .associate { lines -> Pair((lines[0].addNewLineIfMissing() to lines[1].addNewLineIfMissing()), lines.joinToString("\n")) }

  // Step 2: Process so_ds_markov.txt and split into training/validation sets
  object {}.javaClass.classLoader.getResource(markovRepairs)!!.readText()
    .split("\n\n").shuffled()
    .map { it.lines().let { it.first().uncharify() to it.drop(1).map { it.uncharify() } } }
    .filter { it.first.isNotBlank() && it.second.isNotEmpty() }
    .map { (error, repairs) ->
      val errWrds = error.tokenizeByWhitespace()
      val levDist = levenshtein(errWrds, repairs.first().tokenizeByWhitespace())
      val length = errWrds.size
      (floor(length / 10.0) to levDist) to (error to repairs)
    }
    .groupBy { it.first }.mapValues { it.value.map { it.second } }
    .let { bucketedData ->
      val validationSet = mutableListOf<Pair<String, List<String>>>()
      val trainingSet = mutableListOf<Pair<String, List<String>>>()
      for ((_, instances) in bucketedData) {
        validationSet.addAll(instances.take(50))
        trainingSet.addAll(instances.drop(50))
      }

      // Step 3: Map instances to their corresponding 4-tuples
      val validationTuples = validationSet.map { (error, repairs) ->
//        println("Error: $error")
//        println("Repair: ${repairs.first()}")
        val key = error to repairs.first()
        errRepMap[key] ?: error("No matching tuple for:\n${key.first}\n${key.second}\n in validation set")
      }
      val trainingTuples = trainingSet.map { (error, repairs) ->
        val key = error to repairs.first()
        errRepMap[key] ?: error("No matching tuple for:\n${key.first}\n${key.second}\n in training set")
      }

      // Step 4: Write all four output files
      fun formatInstances(instances: List<Pair<String, List<String>>>): String =
        instances.joinToString("\n\n") { (error, repairs) ->
          error.charify() + "\n" + repairs.joinToString("\n") { it.charify() }
        }

      // Write Markov training and validation files
      val validationString = formatInstances(validationSet)
      val trainingString = formatInstances(trainingSet)
      File("src/main/resources/datasets/python/stack_overflow/so_vs_markov.txt").apply { createNewFile() }.writeText(validationString)
      File("src/main/resources/datasets/python/stack_overflow/so_ts_markov.txt").apply { createNewFile() }.writeText(trainingString)

      // Write error-repair tuple training and validation files
      File("src/main/resources/datasets/python/stack_overflow/so_vs_err_rep.txt").apply { createNewFile() }.writeText(validationTuples.joinToString("\n"))
      File("src/main/resources/datasets/python/stack_overflow/so_ts_err_rep.txt").apply { createNewFile() }.writeText(trainingTuples.joinToString("\n"))
    }
}

fun measureIntersection() {
  sizeAndDistBalancedRepairsUnminimized.asStream().parallel().forEach { (brokeStr, fixedStr) ->
    val brokeToks = brokeStr.tokenizeByWhitespace()
    val fixedToks = fixedStr.tokenizeByWhitespace()
    val levAlign = levenshteinAlign(brokeToks, fixedToks)
    val levDist = levAlign.patchSize()
    val sts = TimeSource.Monotonic.markNow()
    val (margin, size) = measureCPU(brokeStr, levDist)
    println("${brokeToks.size}, $levDist, $margin, $size, ${sts.elapsedNow()}")
  }
}

fun filterNgramsByGrammar() {
  val allowed = pythonStatementCNF.terminals + setOf("BOS", "EOS")
  P_BIFI_PY150.counter.nrmCounts
    .getFrequentItems(ErrorType.NO_FALSE_NEGATIVES)
    .filter { it.item.all { it in allowed } }
    .map { it.item.joinToString(" ") to it.estimate }
    .sortedBy { -it.second }
    .forEach { println("${it.first} ::: ${it.second}") }
}

fun collectSingleLineExamples() {
  sizeAndDistBalancedRepairsUnminimized.filter {
    "NEWLINE" !in it.π1.tokenizeByWhitespace().dropLast(1) &&
    "NEWLINE" !in it.π2.tokenizeByWhitespace().dropLast(1) &&
    "\n" !in it.π4 &&
    it.π3.tokenizeByWhitespace().size in 10..15
//        it.π4.length in 50..80
  }.take(100)
    .forEach {
//      println(it.π1)
//      println(it.π2)
      println(it.π3)
      println(it.π4)
      println()
  }
}

fun filterProductionsByPopularity() {
  val map = readPCFG3()
  pythonStatementCNF.filter { (a, b) ->
    b.size == 1 || (map[Triple(a, b[0], b[1])] ?: 10001) > 10000
  }.toSet().normalize().noEpsilonOrNonterminalStubs
    .forEach { println(it.first + " -> " + it.second.joinToString(" ")) }
}

fun errorPredDiscrepancy() {
  var (sum, acc, total) = 0 to 0 to 0
  sizeAndDistBalancedRepairsUnminimized.forEach {
    val trueDist = levenshtein(it.π1, it.π2)
    println("TrueDist = $trueDist")
    val predDist = MakeMore.checkEditRadiusPred(it.π1)
    println("PredDist = $predDist")
    sum += (trueDist - predDist).absoluteValue
    acc += if (trueDist == predDist) 1 else 0
    println("Running average edit distance discrepancy: sum=${sum.toDouble() / ++total}, acc=${acc.toDouble() / total}")
  }
}

// ./gradlew collectSummaryStats --args=--prepare-bifi-good-code
// One natural snippet per line, using the same ASCII alphabet and boundaries as so_ts_wfa.txt.
fun prepareBifiGoodCodeDataset(
  datasetFile: File = File("src/main/resources/datasets/python/bifi/data/orig_good_code/orig.good.json"),
  outputFile: File = File("scripts/reranker_v4/wfa/bifi_good_code.txt")
): File {
  val alphabet = MakeMore.PyTokMap.tm
  require(alphabet.values.all { it.code in 33..126 })
  "pass\n".mapToUnquotedPythonTokens().addNewLineIfMissing().charify()
  val seen = AtomicInteger()
  val skipped = AtomicInteger()
  val unsupported = AtomicLongMap.create<String>()
  var written = 0
  val temporary = Files.createTempFile(outputFile.absoluteFile.parentFile.toPath(), ".bifi-good-", ".txt")
  try {
    Files.newBufferedWriter(temporary, Charsets.US_ASCII).use { writer ->
      Files.lines(datasetFile.toPath()).use { lines ->
        lines.parallel().filter { it.trimStart().startsWith("\"code_string\":") }.map { line ->
          val code = requireNotNull(Klaxon().parseJsonObject(
            "{${line.trim().removeSuffix(",")}}".reader()).string("code_string"))
          seen.incrementAndGet()
          val tokens = code.mapToUnquotedPythonTokens().tokenizeByWhitespace().map {
            when (it) { "->" -> "arrow"; "|" -> "OR"; else -> it }
          }
          val unknown = tokens.filter { it !in alphabet }.toSet()
          if (tokens.isEmpty() || unknown.isNotEmpty()) {
            unknown.forEach { unsupported.incrementAndGet(it) }
            skipped.incrementAndGet()
            null
          } else tokens.joinToString(" ").addNewLineIfMissing().charify()
        }.filter { it != null }.forEachOrdered {
          writer.appendLine(requireNotNull(it))
          if (++written % 10_000 == 0) println("Wrote $written BIFI snippets ($skipped skipped)")
        }
      }
    }
    check(written > 0) { "No encodable snippets found in $datasetFile" }
    Files.move(temporary, outputFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING)
  } finally {
    Files.deleteIfExists(temporary)
  }
  println("Wrote $written/$seen natural snippets to $outputFile; skipped $skipped empty/unsupported snippets")
  if (!unsupported.isEmpty) println("Unsupported tokens (snippet counts): ${unsupported.asMap().toSortedMap()}")
  return outputFile
}

fun prepareRerankerDataset() {
  val training = File("so_ts_wfa.txt").apply { writeText("") }
  val validation =  File("so_vs_wfa.txt").apply { writeText("") }
  val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
  val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()
  var i = 0
  contents.lines().asSequence().windowed(4, 4).map { it[0] to it[1] to it[2] to it[3] }
    .map { (broke, fixed) -> broke to fixed to levenshtein(broke, fixed) }
    .map { (broke, fixed) -> broke.addNewLineIfMissing() to fixed.addNewLineIfMissing() }
    .filter { (broke, fixed) ->
      broke.tokenizeByWhitespace().size in MIN_TOKENS until MAX_TOKENS &&
      fixed.tokenizeByWhitespace().size in MIN_TOKENS until MAX_TOKENS &&
      (levenshtein(broke, fixed) <= MAX_RADIUS)
    }
    .map { (broke, fixed) -> broke.charify() to fixed.charify() to sendCPUAndFetchTop1kRepairs(broke).map { it.charify() } }
    .forEach { (broke, fixed, top1k) ->
      "$broke\n$fixed\n${top1k.joinToString("\n")}\n\n".let {
        println(it.lines().take(20).joinToString("\n") { it.uncharify() } + "\n...\n")
        (if (i++ % 10 == 0) validation else training).appendText(it)
      }
    }
}

/** Mine neural top-1k from a history-2 WDFA stream, pacing the full dataset toward ten hours.
 * Writes so_{ts,vs}_later.txt in the working directory; rerunning resumes committed blocks.
 * First batches may borrow time (at most one minute per instance), so ten hours is a soft target.
 */
fun prepareLaterRerankerDataset() {
  val cfg = vanillaS2PCFG
  val inputs = listOf("vs", "ts").associateWith { split ->
    readWfaBlocks("so_${split}_wfa.txt").map { block ->
      require(block.size >= 2)
      block.take(2).also { pair -> pair.forEach { requireNotNull(decodeWfaLine(it)) } }
    }
  }
  val outputs = inputs.keys.associateWith { File("so_${it}_later.txt") }
  fun sha256(file: File): String {
    val digest = java.security.MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
      val buffer = ByteArray(1024 * 1024)
      while (true) { val n = input.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }
  val manifest = File("so_later.manifest")
  val provenance = "late-neural-v1 history=2 alpha=1 radius=LED+2,max5 target_ms=36000000 max_instance_ms=60000 top=1000\n" +
    listOf("so_ts_wfa.txt", "so_vs_wfa.txt", "bifi_good_code.txt", "scripts/reranker_v3/reranker_2000.js",
      "scripts/reranker_v3/reranker_2000.q8.safetensors").joinToString("\n", postfix = "\n") { "$it ${sha256(File(it))}" } +
    cfg.sortedBy { it.pretty() }.joinToString("\n", postfix = "\n") { it.pretty() }
  if (manifest.exists()) require(manifest.readText() == provenance) { "Inputs/model changed since so_later was started" }
  else {
    require(outputs.values.none { it.exists() && it.length() > 0 }) { "Existing later dataset has no provenance manifest" }
    manifest.writeText(provenance)
  }
  val completed = outputs.mapValues { (split, file) ->
    if (!file.exists()) file.writeText("")
    val text = file.readText()
    val end = text.lastIndexOf("\n\n").let { if (it < 0) 0 else it + 2 }
    val blocks = text.take(end).split("\n\n").filter(String::isNotBlank).map { it.lines() }
    require(blocks.size <= inputs.getValue(split).size)
    blocks.forEachIndexed { i, block ->
      require(block.take(2) == inputs.getValue(split)[i]) { "Output/input mismatch: $split instance ${i + 1}" }
      require(block.size <= 1002 && block.drop(2).distinct().size == block.size - 2)
      block.drop(2).forEach { requireNotNull(decodeWfaLine(it)) }
    }
    if (end != text.length) java.io.RandomAccessFile(file, "rw").use { it.setLength(end.toLong()) }
    blocks.size.also { println("Resume $split: $it/${inputs.getValue(split).size} complete -> ${file.absolutePath}") }
  }
  if (completed.all { (split, count) -> count == inputs.getValue(split).size }) return
  MAX_RADIUS = 3; LED_BUFFER = 2
  val model = run {
    val encoded = File("bifi_good_code.txt").readLines().filter(String::isNotBlank)
    val words = object : AbstractList<List<String>>() {
      override val size = encoded.size
      override fun get(index: Int) = requireNotNull(decodeWfaLine(encoded[index]))
    }
    println("Fitting history=2 WDFA on ${words.size} BIFI snippets")
    buildPDFADFA(s2pg, 2).trainDFAParallel(words).also { it.tokenStartState() }
  }
  val progress = File("so_later_progress.csv")
  if (!progress.exists()) progress.writeText("split,instance,scored,retained,gold_rank,elapsed_ms,status\n")
  val progressText = progress.readText()
  if (!progressText.endsWith('\n')) progress.writeText(progressText.take(progressText.lastIndexOf('\n') + 1))
  var spentMs = progress.readLines().drop(1).filter(String::isNotBlank).sumOf { line ->
    val fields = line.split(',')
    if (fields.size == 7 && fields[1].toInt() <= completed.getValue(fields[0])) fields[5].toLong() else 0L
  }
  var remaining = inputs.values.sumOf { it.size } - completed.values.sum()
  val profile = Files.createTempDirectory("cstk-later-chrome-").toFile()
  val chromePath = if (System.getProperty("os.name").contains("Mac"))
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" else "google-chrome"
  startRerankerServer(openBrowser = false)
  val chrome = ProcessBuilder(chromePath, "--headless=new", "--enable-unsafe-webgpu", "--no-first-run",
    "--no-default-browser-check", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
    "--user-data-dir=${profile.absolutePath}", "http://127.0.0.1:8000/")
    .redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.appendTo(File("so_later_browser.log")))
    .let { builder ->
      try { builder.start() }
      catch (error: Exception) { stopRerankerServer(); profile.deleteRecursively(); throw error }
    }
  val cleanup = Thread { chrome.destroy(); stopRerankerServer() }
  Runtime.getRuntime().addShutdownHook(cleanup)
  try {
    val warmup = inputs.getValue("vs").first().map { it.uncharify() }
    scoreWGPU(warmup[0], List(64) { warmup[1] }, 120_000)
    data class Candidate(val text: String, val score: Double, val ordinal: Long)
    val bestFirst = compareByDescending<Candidate> { it.score }.thenBy { it.ordinal }
    for (i in 0 until inputs.values.maxOf { it.size }) for ((split, pairs) in inputs) {
      if (i < completed.getValue(split) || i >= pairs.size) continue
      check(chrome.isAlive) { "Neural scorer browser exited; restart to resume" }
      val pair = pairs[i]
      val query = pair[0].uncharify()
      val started = System.nanoTime()
      val allowanceMs = ((36_000_000L - spentMs).coerceAtLeast(0) / remaining).coerceAtMost(60_000L)
      val targetDeadline = started + allowanceMs * 1_000_000L
      val hardDeadline = started + 60_000_000_000L
      val top = PriorityQueue(bestFirst.reversed())
      var scored = 0L
      var status = "time_limit"
      var lastBatchMs = 1000L
      fun checkDeadline() {
        if (System.nanoTime() >= hardDeadline) throw java.util.concurrent.TimeoutException("Instance budget exhausted")
      }
      try {
        val repair = repairWithSparseGRE(query.tokenizeByWhitespace(), cfg, ::checkDeadline)
          ?.toDFSMAntimirov(cfg.tmLst, checkDeadline = ::checkDeadline)
        val candidates = repair?.repairsByWDFA(model, cfg.tmLst) {
          if (scored == 0L) hardDeadline else targetDeadline
        }?.iterator() ?: emptyList<Pair<List<String>, Double>>().iterator()
        // Always try one complete batch; expensive constructions borrow from subsequent allowances.
        while (System.nanoTime() < if (scored == 0L) hardDeadline else targetDeadline) {
          val deadline = if (scored == 0L) hardDeadline else targetDeadline
          val batch = ArrayList<String>(1024)
          while (batch.size < 1024 && System.nanoTime() < deadline && candidates.hasNext())
            batch.add(candidates.next().first.joinToString(" "))
          if (batch.isEmpty()) { if (System.nanoTime() < deadline) status = "exhausted"; break }
          val remainingMs = (deadline - System.nanoTime()) / 1_000_000
          if (scored > 0 && remainingMs <= lastBatchMs + 50) break
          if (remainingMs <= 0) break
          val batchStarted = System.nanoTime()
          val scores = scoreWGPU(query, batch, remainingMs)
          if (System.nanoTime() >= deadline) throw java.util.concurrent.TimeoutException("Scoring budget exhausted")
          check(scores.size == batch.size && scores.all(Double::isFinite))
          lastBatchMs = (System.nanoTime() - batchStarted) / 1_000_000
          batch.zip(scores).forEach { (text, score) ->
            val candidate = Candidate(text, score, scored++)
            if (top.size < 1000) top.add(candidate)
            else if (bestFirst.compare(candidate, top.peek()) < 0) { top.poll(); top.add(candidate) }
          }
        }
      } catch (_: java.util.concurrent.TimeoutException) { /* Keep only batches completed before the deadline. */ }
      val retained = top.sortedWith(bestFirst).map { it.text.charify() }
      check(retained.size <= 1000 && retained.distinct().size == retained.size)
      val block = (pair + retained).joinToString("\n", postfix = "\n\n")
      java.io.FileOutputStream(outputs.getValue(split), true).use { it.write(block.toByteArray()); it.fd.sync() }
      val elapsedMs = (System.nanoTime() - started) / 1_000_000
      spentMs += elapsedMs; remaining--
      val goldRank = retained.indexOf(pair[1]).let { if (it < 0) "" else (it + 1).toString() }
      progress.appendText("$split,${i + 1},$scored,${retained.size},$goldRank,$elapsedMs,$status\n")
      println("LATER $split ${i + 1}/${pairs.size} | scored=$scored | kept=${retained.size} | gold_rank=$goldRank | " +
        "${elapsedMs}ms / target ${allowanceMs}ms | $status | remaining=$remaining")
    }
  } finally {
    Runtime.getRuntime().removeShutdownHook(cleanup)
    chrome.destroy(); chrome.waitFor(10, java.util.concurrent.TimeUnit.SECONDS)
    if (chrome.isAlive) chrome.destroyForcibly()
    stopRerankerServer(); profile.deleteRecursively()
  }
}

private fun readWfaBlocks(path: String, maxBlocks: Int? = null): List<List<String>> {
  val blocks = ArrayList<List<String>>()

  File(path).useLines { lines ->
    val block = ArrayList<String>()

    fun flush() {
      if (block.isNotEmpty()) {
        blocks.add(block.toList())
        block.clear()
      }
    }

    val iterator = lines.iterator()
    while (iterator.hasNext() && (maxBlocks == null || blocks.size < maxBlocks)) {
      val line = iterator.next()
      if (line.isBlank()) flush()
      else block.add(line)
    }

    if (maxBlocks == null || blocks.size < maxBlocks) flush()
  }

  return blocks
}

private fun decodeWfaLine(line: String): List<Σᐩ>? =
  try { line.uncharify().tokenizeByWhitespace() } catch (_: Exception) { null }

fun readWfaTrainingInstances(
  path: String = "so_ts_wfa.txt",
  maxBlocks: Int? = null,
  maxCandidatesPerBlock: Int? = null
): List<List<Σᐩ>> =
  readWfaBlocks(path, maxBlocks).asSequence()
    .filter { it.size >= 3 }
    .flatMap { block ->
      val candidates = block.drop(1)
      (if (maxCandidatesPerBlock == null) candidates else candidates.take(maxCandidatesPerBlock)).asSequence()
    }
    .mapNotNull(::decodeWfaLine)
    .toList()

private fun readWfaValidationPositives(
  path: String = "so_vs_wfa.txt",
  maxBlocks: Int? = null
): List<List<Σᐩ>> =
  readWfaBlocks(path, maxBlocks).mapNotNull { block -> block.getOrNull(1)?.let(::decodeWfaLine) }

private fun buildPDFADFA(cfg: CFG, history: Int): NFA {
  val h1 = cfg.toNederhofNFA(startSymbol = "START", historyDepth = history) { removeEpsilonsParallel() }
  println(h1.summary())
  val d1 = if (history > 2) h1.determinizeParallel(reportEvery = 10_000) else h1.determinize()
  println(d1.summary())
  return d1
}

private class PDFASizeLimit(message: String) : RuntimeException(message)

/** Split reachable states by token suffix and a finite output feature; retain every base continuation. */
private fun buildPDFADFAWithTokenHistory(
  base: NFA, tokenHistory: Int = 2, featureStates: Int = 1,
  nextFeature: (Int, String) -> Int = { state, _ -> state },
  parentProjection: MutableMap<Int, Int>? = null, maxBytes: Long = 50_000_000
): NFA {
  require(tokenHistory >= 0) { "Token history must be nonnegative" }
  require(featureStates > 0)
  require(base.startStates.size <= 1) { "Token-history product requires a deterministic base" }
  val clock = TimeSource.Monotonic.markNow()
  val states = base.allStates.sorted()
  val stateIds = states.withIndex().associate { (i, q) -> q to i }
  val alphabet = base.transitions.values.asSequence().flatten().map {
    requireNotNull(it.label) { "Token-history product requires an epsilon-free base" }
  }.distinct().sorted().toList()
  for ((q, edges) in base.transitions)
    require(edges.map { it.label }.toSet().size == edges.size) { "Nondeterministic base state $q" }
  val tokenIds = alphabet.withIndex().associate { (i, token) -> token to i + 1 }
  val radix = alphabet.size.toLong() + 1
  val stateCount = maxOf(1, states.size).toLong()
  val featureStride = Math.multiplyExact(stateCount, featureStates.toLong())
  var historyCapacity = 1L
  repeat(tokenHistory) {
    require(historyCapacity <= Long.MAX_VALUE / featureStride / radix) { "Token history exceeds packed Long capacity" }
    historyCapacity *= radix
  }
  val suffixModulus = if (tokenHistory == 0) 1L else historyCapacity / radix
  val ids = HashMap<Long, Int>()
  val keys = ArrayList<Long>()
  fun id(q: Int, history: Long, feature: Int): Int {
    require(feature in 0 until featureStates)
    val key = history * featureStride + feature * stateCount + stateIds.getValue(q)
    return ids.getOrPut(key) {
      check(keys.size < Int.MAX_VALUE) { "Token-history product exceeds Int state IDs" }
      keys.add(key)
      parentProjection?.put(keys.lastIndex, q)
      keys.lastIndex
    }
  }
  // Nonzero token digits distinguish empty and short prefixes without a separate length field.
  val starts = base.startStates.mapTo(linkedSetOf()) { id(it, 0L, 0) }
  val finals = HashSet<Int>()
  val transitions = HashMap<Int, List<NFA.Edge>>()
  var next = 0
  var edgeCount = 0L
  val headerBytes = 36L + alphabet.sumOf { 4L + it.toByteArray(Charsets.UTF_8).size }
  while (next < keys.size) {
    val current = next++
    val key = keys[current]
    val q = states[(key % stateCount).toInt()]
    val history = key / featureStride
    val feature = ((key / stateCount) % featureStates).toInt()
    if (q in base.finalStates) finals.add(current)
    val edges = base.transitions[q].orEmpty()
    if (edges.isNotEmpty()) transitions[current] = edges.map { edge ->
      val suffix = if (tokenHistory == 0) 0L
        else (history % suffixModulus) * radix + tokenIds.getValue(edge.label!!)
      NFA.Edge(edge.label, id(edge.target, suffix, nextFeature(feature, edge.label!!)))
    }
    edgeCount += edges.size
    val bytes = headerBytes + 12L * keys.size + 16L * edgeCount
    if (bytes > maxBytes) throw PDFASizeLimit("Reachable product exceeded $maxBytes bytes ($bytes so far)")
  }
  println("Token-history k=$tokenHistory, feature states=$featureStates product: ${states.size} -> ${keys.size} states, " +
    "$edgeCount edges, built in ${clock.elapsedNow()}")
  return NFA(starts, finals, transitions)
}

/** Research WDL8 v1: little-endian sparse rows and lossless Double log weights, not quantized WDFA. */
private fun WFA.writeLosslessWDFA(file: File): Long {
  require(startWeights.size == 1) { "Lossless WDFA requires exactly one start state" }
  val states = allStates.sorted()
  val stateIds = states.withIndex().associate { (i, q) -> q to i }
  val rows = states.map { q ->
    transitions[q].orEmpty().also { edges ->
      require(edges.all { it.label != null }) { "Lossless WDFA requires epsilon-free transitions" }
      require(edges.map { it.label }.toSet().size == edges.size) { "Nondeterministic state $q" }
    }.sortedBy { it.label }
  }
  val labels = rows.asSequence().flatten().map { it.label!! }.distinct().sorted().toList()
  val tokenIds = labels.withIndex().associate { (i, label) -> label to i }
  val dictionary = labels.map { it.toByteArray(Charsets.UTF_8) }
  val edgeCount = rows.sumOf { it.size.toLong() }
  val byteCount = 36L + dictionary.sumOf { 4L + it.size } + 12L * states.size + 16L * edgeCount
  require(byteCount <= Int.MAX_VALUE) { "Lossless WDFA exceeds ByteBuffer capacity: $byteCount bytes" }
  val out = java.nio.ByteBuffer.allocate(byteCount.toInt()).order(java.nio.ByteOrder.LITTLE_ENDIAN)
  val start = startWeights.entries.single()
  out.putInt(0x384C4457).putInt(1).putInt(states.size).putInt(edgeCount.toInt()).putInt(stateIds.getValue(start.key))
  out.putLong(start.value.toRawBits()).putInt(labels.size)
  for (token in dictionary) out.putInt(token.size).put(token)
  for (q in states) out.putLong((finalWeights[q] ?: Double.NEGATIVE_INFINITY).toRawBits())
  var offset = 0
  for (row in rows) { out.putInt(offset); offset += row.size }
  out.putInt(offset)
  for (row in rows) for (edge in row) out.putInt(tokenIds.getValue(edge.label!!))
  for (row in rows) for (edge in row) out.putInt(stateIds.getValue(edge.target))
  for (row in rows) for (edge in row) out.putLong(edge.weight.toRawBits())
  check(out.position().toLong() == byteCount && offset.toLong() == edgeCount)
  file.writeBytes(out.array())
  return file.length().also { check(it == byteCount) { "Incomplete lossless WDFA write: $it / $byteCount bytes" } }
}

fun trainPDFA(cfg: CFG = s2pg, history: Int = 2): WFA {
  val d1 = buildPDFADFA(cfg, history)
  val instances = readWfaTrainingInstances()

  println("Training on ${instances.size} total repairs and ${instances.sumOf { it.size }}")

  return d1.trainDFAParallel(instances)
}

private fun WFA.rankRepairLanguage(
  repair: ai.hypergraph.kaliningraph.automata.DFSM, gold: List<String>,
  terminalNames: List<String>, languageSize: BigInteger
): java.math.BigDecimal {
  val model = this
  val sizes = mutableMapOf<String, BigInteger>()
  fun size(q: String): BigInteger = sizes.getOrPut(q) {
    (if (q in repair.F) BigInteger.ONE else BigInteger.ZERO) +
      repair.deltaMap[q].orEmpty().values.fold(BigInteger.ZERO) { sum, next -> sum + size(next) }
  }
  val target = model.scoreTokens(gold)
  check(target.isFinite()) { "Nonfinite WFA score for ground truth" }
  val bounds = mutableMapOf<Pair<String, Int>, Pair<Double, Double>>()
  fun span(q: String, w: Int): Pair<Double, Double> = bounds.getOrPut(q to w) {
    val stop = model.tokenFinalWeight(w)
    var lo = if (q in repair.F) stop else Double.POSITIVE_INFINITY
    var hi = if (q in repair.F) stop else Double.NEGATIVE_INFINITY
    repair.deltaMap[q].orEmpty().forEach { (label, next) ->
      if (size(next).signum() != 0) {
        val edge = model.stepTokenState(w, terminalNames[label])
        val (low, high) = span(next, edge.target)
        lo = minOf(lo, edge.delta + low); hi = maxOf(hi, edge.delta + high)
      }
    }
    lo to hi
  }
  val memo = mutableMapOf<Triple<String, Int, Double>, Pair<BigInteger, BigInteger>>()
  // Count better and tied complete words; whole suffix languages can be counted at once.
  // Bounds have conservative rounding slack; ties within 1e-9 receive their average rank.
  fun count(q: String, w: Int, score: Double): Pair<BigInteger, BigInteger> {
    if (size(q).signum() == 0) return BigInteger.ZERO to BigInteger.ZERO
    val (lo, hi) = span(q, w)
    if (score + hi < target - 1e-7) return BigInteger.ZERO to BigInteger.ZERO
    if (score + lo > target + 1e-7) return size(q) to BigInteger.ZERO
    return memo.getOrPut(Triple(q, w, score)) {
      val delta = score + model.tokenFinalWeight(w) - target
      var better = if (q in repair.F && delta > 1e-9) BigInteger.ONE else BigInteger.ZERO
      var tied = if (q in repair.F && delta.absoluteValue <= 1e-9) BigInteger.ONE else BigInteger.ZERO
      repair.deltaMap[q].orEmpty().forEach { (label, next) ->
        val edge = model.stepTokenState(w, terminalNames[label])
        val (b, t) = count(next, edge.target, score + edge.delta)
        better += b; tied += t
      }
      better to tied
    }
  }
  val (better, tied) = count(repair.q_alpha, model.tokenStartState(), model.tokenStartWeight())
  check(tied.signum() > 0 && better + tied <= languageSize)
  if (languageSize <= 10_000.toBigInteger()) {
    val scores = mutableListOf<Double>()
    val word = mutableListOf<String>()
    fun enumerate(q: String) {
      if (q in repair.F) scores.add(model.scoreTokens(word) - target)
      repair.deltaMap[q].orEmpty().forEach { (label, next) ->
        word.add(terminalNames[label]); enumerate(next); word.removeLast()
      }
    }
    enumerate(repair.q_alpha)
    check(scores.size.toBigInteger() == languageSize &&
      scores.count { it > 1e-9 }.toBigInteger() == better &&
      scores.count { it.absoluteValue <= 1e-9 }.toBigInteger() == tied)
  }
  return (better * BigInteger.TWO + tied + BigInteger.ONE).toBigDecimal().divide(2.toBigDecimal())
}

/** Compare both weight fits over complete Bar-Hillel repair languages, with no candidate shortlist. */
fun compareWfaTrainingCorpora() {
  val clock = TimeSource.Monotonic.markNow()
  val alphabet = MakeMore.PyTokMap.mt
  val decode = { text: String -> text.removePrefix("|").removeSuffix("}").map { alphabet.getValue(it) } }
  val dfa = buildPDFADFA(s2pg, 2)
  val names = listOf("Synthetic repairs", "BIFI good code")
  val models = names.mapIndexed { i, name ->
    val encoded = if (i == 0) readWfaBlocks("so_ts_wfa.txt").filter { it.size >= 3 }.flatMap { it.drop(1) }
      else File("bifi_good_code.txt").useLines { it.filter(String::isNotBlank).toList() }
    println("Fitting $name on ${encoded.size} sequences (${"%.1f".format(clock.elapsedNow().inWholeMilliseconds / 1000.0)}s elapsed)")
    // Decode on demand so millions of expanded token lists need not stay in memory.
    val words = object : AbstractList<List<String>>() {
      override val size = encoded.size
      override fun get(index: Int) = decode(encoded[index])
    }
    dfa.trainDFAParallel(words).also { it.tokenStartState(); println("Fitted $name in ${clock.elapsedNow()} elapsed") }
  }
  MAX_RADIUS = 3; LED_BUFFER = 2
  val cfg = vanillaS2PCFG
  val groups = readWfaBlocks("so_vs_wfa.txt").map { it.take(2).map(decode) }
  require(groups.isNotEmpty() && groups.all { it.size == 2 })
  val csv = File("wfa_corpus_ranks.csv")
  csv.writeText("instance,length,led,lev_radius,language_size,synthetic_rank,bifi_rank,status\n")
  groups.forEachIndexed { i, (broken, gold) ->
    val started = TimeSource.Monotonic.markNow()
    // This is the same L(Python) ∩ Levenshtein-language construction as PythonBarHillelRepair.
    val repair = sendCPU(broken.joinToString(" "), cfg)
    val led = latestLangEditDistance
    val radius = (led + LED_BUFFER).coerceAtMost(MAX_RADIUS + LED_BUFFER)
    val sizes = mutableMapOf<String, BigInteger>()
    fun size(q: String): BigInteger = sizes.getOrPut(q) {
      (if (q in repair!!.F) BigInteger.ONE else BigInteger.ZERO) +
        repair.deltaMap[q].orEmpty().values.fold(BigInteger.ZERO) { sum, next -> sum + size(next) }
    }
    val languageSize = repair?.let { size(it.q_alpha) } ?: BigInteger.ZERO
    val present = repair?.recognizes(gold, cfg.tmLst) == true
    val ranks = models.map { model ->
      if (!present) return@map null
      model.rankRepairLanguage(repair!!, gold, cfg.tmLst, languageSize)
    }
    val status = if (present) "ranked" else "outside_intersection"
    csv.appendText(listOf(i + 1, broken.size, led, radius, languageSize,
      ranks[0] ?: "", ranks[1] ?: "", status).joinToString(",") + "\n")
    println("Intersection ${i + 1}/${groups.size} | words=$languageSize | ranks=$ranks | $status | ${started.elapsedNow()}")
  }
  renderWfaCorpusRanks()
  println("Wrote wfa_corpus_ranks.csv and wfa_corpus_rank_cdf.{svg,pdf,png} in ${clock.elapsedNow()}")
}

/** Add PCFG and history-2 WDFA × PCFG probability ranks over the saved full repair languages. */
fun addBifiPcfgRanks() {
  val weights = trainBifiPCFG()
  val alphabet = MakeMore.PyTokMap.mt
  val encoded = File("bifi_good_code.txt").useLines { it.filter(String::isNotBlank).toList() }
  val corpus = object : AbstractList<List<String>>() {
    override val size = encoded.size
    override fun get(index: Int) = encoded[index].removePrefix("|").removeSuffix("}").map { alphabet.getValue(it) }
  }
  println("Fitting history=2 on ${corpus.size} BIFI snippets for WDFA × PCFG")
  val model = buildPDFADFA(s2pg, 2).trainDFAParallel(corpus).also { it.tokenStartState() }
  val cfg = vanillaS2PCFG
  val csv = File("wfa_corpus_ranks.csv")
  val lines = csv.readLines()
  val columns = lines.first().split(',')
  val originalColumns = columns.filterNot { it.startsWith("pcfg_") || it.startsWith("bifi_product_") }
  val rows = lines.drop(1).filter(String::isNotBlank).map { columns.zip(it.split(',')).toMap() }
  val groups = readWfaBlocks("so_vs_wfa.txt").map { block ->
    block.take(2).map { requireNotNull(decodeWfaLine(it)) }
  }
  require(rows.size == groups.size && groups.all { it.size == 2 }) { "WFA CSV does not match the held-out set" }
  MAX_RADIUS = 3; LED_BUFFER = 2
  val output = File(csv.path + ".pcfg.tmp")
  output.writeText((originalColumns + listOf("pcfg_rank", "pcfg_parse_trees", "pcfg_status",
    "bifi_product_rank", "bifi_product_status")).joinToString(",") + "\n")
  val clock = TimeSource.Monotonic.markNow()
  groups.forEachIndexed { i, (broken, gold) ->
    val started = TimeSource.Monotonic.markNow()
    val row = rows[i]
    require(row.getValue("instance").toInt() == i + 1 && row.getValue("length").toInt() == broken.size)
    val repair = sendCPU(broken.joinToString(" "), cfg)
    check(latestLangEditDistance == row.getValue("led").toInt() &&
      (latestLangEditDistance + LED_BUFFER).coerceAtMost(MAX_RADIUS + LED_BUFFER) == row.getValue("lev_radius").toInt()) {
      "Repair radius changed in group ${i + 1}"
    }
    val sizes = mutableMapOf<String, BigInteger>()
    fun size(q: String): BigInteger = sizes.getOrPut(q) {
      (if (q in repair!!.F) BigInteger.ONE else BigInteger.ZERO) +
        repair.deltaMap[q].orEmpty().values.fold(BigInteger.ZERO) { sum, next -> sum + size(next) }
    }
    val words = repair?.let { size(it.q_alpha) } ?: BigInteger.ZERO
    check(words == row.getValue("language_size").toBigInteger()) { "Repair language changed in group ${i + 1}" }
    val present = repair?.recognizes(gold, cfg.tmLst) == true
    check(present == (row.getValue("status") == "ranked")) { "Gold membership changed in group ${i + 1}" }
    val sampler = repair?.let { IntersectionPCFGSampler(cfg, weights, it, cfg.tmLst) }
    // Equal derivation/string counts certify unambiguity, hence PCFG string probability = tree probability.
    check((sampler?.size ?: BigInteger.ZERO) == words) { "Ambiguous PCFG intersection in group ${i + 1}" }
    val rank = if (present) sampler!!.probabilityRank(gold, words) else null
    check((rank != null) == present)
    if ("pcfg_rank" in row) check((rank?.toPlainString() ?: "") == row.getValue("pcfg_rank")) {
      "PCFG baseline changed in group ${i + 1}"
    }
    val productRank = if (present) sampler!!.probabilityRank(gold, words, model) else null
    check((productRank != null) == present)
    val status = if (present) "ranked" else "outside_intersection"
    output.appendText((originalColumns.map { row.getValue(it) } +
      listOf(rank?.toPlainString() ?: "", sampler?.size ?: BigInteger.ZERO, status,
        productRank?.toPlainString() ?: "", status)).joinToString(",") + "\n")
    println("WDFA × PCFG ${i + 1}/${groups.size} | words=$words | pcfg=$rank | product=$productRank | $status | ${started.elapsedNow()}")
  }
  Files.move(output.toPath(), csv.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
    java.nio.file.StandardCopyOption.ATOMIC_MOVE)
  renderWfaCorpusRanks()
  println("Added full-language PCFG and WDFA × PCFG ranks and rendered CDF in ${clock.elapsedNow()}")
}

/** Extend the saved comparison with the history-3 Nederhof WFA fitted to the same BIFI corpus. */
fun addBifiHistory3Ranks() {
  val clock = TimeSource.Monotonic.markNow()
  val alphabet = MakeMore.PyTokMap.mt
  val encoded = File("bifi_good_code.txt").useLines { it.filter(String::isNotBlank).toList() }
  val words = object : AbstractList<List<String>>() {
    override val size = encoded.size
    override fun get(index: Int) = encoded[index].removePrefix("|").removeSuffix("}").map { alphabet.getValue(it) }
  }
  val dfa = buildPDFADFA(s2pg, 3)
  println("Fitting history=3 on ${words.size} BIFI snippets")
  val model = dfa.trainDFAParallel(words).also { it.tokenStartState() }
  println("Fitted history=3 in ${clock.elapsedNow()}")
  val csv = File("wfa_corpus_ranks.csv")
  val lines = csv.readLines()
  val columns = lines.first().split(',')
  val originalColumns = columns.filterNot { it.startsWith("bifi_h3_") }
  val rows = lines.drop(1).filter(String::isNotBlank).map { columns.zip(it.split(',')).toMap() }
  val groups = readWfaBlocks("so_vs_wfa.txt").map { block -> block.take(2).map { requireNotNull(decodeWfaLine(it)) } }
  require(rows.size == groups.size && groups.all { it.size == 2 })
  MAX_RADIUS = 3; LED_BUFFER = 2
  val cfg = vanillaS2PCFG
  val output = File(csv.path + ".h3.tmp")
  output.writeText((originalColumns + listOf("bifi_h3_rank", "bifi_h3_status")).joinToString(",") + "\n")
  groups.forEachIndexed { i, (broken, gold) ->
    val started = TimeSource.Monotonic.markNow()
    val row = rows[i]
    require(row.getValue("instance").toInt() == i + 1 && row.getValue("length").toInt() == broken.size)
    val repair = sendCPU(broken.joinToString(" "), cfg)
    check(latestLangEditDistance == row.getValue("led").toInt() &&
      (latestLangEditDistance + LED_BUFFER).coerceAtMost(MAX_RADIUS + LED_BUFFER) == row.getValue("lev_radius").toInt())
    val sizes = mutableMapOf<String, BigInteger>()
    fun size(q: String): BigInteger = sizes.getOrPut(q) {
      (if (q in repair!!.F) BigInteger.ONE else BigInteger.ZERO) +
        repair.deltaMap[q].orEmpty().values.fold(BigInteger.ZERO) { sum, next -> sum + size(next) }
    }
    val count = repair?.let { size(it.q_alpha) } ?: BigInteger.ZERO
    check(count == row.getValue("language_size").toBigInteger()) { "Repair language changed in group ${i + 1}" }
    val present = repair?.recognizes(gold, cfg.tmLst) == true
    check(present == (row.getValue("status") == "ranked"))
    val rank = if (present) model.rankRepairLanguage(repair!!, gold, cfg.tmLst, count) else null
    val status = if (present) "ranked" else "outside_intersection"
    output.appendText((originalColumns.map { row.getValue(it) } + listOf(rank?.toPlainString() ?: "", status))
      .joinToString(",") + "\n")
    println("History 3 ${i + 1}/${groups.size} | words=$count | rank=$rank | $status | ${started.elapsedNow()}")
  }
  Files.move(output.toPath(), csv.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
    java.nio.file.StandardCopyOption.ATOMIC_MOVE)
  renderWfaCorpusRanks()
  println("Added history-3 ranks and rendered CDF in ${clock.elapsedNow()}")
}

/** Compare h=2 with its exact two-token-history refinement, using the same corpus and repair languages. */
fun addBifiTokenHistoryRanks() {
  val clock = TimeSource.Monotonic.markNow()
  val alphabet = MakeMore.PyTokMap.mt
  val encoded = File("bifi_good_code.txt").useLines { it.filter(String::isNotBlank).toList() }
  val corpus = object : AbstractList<List<String>>() {
    override val size = encoded.size
    override fun get(index: Int) = encoded[index].removePrefix("|").removeSuffix("}").map { alphabet.getValue(it) }
  }
  val (base, baseBuildTime) = measureTimedValue { buildPDFADFA(s2pg, 2) }
  val (refined, refineTime) = measureTimedValue { buildPDFADFAWithTokenHistory(base, tokenHistory = 2) }
  val automata = listOf(base, refined)
  val names = listOf("BIFI h=2", "BIFI h=2 + token history=2")
  val fits = automata.mapIndexed { i, dfa ->
    println("Fitting ${names[i]} on ${corpus.size} BIFI snippets; ${dfa.summary()}")
    measureTimedValue { dfa.trainDFAParallel(corpus).also { it.tokenStartState() } }
      .also { println("Fitted ${names[i]} in ${it.duration}") }
  }
  val models = fits.map { it.value }
  val csv = File("wfa_corpus_ranks.csv")
  val lines = csv.readLines()
  val columns = lines.first().split(',')
  val originalColumns = columns.filterNot { it.startsWith("bifi_tok2_") || it.startsWith("bench_") }
  val rows = lines.drop(1).filter(String::isNotBlank).map { columns.zip(it.split(',')).toMap() }
  val groups = readWfaBlocks("so_vs_wfa.txt").map { block -> block.take(2).map { requireNotNull(decodeWfaLine(it)) } }
  require(rows.size == groups.size && groups.all { it.size == 2 })
  // Compile and warm scoring before measuring retrieval. The first intersection warms both decoders too.
  repeat(3) { groups.take(100).forEach { (_, gold) -> models.forEach { it.scoreTokens(gold) } } }
  MAX_RADIUS = 3; LED_BUFFER = 2
  val cfg = vanillaS2PCFG
  val output = File(csv.path + ".tok2.tmp")
  val extraColumns = listOf("bifi_tok2_rank", "bifi_tok2_status", "bench_intersection_ms",
    "bench_bifi_top1000_ms", "bench_tok2_top1000_ms", "bench_bifi_rank_ms", "bench_tok2_rank_ms",
    "bench_bifi_returned", "bench_tok2_returned")
  output.writeText((originalColumns + extraColumns).joinToString(",") + "\n")
  val decodeTimes = Array(2) { mutableListOf<Double>() }
  val rankTimes = Array(2) { mutableListOf<Double>() }
  val intersectionTimes = mutableListOf<Double>()
  var warmed = false
  groups.forEachIndexed { i, (broken, gold) ->
    val row = rows[i]
    require(row.getValue("instance").toInt() == i + 1 && row.getValue("length").toInt() == broken.size)
    val (repair, constructionTime) = measureTimedValue { sendCPU(broken.joinToString(" "), cfg) }
    check(latestLangEditDistance == row.getValue("led").toInt() &&
      (latestLangEditDistance + LED_BUFFER).coerceAtMost(MAX_RADIUS + LED_BUFFER) == row.getValue("lev_radius").toInt())
    val sizes = mutableMapOf<String, BigInteger>()
    fun size(q: String): BigInteger = sizes.getOrPut(q) {
      (if (q in repair!!.F) BigInteger.ONE else BigInteger.ZERO) +
        repair.deltaMap[q].orEmpty().values.fold(BigInteger.ZERO) { sum, next -> sum + size(next) }
    }
    val count = repair?.let { size(it.q_alpha) } ?: BigInteger.ZERO
    check(count == row.getValue("language_size").toBigInteger()) { "Repair language changed in group ${i + 1}" }
    val present = repair?.recognizes(gold, cfg.tmLst) == true
    check(present == (row.getValue("status") == "ranked"))
    val timings = DoubleArray(2)
    val returned = IntArray(2)
    if (repair != null) {
      fun decode(model: WFA) = repair.repairsByWDFA(model, cfg.tmLst) { Long.MAX_VALUE }.take(1000).toList()
      if (!warmed) { models.forEach { decode(it) }; warmed = true }
      // Alternate order; each of three repetitions starts with a fresh suffix cache.
      for (m in if (i % 2 == 0) listOf(0, 1) else listOf(1, 0)) {
        val samples = List(3) {
          val (top, duration) = measureTimedValue { decode(models[m]) }
          check(top.size.toBigInteger() == minOf(count, 1000.toBigInteger()))
          check(top.map { it.first }.toSet().size == top.size)
          check(top.zipWithNext().all { (a, b) -> a.second >= b.second - 1e-9 })
          if (top.isNotEmpty()) check((top.first().second - models[m].scoreTokens(top.first().first)).absoluteValue < 1e-9)
          returned[m] = top.size
          duration.inWholeNanoseconds / 1e6
        }
        timings[m] = samples.sorted()[1]
      }
    }
    val ranks = models.mapIndexed { m, model ->
      val (rank, duration) = measureTimedValue {
        if (present) model.rankRepairLanguage(repair!!, gold, cfg.tmLst, count) else null
      }
      rankTimes[m].add(duration.inWholeNanoseconds / 1e6)
      rank
    }
    check((ranks[0]?.toPlainString() ?: "") == row.getValue("bifi_rank")) { "BIFI baseline changed in group ${i + 1}" }
    val status = if (present) "ranked" else "outside_intersection"
    val constructionMs = constructionTime.inWholeNanoseconds / 1e6
    intersectionTimes.add(constructionMs)
    models.indices.forEach { decodeTimes[it].add(timings[it]) }
    output.appendText((originalColumns.map { row.getValue(it) } + listOf(ranks[1]?.toPlainString() ?: "", status,
      constructionMs, timings[0], timings[1], rankTimes[0].last(), rankTimes[1].last(), returned[0], returned[1]))
      .joinToString(",") + "\n")
    println("Token history ${i + 1}/${groups.size} | words=$count | ranks=$ranks | " +
      "top1k_ms=${timings.toList()} | intersection_ms=$constructionMs | ${clock.elapsedNow()}")
  }
  Files.move(output.toPath(), csv.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
    java.nio.file.StandardCopyOption.ATOMIC_MOVE)
  val benchmark = File("wfa_corpus_ranks_benchmark.csv")
  benchmark.writeText("model,corpus_snippets,states,transitions,build_ms,fit_ms,instances,mean_top1000_ms," +
    "median_top1000_ms,p95_top1000_ms,mean_rank_ms,mean_intersection_ms\n")
  models.indices.forEach { m ->
    val times = decodeTimes[m].sorted()
    val buildTime = if (m == 0) baseBuildTime else baseBuildTime + refineTime
    benchmark.appendText(listOf(names[m], corpus.size, automata[m].allStates.size,
      automata[m].transitions.values.sumOf { it.size }, buildTime.inWholeNanoseconds / 1e6,
      fits[m].duration.inWholeNanoseconds / 1e6, groups.size, times.average(), times[times.size / 2],
      times[kotlin.math.ceil(times.size * .95).toInt() - 1], rankTimes[m].average(), intersectionTimes.average())
      .joinToString(",") + "\n")
  }
  renderWfaCorpusRanks()
  println(benchmark.readText())
  println("Added token-history ranks and retrieval benchmarks in ${clock.elapsedNow()}")
}

private data class BifiWdfaTrial(
  val key: String, val label: String, val model: WFA, val build: Duration, val fit: Duration,
  var losslessBytes: Long = 0, var packedBytes: Long = 0,
  val ranks: MutableList<Double> = mutableListOf(), val decodeMs: MutableList<Double> = mutableListOf(),
  val rankMs: MutableList<Double> = mutableListOf(), val goldHits: MutableList<Boolean> = mutableListOf()
)

/** Scale token context with h=2 fixed: 4/5-grams remember the previous 3/4 tokens respectively.
 * Both use the same trigram prior, so the existing fourgram is an exact reproducibility baseline. */
fun addBifiNgramRanks() {
  val limit = 500_000_000L
  val encoded = File("bifi_good_code.txt").useLines { it.filter(String::isNotBlank).toList() }
  val alphabet = MakeMore.PyTokMap.mt
  val corpus = object : AbstractList<List<String>>() {
    override val size = encoded.size
    override fun get(index: Int) = encoded[index].removePrefix("|").removeSuffix("}").map { alphabet.getValue(it) }
  }
  val (trigram, baseBuild) = measureTimedValue { buildPDFADFAWithTokenHistory(buildPDFADFA(s2pg, 2), 2) }
  val parent = trigram.trainDFAParallel(corpus)
  val prior = parent.transitions.mapValues { (_, edges) -> edges.associate { it.label to kotlin.math.exp(it.weight) } }
  val trials = listOf(4 to "fourgram", 5 to "fivegram").map { (order, key) ->
    val projection = HashMap<Int, Int>()
    val (dfa, build) = measureTimedValue {
      buildPDFADFAWithTokenHistory(trigram, order - 1, parentProjection = projection, maxBytes = limit)
    }
    println("Fitting h2 + $order-gram on ${corpus.size} snippets; ${dfa.summary()}")
    val (model, fit) = measureTimedValue {
      dfa.trainDFAParallel(corpus, priorProbability = { q, token ->
        val p = projection.getValue(q)
        if (token == null) kotlin.math.exp(parent.finalWeights.getValue(p)) else prior.getValue(p).getValue(token)
      }).also { it.tokenStartState() }
    }
    println("Fitted h2 + $order-gram in $fit")
    BifiWdfaTrial(key, "h2 + $order-gram", model, baseBuild + build, fit)
  }
  benchmarkBifiWdfas(trials, corpus.size, limit, "ngram_intersection_ms")
}

/** Fixed unconditional state refinements, with a 50 MB lossless-model limit and trigram pseudocounts. */
fun addBifiStateRefinementRanks() {
  val clock = TimeSource.Monotonic.markNow()
  val limit = 50_000_000L
  val encoded = File("bifi_good_code.txt").useLines { it.filter(String::isNotBlank).toList() }
  val alphabet = MakeMore.PyTokMap.mt
  val corpus = object : AbstractList<List<String>>() {
    override val size = encoded.size
    override fun get(index: Int) = encoded[index].removePrefix("|").removeSuffix("}").map { alphabet.getValue(it) }
  }
  val (trigram, baseBuild) = measureTimedValue { buildPDFADFAWithTokenHistory(buildPDFADFA(s2pg, 2), 2) }
  val (parent, baseFit) = measureTimedValue { trigram.trainDFAParallel(corpus).also { it.tokenStartState() } }
  val prior = parent.transitions.mapValues { (_, edges) -> edges.associate { it.label to kotlin.math.exp(it.weight) } }
  val assignments = setOf("=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<=")
  fun statement(state: Int, token: String): Int = when {
    token == "NEWLINE" || token == ";" -> 0
    token == "INDENT" || token == "DEDENT" -> state
    state == 0 -> when (token) {
      "return", "yield" -> 2
      "import", "from", "global", "nonlocal" -> 3
      "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "match", "case" -> 4
      "def", "class", "async" -> 5
      "raise", "assert" -> 6
      else -> 1
    }
    state == 1 && token in assignments -> 7
    else -> state
  }
  fun keyword(state: Int, token: String): Int = when (token) {
    "NEWLINE", ";" -> 0
    "for", "in" -> 1
    "if", "elif", "else" -> 2
    "lambda" -> 3
    "and", "or", "not" -> 4
    "import", "from", "as" -> 5
    "def", "class", "return", "yield", "raise" -> 6
    else -> state
  }
  fun depth(state: Int, token: String): Int = when (token) {
    "NEWLINE" -> 0
    "(", "[", "{" -> (state + 1) % 4
    ")", "]", "}" -> (state + 3) % 4
    else -> state
  }
  val trials = mutableListOf(BifiWdfaTrial("trigram", "h2 + trigram", parent, baseBuild, baseFit))
  fun add(key: String, label: String, history: Int = 0, states: Int = 1, update: (Int, String) -> Int = { s, _ -> s }) {
    val projection = HashMap<Int, Int>()
    try {
      val (dfa, build) = measureTimedValue {
        buildPDFADFAWithTokenHistory(trigram, history, states, update, projection, limit)
      }
      println("Fitting $label on ${corpus.size} snippets; ${dfa.summary()}")
      val (model, fit) = measureTimedValue {
        dfa.trainDFAParallel(corpus, priorProbability = { q, token ->
          val p = projection.getValue(q)
          if (token == null) kotlin.math.exp(parent.finalWeights.getValue(p)) else prior.getValue(p).getValue(token)
        }).also { it.tokenStartState() }
      }
      trials.add(BifiWdfaTrial(key, label, model, baseBuild + build, fit))
      println("Fitted $label in $fit")
    } catch (e: PDFASizeLimit) { println("Skipping $label: ${e.message}") }
  }
  add("fourgram", "h2 + 4-gram", history = 3)
  add("statement", "trigram + statement mode", states = 8, update = ::statement)
  add("keyword", "trigram + recent keyword", states = 7, update = ::keyword)
  add("depth", "trigram + bracket depth mod 4", states = 4, update = ::depth)
  add("statement_depth", "trigram + coarse statement + depth", states = 20, update = { s, t ->
    val mode = when (val next = statement(s / 4, t)) { 5 -> 4; 6 -> 2; 7 -> 1; else -> next }
    mode * 4 + depth(s % 4, t)
  })
  benchmarkBifiWdfas(trials, corpus.size, limit, "ref_intersection_ms")
  println("State-refinement training and comparison finished in ${clock.elapsedNow()}")
}

/** Append model results while retaining earlier experiments; time fresh-cache exact top-1k decoding. */
private fun benchmarkBifiWdfas(
  trials: List<BifiWdfaTrial>, corpusSize: Int, limit: Long, intersectionColumn: String
) {
  val clock = TimeSource.Monotonic.markNow()
  val directory = File("models/wdfa_refinements").also { it.mkdirs() }
  trials.forEach { trial ->
    trial.losslessBytes = trial.model.writeLosslessWDFA(File(directory, "${trial.key}.wdfa64"))
    val packed = File(directory, "${trial.key}.bin")
    trial.model.writeWDFA(packed.path, s2pg)
    trial.packedBytes = packed.length()
    check(trial.losslessBytes < limit && trial.packedBytes < limit)
    println("${trial.label}: ${trial.model.summary()}, lossless=${trial.losslessBytes}, compact=${trial.packedBytes} bytes")
  }
  val csv = File("wfa_corpus_ranks.csv")
  val lines = csv.readLines()
  val columns = lines.first().split(',')
  val original = columns.filterNot { column ->
    column == intersectionColumn || trials.any { column.startsWith("ref_${it.key}_") }
  }
  val rows = lines.drop(1).filter(String::isNotBlank).map { columns.zip(it.split(',')).toMap() }
  val groups = readWfaBlocks("so_vs_wfa.txt").map { block -> block.take(2).map { requireNotNull(decodeWfaLine(it)) } }
  check(rows.size == groups.size && groups.size == 723)
  val output = File(csv.path + ".refinements.tmp")
  output.writeText((original + intersectionColumn + trials.flatMap {
    listOf("ref_${it.key}_rank", "ref_${it.key}_top1000_ms", "ref_${it.key}_rank_ms", "ref_${it.key}_gold_in_top1000")
  }).joinToString(",") + "\n")
  repeat(3) { groups.take(100).forEach { (_, gold) -> trials.forEach { it.model.scoreTokens(gold) } } }
  MAX_RADIUS = 3; LED_BUFFER = 2
  val cfg = vanillaS2PCFG
  val constructionMs = mutableListOf<Double>()
  groups.forEachIndexed { i, (broken, gold) ->
    val row = rows[i]
    check(row.getValue("instance").toInt() == i + 1 && row.getValue("length").toInt() == broken.size)
    val (repair, elapsed) = measureTimedValue { sendCPU(broken.joinToString(" "), cfg) }
    constructionMs.add(elapsed.inWholeNanoseconds / 1e6)
    check(latestLangEditDistance == row.getValue("led").toInt() &&
      (latestLangEditDistance + LED_BUFFER).coerceAtMost(MAX_RADIUS + LED_BUFFER) == row.getValue("lev_radius").toInt())
    val sizes = mutableMapOf<String, BigInteger>()
    fun size(q: String): BigInteger = sizes.getOrPut(q) {
      (if (q in repair!!.F) BigInteger.ONE else BigInteger.ZERO) +
        repair.deltaMap[q].orEmpty().values.fold(BigInteger.ZERO) { total, next -> total + size(next) }
    }
    val count = repair?.let { size(it.q_alpha) } ?: BigInteger.ZERO
    check(count == row.getValue("language_size").toBigInteger())
    val present = repair?.recognizes(gold, cfg.tmLst) == true
    check(present == (row.getValue("status") == "ranked"))
    fun decode(model: WFA) = repair!!.repairsByWDFA(model, cfg.tmLst) { Long.MAX_VALUE }.take(1000).toList()
    if (i == 0 && repair != null) trials.forEach { decode(it.model) }
    // Rotate model order. Every timed call rebuilds suffix scores; rank counting is timed separately.
    for (offset in trials.indices) {
      val trial = trials[(i + offset) % trials.size]
      var goldHit = false
      val samples = List(3) { repetition ->
        val (top, duration) = measureTimedValue { if (repair == null) emptyList() else decode(trial.model) }
        check(top.size.toBigInteger() == minOf(count, 1000.toBigInteger()))
        check(top.map { it.first }.toSet().size == top.size)
        check(top.zipWithNext().all { (a, b) -> a.second >= b.second - 1e-9 })
        if (top.isNotEmpty()) check((top.first().second - trial.model.scoreTokens(top.first().first)).absoluteValue < 1e-9)
        val hit = top.any { it.first == gold }
        if (repetition == 0) goldHit = hit else check(hit == goldHit)
        duration.inWholeNanoseconds / 1e6
      }
      trial.decodeMs.add(samples.sorted()[1])
      trial.goldHits.add(goldHit)
    }
    val rankStrings = trials.map { trial ->
      val (rank, duration) = measureTimedValue {
        if (present) trial.model.rankRepairLanguage(repair!!, gold, cfg.tmLst, count) else null
      }
      trial.rankMs.add(duration.inWholeNanoseconds / 1e6)
      if (rank != null) trial.ranks.add(rank.toDouble())
      rank?.toPlainString() ?: ""
    }
    trials.forEachIndexed { m, trial ->
      val previous = row["ref_${trial.key}_rank"] ?: if (trial.key == "trigram") row["bifi_tok2_rank"] else null
      if (previous != null) check(rankStrings[m] == previous) { "${trial.key} baseline changed at ${i + 1}" }
      row["ref_${trial.key}_gold_in_top1000"]?.let { previousHit ->
        check(trial.goldHits.last().toString() == previousHit) { "${trial.key} top-1k changed at ${i + 1}" }
      }
    }
    output.appendText((original.map { row.getValue(it) } + constructionMs.last() + trials.indices.flatMap { m ->
      listOf(rankStrings[m], trials[m].decodeMs.last(), trials[m].rankMs.last(), trials[m].goldHits.last())
    }).joinToString(",") + "\n")
    println("Refinements ${i + 1}/${groups.size} | ranks=$rankStrings | top1k_ms=${trials.map { it.decodeMs.last() }} | ${clock.elapsedNow()}")
  }
  Files.move(output.toPath(), csv.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
    java.nio.file.StandardCopyOption.ATOMIC_MOVE)
  val benchmark = File("wfa_corpus_refinements.csv")
  val header = ("key,model,corpus_snippets,states,transitions,lossless_bytes,compact_bytes,build_ms,incremental_fit_ms," +
    "instances,average_tie_hit1000,average_tie_hit1000_percent,mean_rank,median_rank,mrr,mean_top1000_ms,median_top1000_ms,p95_top1000_ms," +
    "mean_rank_ms,mean_intersection_ms,actual_hit1000,actual_hit1000_percent")
  val summaries = linkedMapOf<String, String>()
  if (benchmark.exists()) {
    val previous = benchmark.readLines()
    check(previous.first() == header)
    previous.drop(1).filter(String::isNotBlank).forEach { summaries[it.substringBefore(',')] = it }
  }
  trials.forEach { trial ->
    val times = trial.decodeMs.sorted()
    val ranks = trial.ranks.sorted()
    val hits = ranks.count { it <= 1000 }
    val medianRank = (ranks[(ranks.size - 1) / 2] + ranks[ranks.size / 2]) / 2
    summaries[trial.key] = listOf(trial.key, trial.label, corpusSize, trial.model.allStates.size,
      trial.model.transitions.values.sumOf { it.size }, trial.losslessBytes, trial.packedBytes,
      trial.build.inWholeNanoseconds / 1e6, trial.fit.inWholeNanoseconds / 1e6, groups.size,
      hits, 100.0 * hits / groups.size, ranks.average(), medianRank, ranks.sumOf { 1 / it } / groups.size,
      times.average(), times[times.size / 2], times[kotlin.math.ceil(times.size * .95).toInt() - 1],
      trial.rankMs.average(), constructionMs.average(), trial.goldHits.count { it },
      100.0 * trial.goldHits.count { it } / groups.size).joinToString(",")
  }
  benchmark.writeText(header + "\n" + summaries.values.joinToString("\n", postfix = "\n"))
  renderWfaCorpusRanks()
  println(benchmark.readText())
  println("WDFA comparison finished in ${clock.elapsedNow()}")
}

private fun renderWfaCorpusRanks() {
  val plot = """
    import csv, pathlib, numpy as np, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open('wfa_corpus_ranks.csv') as f: rows = list(csv.DictReader(f))
    series = [('synthetic_rank', 'WFA h=2: synthetic repairs', '#d95f02'),
              ('bifi_rank', 'WFA h=2: BIFI good code', '#2457a7')]
    if 'pcfg_rank' in rows[0]: series.append(('pcfg_rank', 'PCFG: BIFI good code', '#238b45'))
    if 'bifi_h3_rank' in rows[0]: series.append(('bifi_h3_rank', 'WFA h=3: BIFI good code', '#8b3fb0'))
    if 'bifi_product_rank' in rows[0]: series.append(('bifi_product_rank', 'WDFA h=2 × PCFG: BIFI good code', '#c42b35'))
    if 'bifi_tok2_rank' in rows[0]: series.append(('bifi_tok2_rank', 'WDFA h=2 + previous 2 tokens: BIFI', '#008b8b'))
    if pathlib.Path('wfa_corpus_refinements.csv').exists():
        with open('wfa_corpus_refinements.csv') as f: refinements = list(csv.DictReader(f))
        colors = ['#e377c2', '#8c564b', '#7f7f7f', '#bcbd22', '#17becf', '#111111']
        for r, color in zip([r for r in refinements if r['key'] != 'trigram'], colors):
            col = 'ref_' + r['key'] + '_rank'
            if col in rows[0]: series.append((col, r['model'], color))
    xmax = max([10.] + [float(r[c]) for r in rows for c, _, _ in series if r[c]]) * 1.05
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for col, label, color in series:
        ranks = np.sort([float(row[col]) for row in rows if row[col]])
        mean = ranks.mean() if len(ranks) else float('nan')
        print(f'{label}: retrieved={len(ranks)}/{len(rows)}, mean rank (retrieved)={mean:.3f}, '
              f'MRR={sum(1/ranks)/len(rows):.5f}, hit@10={100*sum(ranks <= 10)/len(rows):.2f}%, '
              f'hit@1000={100*sum(ranks <= 1000)/len(rows):.2f}%')
        ax.step(np.r_[1, ranks, xmax], np.r_[0, 100*np.arange(1, len(ranks)+1)/len(rows), 100*len(ranks)/len(rows)],
                where='post', color=color, label=f'{label} (mean retrieved rank {mean:,.1f})')
    ax.set(xscale='log', xlim=(1, xmax), ylim=(0, 100),
           xlabel='Ground-truth rank in intersection (log scale; average ties)',
           ylabel='Test set retrieved at or before rank (%)',
           title=f'Full repair-language ranks (N={len(rows)})')
    ax.grid(True, which='both', alpha=.2)
    ax.legend(loc='lower right', fontsize=8)
    for ext in ('svg', 'pdf', 'png'): fig.savefig(f'wfa_corpus_rank_cdf.{ext}', dpi=180)
  """.trimIndent()
  check(ProcessBuilder("python", "-c", plot).inheritIO().start().waitFor() == 0) { "Rank CDF plotting failed" }
}

data class PDFARefinementConfig(
  val name: String,
  val contextDepth: Int,
  val maxContexts: Int,
  val minContextCount: Int,
  val localWeight: Double = 0.50
)

data class PDFABenchmarkResult(
  val name: String,
  val meanPositiveLogProb: Double,
  val meanImprovement: Double,
  val finitePairs: Int,
  val states: Int,
  val transitions: Int,
  val duration: Duration
)

private fun pairedImprovement(
  baselineScores: List<Double>,
  refinedScores: List<Double>
): Pair<Double, Int> {
  val deltas =
    baselineScores.zip(refinedScores)
      .mapNotNull { (base, refined) ->
        if (base.isFinite() && refined.isFinite()) refined - base else null
      }

  return deltas.average() to deltas.size
}

fun evaluateRerankerMRR(path: String = "so_vs_wfa.txt", reportEvery: Int = 100) {
  data class Instance(val broke: String, val fixed: String, val candidates: List<String>)
  data class Scorer(val name: String, val score: (String) -> Double)

  class Stats {
    val lock = Any()
    var rrSum = 0.0
    var n = 0
    var batchNs = 0L
  }

  val instances = File(path)
    .readText().split(Regex("""\R\s*\R+""")).filter { it.isNotBlank() }
    .mapIndexedNotNull { idx, block ->
      val lines = block.lines().filter { it.isNotBlank() }

      if (lines.size < 3) {
        synchronized(System.out) { println("Skipping malformed block ${idx + 1}: only ${lines.size} lines") }
        null
      } else {
        val broke = lines[0]
        val fixed = lines[1]
        val top1k = lines.drop(2)

        // Ensure fixed is present. Preserve first occurrence.
        val candidates = (listOf(fixed) + top1k).distinct()

        Instance(broke, fixed, candidates)
      }
    }

  if (instances.isEmpty()) { println("No valid instances found in $path"); return }

  val scorers = listOf(
//    Scorer("WDFA") { it.scoreWithWDFA() },
    Scorer("WNFA") { it.scoreWithWNFA() },
    Scorer("NGMC") { it.scoreWithMC() },
    Scorer("WDFA") { it.scoreWithWDFA() }
  )

  val stats = Array(scorers.size) { Stats() }
  val every = reportEvery.coerceAtLeast(1)

  fun rankOfFixed(instance: Instance, score: (String) -> Double): Int {
    val scored = instance.candidates.map { candidate -> candidate to score(candidate) }
    val sorted = scored.sortedWith(compareBy<Pair<String, Double>> { it.second }.thenBy { it.first })
    return sorted.indexOfFirst { it.first == instance.fixed } + 1
  }

  instances.parallelStream().forEach { instance ->
    scorers.forEachIndexed { scorerIdx, scorer ->
      val t0 = System.nanoTime()
      val rank = rankOfFixed(instance, scorer.score)
      val evalNs = System.nanoTime() - t0

      val rr = 1.0 / rank
      val s = stats[scorerIdx]

      synchronized(s.lock) {
        s.rrSum += rr
        s.batchNs += evalNs
        s.n++

        if (s.n % every == 0 || s.n == instances.size) {
          val runningMRR = s.rrSum / s.n
          val batchSec = s.batchNs / 1e9
          val avgMs = s.batchNs / 1e6 / every.coerceAtMost(s.n)

          s.batchNs = 0L

          synchronized(System.out) {
            println(
              "${scorer.name} | seen ${s.n}/${instances.size} | " +
                  "MRR %.6f | last_rank $rank | last_RR %.6f | batch_eval %.2fs | avg %.2fms"
                    .format(runningMRR, rr, batchSec, avgMs)
            )
          }
        }
      }
    }
  }

  scorers.forEachIndexed { scorerIdx, scorer ->
    val s = stats[scorerIdx]

    synchronized(s.lock) {
      synchronized(System.out) {
        println("${scorer.name} | FINAL | n=${s.n} | MRR %.6f".format(if (s.n == 0) Double.NaN else s.rrSum / s.n))
      }
    }
  }
}

fun prepareFixerDataset() =
  File("names.txt").readLines().stream().parallel()
    .map { MakeMore.decode(it) }
    .flatMap { goodCode ->
      goodCode.naturalPythonCorruptions().distinct()
        .filter { levenshtein(goodCode, it) in 1..3 && it !in vanillaS2PCFG.language }
        .take(10).map { it to goodCode }.asStream()
    }.forEach { (a, b) ->
      val i = a.tokenizeByWhitespace().map { MakeMore.PyTokMap.tm[it] }.joinToString("")
      val o = b.tokenizeByWhitespace().map { MakeMore.PyTokMap.tm[it] }.joinToString("")
      val d = levenshtein(a, b)
      println("|$i $d $o}")
    }

fun measureThroughput() {
  var str = ""
  val startTime = TimeSource.Monotonic.markNow()
  var i = 0

  while (startTime.elapsedNow() < 10.seconds) {
    try {
      str += MakeMore.nextTokensReadable(str).random()
      println(str)
    } catch (_: Exception) { str = "" }
    i++
  }

  println("Done: ${i/10f} tok/sec")
}

fun prepareUnsupervisedMakemoreDataset() {
  streamBIFIContents().forEach {
    val str = it.mapToUnquotedPythonTokens() + " NEWLINE"
    if (str in vanillaS2PCFG.language) println(str.encodeToMakemore())
  }
}

fun prepareBreakerDataset() {
  val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
  val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()
  contents.lines().asSequence().windowed(4, 4).map { it[0] to it[1] to it[2] to it[3] }
    .asStream().parallel()
    .map { (broke, fixed) -> broke.addNewLineIfMissing() to fixed.addNewLineIfMissing() }
    .map { (broke, fixed) -> broke to fixed to levenshtein(broke, fixed) }
    .forEach { (broke, fixed, d) -> try { println("${(MakeMore.PyTokMap.size + 34).toChar()}" + fixed.encodeToMakemore() + " $d " + broke.encodeToMakemore() + "${(MakeMore.PyTokMap.size + 35).toChar()}") } catch (_: Exception) {} }
}

fun checkSemanticAdmissibility() {
  sizeAndDistBalancedRepairsUnminimized.toList().forEach {
    println(prettyDiffHorizontal(it.π3, it.π4))
    println("" + it.π3.isInterpretablePython() + " :: " + it.π4.isInterpretablePython())
  }
}

// ./gradlew collectSummaryStats 2>&1 | tee shortRepairSpecimens.log
fun collectShortRepairSpecimens() {
  preprocessStackOverflowQuickly()
    .filter { (a, b) ->
      a.lines().size < 10 &&
      (a.lines().maxOf { it.length } + b.lines().maxOf { it.length }) < 92 &&
      levenshtein(a.mapToUnquotedPythonTokens(), b.mapToUnquotedPythonTokens()) in 2..5 &&
      "\u001B" in prettyDiffHorizontal(a, b, "broken", "fixed")
    }.forEach { (a, b) ->
      println(a.mapToUnquotedPythonTokens())
      println(prettyDiffHorizontal(a, b, "broken", "fixed"))
      latexDiffMultilineStrings(a, b).let { (a, b) -> println("$a\n\n$b\n\n") }
    }
}

fun fetchLevenshteinAlignment() {
  preprocessStackOverflowQuickly()
    .map { (a, b) -> a.mapToUnquotedPythonTokens() to b.mapToUnquotedPythonTokens() }
    .filter { (a, b) -> b.length < 120 }
    .filter { (a, b) -> levenshtein(a, b) == 4 }
    .map { (a, b) -> levenshteinAlign(b, a).let { it.paintANSIColors() + "\n" + it.printLaTeX() } }
    .filter { ANSI_GREEN_BACKGROUND in it && ANSI_RED_BACKGROUND in it && ANSI_ORANGE_BACKGROUND in it }
    .forEach { println("$it\n") }
}

/**
 * cd src/main/resources/datasets/python &&
 * wget https://nlp.stanford.edu/projects/myasu/BIFI/data_minimal.zip &&
 * unzip data_minimal.zip && rm data_minimal.zip
 */

fun streamBIFIContents(
  good: Boolean = true,
  kind: String = if (good) "good" else "bad",
  filename: String = "src/main/resources/datasets/python/bifi/data/orig_${kind}_code/orig.${kind}.json",
  file: File = File(filename)
): Stream<String> =
  file.readLines().stream().parallel().filter { it.trimStart().startsWith("\"code_string\": \"") }
    .map { Klaxon().parseJsonObject("{$it}".reader()).string("code_string") }

fun iterBIFIContents(
  good: Boolean = true,
  kind: String = if (good) "good" else "bad",
  filename: String = "src/main/resources/datasets/python/bifi/data/orig_${kind}_code/orig.${kind}.json",
  file: File = File(filename)
): Sequence<String> =
  file.readLines().asSequence().filter { it.trimStart().startsWith("\"code_string\": \"") }
    .mapNotNull { Klaxon().parseJsonObject("{$it}".reader()).string("code_string") }

fun collectPCFGTriples() {
  val i = AtomicInteger(0)
  streamBIFIContents()
    .map { it.mapToUnquotedPythonTokens() + " NEWLINE" }
    .flatMap { if (i.incrementAndGet() % 100 == 0) println(i); vanillaS2PCFG.parseForest(it).map { it.triples() }.flatten().stream() }
    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
    .toList().sortedByDescending { it.second }
    .also {
//      File("/scratch/b/bengioy/breandan/pcfg_bifi.csv")
      File(File("").absolutePath + "/src/main/resources/models/pcfg3_BIFI.csv").writeText(it.joinToString("\n") {
        (t, count) ->
        "${t.π1} ${t.π2} ${t.π3} ::: $count" })
    }.forEach { (a, b) ->
      println("${a.π1} ${a.π2} ${a.π3} ::: $b")
    }
}

fun collectPCFGQuintuples() {
  val i = AtomicInteger(0)
  streamBIFIContents()
    .map { it.mapToUnquotedPythonTokens() + " NEWLINE" }
    .flatMap { if (i.incrementAndGet() % 1000 == 0) println(i); vanillaS2PCFG.parseForest(it).map { it.quintuples() }.flatten().stream() }
    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
    .toList().sortedByDescending { it.second }
    .also {
//      File("/scratch/b/bengioy/breandan/pcfg_bifi.csv")
      File(File("").absolutePath + "/src/main/resources/models/pcfg5_BIFI.csv").writeText(it.joinToString("\n") {
          (t, count) ->
        "${t.π1} ${t.π2} ${t.π3} ${t.π4} ${t.π5} ::: $count" })
    }.forEach { (a, b) ->
      println("${a.π1} ${a.π2} ${a.π3} ${a.π4} ${a.π5} ::: $b")
    }
}

fun paperExample() {
  val broken = "f = f.f(1:, 1:)"
  val brokeLexed = broken.mapToUnquotedPythonTokens() + " NEWLINE"
  val fixed = "f = f.f[1:, 1:]"
  val fixedLexed = fixed.mapToUnquotedPythonTokens() + " NEWLINE"
  println("Valid pair: " + (brokeLexed !in vanillaS2PCFG.language && fixedLexed in vanillaS2PCFG.language))

  MAX_TOKENS = brokeLexed.tokenizeByWhitespace().size + 5

  val intGram = vanillaS2PCFG.jvmIntersectLevFSA(makeLevFSA(brokeLexed, 2))
  println(intGram.size)

  val clock = TimeSource.Monotonic.markNow()
  intGram.toPTree()
    .sampleDirectlyWOR(stoppingCriterion = { clock.elapsedNow() < 20.seconds })
    .distinct().forEach { println(levenshteinAlign(brokeLexed, it).printLaTeX()) }
}

fun readPy150() {
  println(P_BIFI_PY150.size)
}

/** See: [symbolCounts] */
fun Σᐩ.parseAndCountActiveSymbols(): Σᐩ {
  val ntCounts =
    lines().parallelStream().map { "$it NEWLINE" }
      .flatMap { vanillaS2PCFG.parseAll(it, false).flatMap { it.activeSymbols().toList() }.stream() }
      .toList().groupingBy { it }.eachCount()

  val postProc = ".lines().filter { it.isNotEmpty() }.map { it.split(\" ::: \") }.associate { (nt, count) -> nt.trim() to count.trim().toInt() }"

  return ntCounts.entries.sortedBy { it.value }
    .joinToString("\n", "val symbolCounts by lazy { \"\"\"\n", "\n\"\"\"$postProc }") {
      "  ${it.key} ::: ${it.value}"
    }
}

// Current distribution (1000 samples):
// 1: 37.90%, 2: 18.90%, 3: 9.50%, 4+: 33.70%
fun estimateLevenshteinDistanceDistribution() {
  var sampleSize = 0
  val totalByLevDist = AtomicLongMap.create<Int>()
  val lenDist = AtomicLongMap.create<Int>()
  preprocessStackOverflow(10, 1..100)
    .map { (a, _, c) -> a to c }.distinct().take(1000)
    .forEach { (broke, minfix) ->
      val brokeLexed = broke.lexToStrTypesAsPython()
      val minfixLexed = minfix.lexToStrTypesAsPython()
      val levDist = levenshtein(brokeLexed, minfixLexed)
      totalByLevDist.incrementAndGet(levDist)
      lenDist.incrementAndGet(brokeLexed.size)
      sampleSize++

      // Normalize and print the distribution
      print("Current distribution ($sampleSize samples):\n${totalByLevDist.asMap().entries.filter { it.key <= 3 }
        .sortedBy { it.key }.joinToString(", ") { (dist, count) -> "$dist: ${"%.2f".format(100.0 * count / sampleSize)}%" }}")
      // Now print the distribution > 3
      println(", 4+: ${"%.2f".format(100.0 * totalByLevDist.asMap().entries.filter { it.key > 3 }.sumOf { it.value } / sampleSize)}%")
//      println("Current distribution:\n${lenDist.asMap().entries.sortedBy { it.key }.joinToString(", ") { (dist, count) -> "$dist: ${"%.2f".format(100.0 * count / sampleSize)}%" }}")
    }
}

fun String.addNewLineIfMissing() = if (endsWith("NEWLINE")) this else "$this NEWLINE"

fun collectSyntheticRepairs() {
  readBIFIContents()
    .map { it.mapToUnquotedPythonTokens() }
    .filter { it.tokenizeByWhitespace().size in 3..MAX_TOKENS }
    .asStream().parallel()
    .flatMap { goodCodeTks ->
      val goodCode = "$goodCodeTks NEWLINE"
      goodCode.naturalPythonCorruptions().distinct().filter {
        levenshtein(goodCode, it) <= MAX_RADIUS &&
            it !in vanillaS2PCFG.language
      }.take(10).map { it to goodCode }.asStream()
    }
    .forEach { (a, c) -> println("$a\n$c\n") }
}

// Takes ~1.5 hrs to run on M1 serially, ~17 mins w/ parallel streaming
fun collectNaturallySmallRepairs() {
  MAX_PATCH_SIZE = 6
  val encoder = Base64.getEncoder()
  val filename = "src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
    .also { File(it).also { if (it.exists()) it.delete(); it.createNewFile() } }

  var processed = 0
  preprocessStackOverflowStreaming(MAX_PATCH_SIZE, 3..100)
    .map { (a, c) ->
      if (processed++ % 100 == 0) println(processed)
      "${a.mapToUnquotedPythonTokens()}\n${c.mapToUnquotedPythonTokens()}\n" +
      "${encoder.encodeToString(a.toByteArray())}\n${encoder.encodeToString(c.toByteArray())}\n"
  }
    .distinct().toList().forEach { File(filename).appendText(it) }
}

fun collectPairwisePythonRepairs() {
  MAX_PATCH_SIZE = 3
  val lengthBounds = 20..40
  val (brokeSnips, fixedSnips) = mutableListOf<Σᐩ>() to mutableListOf<Σᐩ>()
  val errorMessages = mutableListOf<Σᐩ>()
  preprocessStackOverflow(MAX_PATCH_SIZE, lengthBounds)
    .map { (a, _, c) -> a to c }.distinct().take(1000)
    .forEach { (broke, minFix) ->
      broke.mapToUnquotedPythonTokens().also { brokeSnips.add(it) }
      minFix.mapToUnquotedPythonTokens().also { fixedSnips.add(it) }
      println("Parsing: $broke\nError message:${broke.getPythonErrorMessage().also { errorMessages.add(it) }}")
      println(levenshteinAlign(broke.mapToUnquotedPythonTokens(),
        minFix.mapToUnquotedPythonTokens()).paintANSIColors())
    }

  """
  // The following are length $lengthBounds Python statements with a human fix <=$MAX_PATCH_SIZE Levenshtein edits away
  val invalidLexedPythonStatements$MAX_PATCH_SIZE = ""${'"'}
     ${brokeSnips.joinToString("\n")} 
  ""${'"'}.trimIndent()

  val validLexedPythonStatements$MAX_PATCH_SIZE = ""${'"'}
     ${fixedSnips.joinToString("\n")} 
  ""${'"'}.trimIndent()
  
  ${""/*fixedSnips.joinToString("\n").parseAndCountActiveSymbols()*/}
  
  val errorMessages$MAX_PATCH_SIZE = ""${'"'}
     ${errorMessages.joinToString("\n")}
  ""${'"'}.trimIndent()
  """
    .trimIndent().also { it.alsoCopy() }
    .also {
      File("src/main/kotlin/edu/mcgill/cstk/experiments/repair/data/PairwisePythonRepairsL$MAX_PATCH_SIZE.kt")
        .writeText("""
          package edu.mcgill.cstk.experiments.repair.data

          $it
        """.trimIndent())
    }
}

fun testContextEditIssue() {
  ContextEdit(EditType.INS, Context("')'", "", "NEWLINE"), "':'".toPythonIntType())
    .let { it to it.toString() }
//  INS, (( ')' [')'] ',' // 53 [53] 54 ))
//  INS, (( ')' [':'] NEWLINE // 53 [55] 39 ))
//  INS, (( ')' [':'] NEWLINE // 53 [55] 39 ))
//  INS, (( ')' [':'] NEWLINE // 53 [55] 39 ))

  .also { println(it.second) }
    .also { println(it.first in contextCSV.allProbs) }
}

fun mostCommonSubstitutions() =
  File("context_edits.csv").readTrigramStats()
    .allProbs.map { it }.filter { it.key.type == EditType.SUB }
    .filter { it.key.context.mid.toPyRuleName().contains('\'') }
    .filter { it.key.newMid.toPyRuleName().contains('\'') }
    .map { Triple(it.value, it.key.context.mid, it.key.newMid) }
    .groupBy { it.second to it.third }.mapValues { it.value.sumOf { it.first } }
    .entries.sortedByDescending { it.value }
    .joinToString("\n") { (pair, freq) ->
      "$freq, ${pair.first.toPyRuleName()}, ${pair.second.toPyRuleName()}" }
    .also { println(("freq, before, after\n$it").reformatCSVIntoPrettyColumns()) }

/*
Percentage of snippets with lexical token length <= n

| n  | StackOverflow | Seq2Parse |
|:--:|:-------------:|:---------:|
| 10 |      9%       |    29%    |
| 20 |      16%      |    47%    |
| 30 |      25%      |    62%    |
| 40 |      34%      |    73%    |
| 50 |      41%      |    79%    |
 */

// Returns the frequency of tokenized snippet lengths for buckets of size 10
fun Sequence<List<String>>.computeLengthDistributionStats(
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
) =
  map { it.size }.take(10000).groupBy { it / 10 }.mapValues { it.value.size }
    .toList().sortedBy { it.first }.map { it.first * 10 to it.second }
    .runningFold(0 to 0) { (_, prevCount), (n, count) -> n to (prevCount + count) }
    .let { it.map { (n, count) -> n to count.toDouble() / it.last().second } }
    .joinToString("\n") { "<=${it.first}, ${it.second}" }.also { println(it) }

fun Sequence<List<String>>.computeRawTokenFrequencies() =
  take(10000).flatten().groupingBy { it }.eachCount()
    // Now normalize by total number of tokens
    .let { val total = it.values.sum(); it.mapValues { (_, count) -> count.toDouble() / total } }
    // Use precision of 4 decimal places
    .mapValues { (_, count) -> "%.4f".format(count) }
    .toList().sortedByDescending { it.second }
    .joinToString("\n") { "${it.first}, ${it.second}" }.also { println("Token, Frequency\n$it") }

/*
Top Bigrams:

Bigram, Frequency
"_NAME_ (", 0.0551
") _NEWLINE_", 0.0424
"( _NAME_", 0.0330
"_NEWLINE_ _DEDENT_", 0.0321
"_NEWLINE_ _INDENT_", 0.0258
"_NEWLINE_ _NAME_", 0.0255
": _NEWLINE_", 0.0238
"_NAME_ )", 0.0237
"_NAME_ =", 0.0232
"_NEWLINE_ _NEWLINE_", 0.0224
"Colon Newline", 0.0197
"_INDENT_ _NAME_", 0.0190
"Newline Indent", 0.0167
"( _STRING_", 0.0149
"= _NAME_", 0.0146
"_NEWLINE_ _ENDMARKER_", 0.0146
"_STRING_ )", 0.0136
"_DEDENT_ _DEDENT_", 0.0128
"_NUMBER_ ,", 0.0126
", _NUMBER_", 0.0116
"_NAME_ _NEWLINE_", 0.0113
"_DEDENT_ _NAME_", 0.0112
", _NAME_", 0.0102
"_NUMBER_ _NEWLINE_", 0.0101
"_NUMBER_ )", 0.0097
"_ENDMARKER_ Stmts_Or_Newlines", 0.0095
". _NAME_", 0.0092
"Indent Stmts_Or_Newlines", 0.0090
"_DEDENT_ _ENDMARKER_", 0.0085
"_NAME_ .", 0.0084
"Def_Keyword Simple_Name", 0.0081
 */

fun Sequence<List<String>>.computeBigramFrequencies() =
    take(10000).flatten().zipWithNext().groupingBy { it }.eachCount()
    // Now normalize by total number of tokens
    .let { val total = it.values.sum(); it.mapValues { (_, count) -> count.toDouble() / total } }
    // Use precision of 4 decimal places
    .mapValues { (_, count) -> "%.4f".format(count) }
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "\"${it.first.first} ${it.first.second}\", ${it.second}" }.also { println("Bigram, Frequency\n$it") }

fun stackOverflowSnips() =
  readContents("parse_errors.json").map { it.lexToStrTypesAsPython() }

fun seq2ParseSnips() =
  brokenPythonSnippets.map { it.substringBefore("<||>").tokenizeByWhitespace() }

/*
Number of edits, Frequency
1, 0.512
2, 0.834
3, 1.0
*/
fun computeErrorSizeFreq() =
  preprocessStackOverflow().map { (broke, humfix, minfix) ->
    val brokeLex = broke.lexToStrTypesAsPython()
    val minfixLex = minfix.lexToStrTypesAsPython()
    val minpatch = extractPatch(brokeLex, minfixLex)
//    println(prettyDiffs(listOf(brokeLex.joinToString(" "), minfixLex.joinToString(" ")), listOf("broken", "minimized fix")))
    minpatch.changedIndices().size
  }.take(1000).groupBy { it }.mapValues { it.value.size }
    .toList().sortedBy { it.first }.map { it.first to it.second }
    .runningFold(0 to 0) { (_, prevCount), (n, count) -> n to (prevCount + count) }
    .let { it.map { (n, count) -> n to count.toDouble() / it.last().second } }
    .joinToString("\n") { "${it.first}, ${it.second}" }
      .also { println("Number of edits, Frequency\n$it") }

// Approximate location of edits normalized by snippet length
//10%, 11.6539%
//20%, 5.7252%
//30%, 6.2087%
//40%, 5.9542%
//50%, 5.5980%
//60%, 7.9389%
//70%, 7.0738%
//80%, 6.9466%
//90%, 12.4173%
//100%, 30.4835%
fun computeEditLocationFrequency() =
  preprocessStackOverflow().runningFold(List(10) { 0 }.toMutableList()) { hist, (b, h, m) ->
    val brokeLex = b.lexToStrTypesAsPython()
    val minfixLex = m.lexToStrTypesAsPython()
    val minpatch = extractPatch(brokeLex, minfixLex)
//    println(prettyDiffs(listOf(brokeLex.joinToString(" "), minfixLex.joinToString(" ")), listOf("broken", "minimized fix")))
    minpatch.changedIndices()
      .map { ((100.0 * it / minfixLex.size) / 10).toInt().coerceAtMost(9) }
      .forEach { hist[it]++ }

    hist
  }.forEachIndexed { i, rawCounts ->
    if (i % 10 == 0) {
      val sum = rawCounts.sum()
      rawCounts.forEachIndexed { i, it ->
        println("${(i + 1) * 10}%, ${"%.4f".format(100.0 * it / sum)}%")
      }; println()
    }
  }

// Returns a distribution of snippet lengths in buckets of 10, up to maximum 100+
// 10, 5.00%
// 20, 10.00%
// 30, 15.00%
//    ...
// 100+, 100.00%
fun computeSnippetLengthDistribution() {
  val buckets: MutableMap<IntRange, Int> =
    0.rangeTo(10).associate { (it * 10)..(it * 10 + 9) to 0 }.toMutableMap()
  buckets[100..Int.MAX_VALUE] = 0

  var total = 0
  preprocessStackOverflowQuickly().limit(10_000).toList().map { it.first }
    .map { it.lexToStrTypesAsPython().size }
    .forEach { len ->
      val bucket = buckets.keys.first { len in it }
      buckets[bucket] = buckets.getValue(bucket) + 1
      total++
    }

//  buckets.toList().sortedBy { it.first.first }.joinToString("\n") { (range, count) ->
//    "${range.first}, ${"%.2f".format(100.0 * count / total.toDouble())}%"
//  }.also { println(it) }
  // Cumulative distribution

  buckets.toList().sortedBy { it.first.first }.runningFold(0 to 0) { (prev, prevCount), (range, count) ->
    range.first to (prevCount + count)
  }.map { (n, count) -> n to count.toDouble() / total.toDouble() }
    .joinToString("\n") { "${it.first}, ${"%.2f".format(100.0 * it.second)}%" }
    .also { println(it) }
}

fun computeLevDistDistribution( ){
  val buckets = 0.rangeTo(10).associate { it..it+1 to 0 }.toMutableMap()
  buckets[10..Int.MAX_VALUE] = 0

  var total = 0
  preprocessStackOverflowQuickly().limit(10_000).toList()
    .map { it.first.lexToStrTypesAsPython() to it.second.lexToStrTypesAsPython() }
    .map { (broke, minfix) ->
      if (broke.size - minfix.size > 10) Int.MAX_VALUE
      else levenshtein(broke, minfix)
    }
    .forEach { len ->
      val bucket = buckets.keys.first { len in it }
      buckets[bucket] = buckets.getValue(bucket) + 1
      total++
    }

  buckets.toList().sortedBy { it.first.first }.runningFold(0 to 0) { (prev, prevCount), (range, count) ->
    range.first to (prevCount + count)
  }.map { (n, count) -> n to count.toDouble() / total.toDouble() }
    .joinToString("\n") { "${it.first}, ${"%.2f".format(100.0 * it.second)}%" }
    .also { println(it) }
}

fun preprocessStackOverflowQuickly(
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
  fixedSnippets: Sequence<String> = readContents("parse_fixes.json"),
): Stream<Π2A<Σᐩ>> =
  brokeSnippets.zip(fixedSnippets).asStream().parallel()

//1, 40.66%
//2, 15.00%
//3, 5.80%
//4, 4.86%
//5, 4.26%
//6, 2.98%
//7, 2.05%
//8, 2.73%
//9, 1.62%
//10, 2.30%
//11, 1.88%
//12, 2.81%
//13, 1.11%
//14, 0.60%
//15, 1.28%
//16, 1.45%
//17, 1.02%
//18, 0.68%
//19, 0.51%
// Relative distance between edits in multi-edit patches
fun computeRelativeIntraEditDistance() =
  preprocessStackOverflow().map { (b, h, m) ->
    val brokeLex = b.lexToStrTypesAsPython()
    val minfixLex = m.lexToStrTypesAsPython()
    extractPatch(brokeLex, minfixLex)
  }.filter { 1 < it.size }.runningFold(mutableMapOf<Int, Int>()) { hist, patch ->
    patch.changedIndices().zipWithNext().forEach { (i, j) ->
      val d = (j - i).absoluteValue
      hist[d] = hist.getOrDefault(d, 0) + 1
    }
    hist
  }.forEachIndexed { i, rawCounts ->
    if (i % 10 == 0) {
      val sum = rawCounts.values.sum()
      rawCounts.toList().sortedBy { it.first }.forEach { (dist, count) ->
        println("${dist}, ${"%.2f".format(100.0 * count / sum)}%")
      }; println()
    }
  }

//1, 0.72%
//2, 53.48%
//3, 29.48%
//4, 11.35%
//5, 3.32%
//6, 0.83%
//7, 0.29%
//9, 0.20%
//10, 0.04%
fun totalCharacterEditDistance() =
  preprocessStackOverflow().map { (b, h, m) ->
    val brokeLex = b.tokenizeAsPython()
    val minfixLex = m.tokenizeAsPython()
    extractPatch(brokeLex, minfixLex).totalCharacterwiseEditDistance()
  }.runningFold(mutableMapOf<Int, Int>()) { h, p ->
    h[p] = h.getOrDefault(p, 0) + 1; h
  }.forEachIndexed { i, rawCounts ->
    if (i % 10 == 0) {
      val sum = rawCounts.values.sum()
      rawCounts.toList().sortedBy { it.first }.forEach { (dist, count) ->
        println("${dist + 1}, ${"%.2f".format(100.0 * count / sum)}%")
      }; println()
    }
  }

/*
Insertion, Frequency, Deletion     , Frequency, Substitution            , Frequency
')'      , 98       , NAME         , 219      , (UNKNOWN_CHAR -> STRING), 53
','      , 87       , UNKNOWN_CHAR , 160      , (NAME -> ',')           , 22
98       , 59       , '.'          , 50       , (':' -> ',')            , 12
99       , 58       , NUMBER       , 41       , (NEWLINE -> ':')        , 12
'}'      , 46       , '>'          , 30       , ('=' -> ':')            , 11
']'      , 36       , NEWLINE      , 28       , (STRING -> ',')         , 8
':'      , 35       , ')'          , 26       , ('.' -> ',')            , 8
'('      , 33       , '**'         , 25       , (NAME -> STRING)        , 8
NEWLINE  , 28       , ':'          , 23       , ('pass' -> NAME)        , 8
NAME     , 19       , '>>'         , 20       , ('=' -> '==')           , 7
STRING   , 17       , STRING       , 19       , ('class' -> NAME)       , 6
'{'      , 13       , 98           , 13       , ('[' -> '{')            , 6
'='      , 12       , '...'        , 12       , ('break' -> NAME)       , 6
'['      , 10       , '('          , 10       , (UNKNOWN_CHAR -> ',')   , 6
'class'  , 8        , '{'          , 10       , (',' -> ':')            , 6
'def'    , 7        , 99           , 10       , ('(' -> '[')            , 6
'import' , 6        , '['          , 9        , (']' -> '}')            , 5
'.'      , 6        , ']'          , 7        , (UNKNOWN_CHAR -> '.')   , 4
'from'   , 5        , '*'          , 6        , (NAME -> ']')           , 4
'return' , 3        , '/'          , 5        , ('...' -> ',')          , 4
'in'     , 2        , '//'         , 5        , ('in' -> NAME)          , 4
'...'    , 2        , ','          , 4        , ('<' -> '&')            , 3
'except' , 2        , '}'          , 4        , ('>' -> ';')            , 3
*/

fun computePatchStats() =
  preprocessStackOverflowInParallel(take = 100_000).map { (broke, _, minfix) ->
    val brokeLexed = broke.lexToStrTypesAsPython()
    val minfixLexed = minfix.lexToStrTypesAsPython()
    val patch = extractPatch(brokeLexed, minfixLexed)
    patch.changedIndices().map {
      if (patch[it].old == "") "INS, ${patch[it].new}"
      else if (patch[it].new == "") "DEL, ${patch[it].old}"
      else "SUB, ${patch[it].old} -> ${patch[it].new}"
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "${it.first}, ${it.second}" }
    .also { println("Type, Edit, Frequency\n$it") }

/*
Type, Edit, Frequency
INS, ')' [')'] END, 467
INS, NAME ['('] NAME, 456
INS, NAME [')'] END, 198
INS, NAME ['('] STRING, 174
DEL, NAME [NAME] NEWLINE, 154
DEL, START ['>>'] '>', 142
INS, NAME [')'] NEWLINE, 131
DEL, NAME [NAME] END, 124
DEL, START ['>'] NAME, 117
DEL, ')' [UNKNOWN_CHAR] END, 112
INS, ')' [NEWLINE] NAME, 110
INS, ']' [')'] END, 103
INS, NEWLINE [98] NAME, 92
INS, ')' [')'] NEWLINE, 92
INS, ':' [NAME] END, 89
INS, NAME ['='] NAME, 82
DEL, NAME [NAME] '.', 81
INS, ')' [':'] NEWLINE, 80
DEL, NAME [NAME] ':', 80
INS, 99 [99] END, 79
DEL, NAME [NAME] NAME, 76
INS, STRING [')'] END, 76
DEL, ')' [')'] END, 70
DEL, STRING [NAME] UNKNOWN_CHAR, 65
INS, STRING [')'] NEWLINE, 62
DEL, START [UNKNOWN_CHAR] NAME, 60
DEL, NAME [NAME] '(', 58
SUB, '(' [UNKNOWN_CHAR -> STRING] NAME, 57
DEL, STRING [NAME] ')', 56
INS, NUMBER [','] NUMBER, 55
SUB, 98 ['pass' -> NAME] NEWLINE, 55
DEL, STRING [NAME] ']', 54
SUB, ',' [UNKNOWN_CHAR -> STRING] NAME, 53
INS, NEWLINE [99] NAME, 53
INS, ']' [')'] NEWLINE, 52
INS, '}' ['}'] END, 48
DEL, NAME [UNKNOWN_CHAR] ']', 47
INS, START ['import'] NAME, 45
INS, ')' [')'] NAME, 42
INS, NEWLINE [98] 'def', 41
DEL, ']' [UNKNOWN_CHAR] END, 40
DEL, ')' ['.'] END, 39
INS, NAME [','] NAME, 39
DEL, STRING [NAME] STRING, 39
INS, ')' [']'] END, 39
DEL, NAME [UNKNOWN_CHAR] ')', 38
SUB, STRING [NAME -> ','] UNKNOWN_CHAR, 37
DEL, NAME [NAME] '=', 37
INS, ']' ['}'] END, 36
INS, ']' [']'] END, 35
INS, NAME ['.'] NAME, 34
DEL, NAME [UNKNOWN_CHAR] END, 33
SUB, '=' [UNKNOWN_CHAR -> STRING] NAME, 33
DEL, NAME [':'] NEWLINE, 33
INS, '(' [')'] END, 33
DEL, NAME ['('] NAME, 33
DEL, 99 [99] END, 32
INS, ',' [']'] END, 32
SUB, '[' [UNKNOWN_CHAR -> STRING] NAME, 32
DEL, NAME [NAME] ',', 32
INS, STRING [STRING] NEWLINE, 31
DEL, NAME [NAME] ')', 31
INS, STRING [','] NAME, 31
INS, NAME [':'] NEWLINE, 31
INS, STRING [','] STRING, 31
DEL, STRING [UNKNOWN_CHAR] ')', 31
DEL, ']' ['.'] END, 29
INS, NAME ['in'] NAME, 29
INS, ']' [NEWLINE] NAME, 28
INS, NEWLINE [98] 'if', 28
INS, START ['def'] NAME, 27
INS, START ['{'] STRING, 27
DEL, STRING [UNKNOWN_CHAR] ']', 27
INS, NUMBER [','] STRING, 27
INS, ',' ['}'] END, 27
DEL, START ['**'] NAME, 27
DEL, ')' [':'] END, 26
INS, ':' ['...'] END, 26
DEL, NAME [NAME] '[', 25
INS, STRING [']'] ')', 25
DEL, STRING [NAME] '.', 25
INS, ')' [')'] 'for', 25
INS, STRING ['}'] END, 24
INS, START ['from'] NAME, 24
DEL, ')' ['**'] END, 24
DEL, NAME ['>'] NEWLINE, 24
INS, START ['class'] NAME, 24
INS, 'for' [NAME] 'in', 24
INS, NAME [','] STRING, 23
SUB, STRING ['=' -> ':'] STRING, 23
INS, '}' [')'] END, 23
DEL, NAME ['.'] END, 22
INS, ':' ['('] END, 22
DEL, STRING [NAME] ',', 22
DEL, START [NEWLINE] 98, 22
INS, NAME ['('] '[', 22
INS, ']' [']'] ')', 21
DEL, ',' [NEWLINE] STRING, 20
INS, NEWLINE [98] 'for', 20
SUB, 98 ['break' -> NAME] NEWLINE, 20
*/

fun readGoodBIFIAndCorrupt() =
  readBIFIContents().take(100_000)
    .map { "\n$it\n".lexToStrTypesAsPython().joinToString(" ") }
    .map { it.syntheticallyCorrupt() to it }

fun String.syntheticallyCorrupt(): String =
  tokenizeByWhitespace().toMutableList().let { tokens ->
    var corrupted = tokens.map { it.toPythonIntType() }

    val initREAs = contextCSV.relevantEditActions(corrupted)
    while (corrupted.isValidPython()) {
      corrupted = corrupted.sampleEditTrajectory(contextCSV, initREAs).first
    }

    corrupted.joinToString(" ") { it.toPyRuleName() }
  }

var progress = 0
fun computePatchTrigramStats(toTake: Int = 100000) =
  preprocessStackOverflowInParallel(take=toTake).map { (broke, _, minfix) ->
    val brokeLexed = listOf("BOS") + broke.lexToStrTypesAsPython() + listOf("EOS")
    val minfixLexed = listOf("BOS") + minfix.lexToStrTypesAsPython() + listOf("EOS")
    val patch: Patch = extractPatch(brokeLexed, minfixLexed)
    progress++.also { if (it % 100 == 0) println("Processed $it/$toTake patches") }
    patch.run {
      changedIndices().map { i ->
        val (old, new) = get(i).old to get(i).new

        if (old == "") "INS, ${sln(i)}, , ${sro(i)}, $new"
        else if (new == "") "DEL, ${sln(i)}, $old, ${sro(i)}, "
        else "SUB, ${sln(i)}, $old, ${sro(i)}, $new"
      }
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }
    .joinToString("\n", "Type, Left, Old Mid, Right, New Mid, Frequency\n") { "${it.first}, ${it.second}" }
    .also {
      File("context_edits.csv").apply {
        writeText(it.reformatCSVIntoPrettyColumns())
        println(readLines().take(100).joinToString("\n"))
        println("Edit trigrams written to: $absolutePath")
      }
    }

// Same as above, but with the reverse patch order (minfix -> broke) instead
fun corruptedTrigramStats(toTake: Int = 100000) =
  preprocessStackOverflowInParallel(take=toTake).map { (broke, _, minfix) ->
    val brokeLexed = listOf("BOS") + broke.lexToStrTypesAsPython() + listOf("EOS")
    val minfixLexed = listOf("BOS") + minfix.lexToStrTypesAsPython() + listOf("EOS")
    val patch: Patch = extractPatch(minfixLexed, brokeLexed)
    progress++.also { if (it % 100 == 0) println("Processed $it/$toTake patches") }
    patch.run {
      changedIndices().map { i ->
        val (old, new) = get(i).old to get(i).new

        if (old == "") "INS, ${sln(i)}, , ${sro(i)}, $new"
        else if (new == "") "DEL, ${sln(i)}, $old, ${sro(i)}, "
        else "SUB, ${sln(i)}, $old, ${sro(i)}, $new"
      }
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }
    .joinToString("\n", "Type, Left, Old Mid, Right, New Mid, Frequency\n") { "${it.first}, ${it.second}" }
    .also {
      File("context_typos.csv").apply {
        writeText(it.reformatCSVIntoPrettyColumns())
        println(readLines().take(100).joinToString("\n"))
        println("Typo trigrams written to: $absolutePath")
      }
    }

fun String.reformatCSVIntoPrettyColumns(): String {
  val lines = split('\n')
  if (lines.isEmpty()) return this

  // Split each line into columns
  val linesByColumns = lines.map { it.split(", ").toMutableList() }

  // Find the max length of each column
  val maxLengths = IntArray(linesByColumns[0].size) { 0 }
  for (columns in linesByColumns)
    for ((index, column) in columns.withIndex())
      maxLengths[index] = maxOf(maxLengths[index], column.trim().length)

  // Pad each element in the columns
  for (columns in linesByColumns)
    for ((index, column) in columns.withIndex())
      columns[index] = column.trim().padEnd(maxLengths[index], ' ')

  // Reassemble the lines and then the entire string
  return linesByColumns.joinToString("\n") { it.joinToString(" , ") }
}

fun Sequence<Σᐩ>.train(csv: File) {
  measureTimedValue {
    println("Training $MARKOV_MEMORY Markov chain")
    asStream().parallel().map {
      "\n$it\n".mapToUnquotedPythonTokens().let { "BOS $it EOS" }
        .tokenizeByWhitespace().asSequence().toMarkovChain(MARKOV_MEMORY)
    }.reduce { t, u -> t + u }.get()
      .also { csv.also { println("Writing CSV to ${it.absolutePath}") }.writeText(it.toCSV()) }
  }.let { println("Trained $MARKOV_MEMORY-gram Markov chain on ${it.value.counter.total.get()} " +
      "PY150 tokens in ${it.duration.inWholeSeconds}s"); it.value }
}

/** Streams an acyclic repair DFA's distinct words in descending WDFA log probability, including STOP. */
fun ai.hypergraph.kaliningraph.automata.DFSM.repairsByWDFA(
  model: WFA, terminalNames: List<String>, deadlineNanos: () -> Long
): Sequence<Pair<List<String>, Double>> = sequence {
  class Deadline : RuntimeException()
  fun checkDeadline() { if (System.nanoTime() - deadlineNanos() >= 0) throw Deadline() }
  data class Prefix(val previous: Prefix?, val terminal: Int, val depth: Int)
  data class Path(
    val state: String, val weightedState: Int, val score: Double, val prefix: Prefix?,
    val complete: Boolean, val bound: Double, val order: Long
  )
  val suffixScores = HashMap<Pair<String, Int>, Double>()
  fun bestSuffix(state: String, weightedState: Int): Double {
    checkDeadline()
    val key = state to weightedState
    suffixScores[key]?.let { return it }
    var best = if (state in F) model.tokenFinalWeight(weightedState) else Double.NEGATIVE_INFINITY
    for ((terminal, next) in deltaMap[state].orEmpty()) {
      checkDeadline()
      val step = model.stepTokenState(weightedState, terminalNames[terminal])
      best = maxOf(best, step.delta + bestSuffix(next, step.target))
    }
    suffixScores[key] = best
    return best
  }
  val queue = PriorityQueue(
    compareByDescending<Path> { it.bound }
      .thenByDescending { it.complete }
      .thenByDescending { it.prefix?.depth ?: 0 }
      .thenBy { it.order }
  )
  var order = 0L
  try {
    checkDeadline()
    val start = model.tokenStartState()
    val startScore = model.tokenStartWeight()
    queue.add(Path(q_alpha, start, startScore, null, false, startScore + bestSuffix(q_alpha, start), order++))
    while (queue.isNotEmpty()) {
      checkDeadline()
      val path = queue.remove()
      if (path.complete) {
        val tokens = ArrayList<String>(path.prefix?.depth ?: 0)
        var prefix = path.prefix
        while (prefix != null) {
          tokens.add(terminalNames[prefix.terminal])
          prefix = prefix.previous
        }
        tokens.reverse()
        yield(tokens to path.score)
        continue
      }
      if (path.state in F) {
        val score = path.score + model.tokenFinalWeight(path.weightedState)
        queue.add(path.copy(score = score, complete = true, bound = score, order = order++))
      }
      for ((terminal, next) in deltaMap[path.state].orEmpty()) {
        checkDeadline()
        val step = model.stepTokenState(path.weightedState, terminalNames[terminal])
        val score = path.score + step.delta
        val prefix = Prefix(path.prefix, terminal, (path.prefix?.depth ?: 0) + 1)
        queue.add(Path(next, step.target, score, prefix, false, score + bestSuffix(next, step.target), order++))
      }
    }
  } catch (_: Deadline) {
    // A caller can consume every candidate available before its shared per-instance deadline.
  }
}
