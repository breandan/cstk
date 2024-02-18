package edu.mcgill.cstk.experiments.repair

import NUM_CORES
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.sampling.pow
import ai.hypergraph.markovian.mcmc.*
import ai.hypergraph.kaliningraph.types.*
import com.beust.klaxon.*
import edu.mcgill.cstk.experiments.repair.data.symbolCounts
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.utils.*
import org.apache.datasketches.frequencies.ErrorType
import org.kosat.round
import java.io.*
import java.math.*
import java.net.*
import java.util.regex.Pattern
import java.util.stream.*
import kotlin.math.*
import kotlin.streams.*
import kotlin.system.measureTimeMillis
import kotlin.time.*


val brokenSnippetURL =
  "https://raw.githubusercontent.com/gsakkas/seq2parse/main/src/datasets/python/erule-test-set-generic.txt"

val brokenPythonSnippets by lazy {
  // Download file if "resources/datasets/python/seq2parse/erule-test-set-generic.txt" doesn't exist
  "/src/main/resources/datasets/python/seq2parse/erule-test-set-generic.txt"
    .let { File(File("").absolutePath + it) }
    .apply {
      if (!exists()) URL(brokenSnippetURL).also {
        println("Downloading broken Python snippets from $it")
        writeText(it.readText())
      }
    }.readLines().asSequence()
}

val P_seq2parse: MarkovChain<Σᐩ> by lazy {
  measureTimedValue {
    brokenPythonSnippets.toList().parallelStream().map { "BOS $it EOS" }
      .map { it.tokenizeByWhitespace().asSequence().toMarkovChain(1) }
      .reduce { t, u -> t + u }.get()
  }.let { println("Trained Markov chain on ${it.value.counter.total.get()} Seq2Parse tokens in ${it.duration.inWholeMilliseconds}ms"); it.value }
}

const val bifi_filename = "src/main/resources/datasets/python/bifi/data/orig_good_code/orig.good.json"
const val home_prefix = "/scratch/b/bengioy/breandan"
const val bifi_filenameCC = "$home_prefix/bifi/data/orig_good_code/orig.good.cc.json"
const val MARKOV_MEMORY = 6

// Python3 snippets
// https://github.com/michiyasunaga/BIFI?tab=readme-ov-file#about-the-github-python-dataset
val P_BIFI: MarkovChain<Σᐩ> by lazy {
  measureTimedValue {
    val numToks = 100_000.let { if (NUM_CORES < 20) it else Int.MAX_VALUE }
    // If running on Compute Canada, use the larger dataset
    val file: File = File(bifi_filenameCC).let { if (it.exists()) it else File(bifi_filename) }
    readBIFIContents(file = file).take(numToks).asStream().parallel().map {
      "\n$it\n".mapToUnquotedPythonTokens().let { "BOS $it EOS" }
      .tokenizeByWhitespace().asSequence().toMarkovChain(MARKOV_MEMORY)
    }.reduce { t, u -> t + u }.get()
    .also { if (20 < NUM_CORES) { File("$home_prefix/ngrams_BIFI_$MARKOV_MEMORY.csv".also { println("Writing CSV to $it")}).writeText(it.toCSV()) } }
  }.let { println("Trained Markov chain on ${it.value.counter.total.get()}" +
      "BIFI tokens in ${it.duration.inWholeSeconds}s"); it.value }
}

// Python2 snippets, about ~20x longer on average than BIFI
// https://www.sri.inf.ethz.ch/py150
val P_PY150: MarkovChain<Σᐩ> by lazy {
  measureTimedValue {
    val numToks = 10_000.let { if (NUM_CORES < 20) it else Int.MAX_VALUE }
    readPY150Contents().take(numToks).asStream().parallel().map {
      "\n$it\n".mapToUnquotedPythonTokens().let { "BOS $it EOS" }
        .tokenizeByWhitespace().asSequence().toMarkovChain(MARKOV_MEMORY)
    }.reduce { t, u -> t + u }.get()
    .also { if (20 < NUM_CORES) { File("$home_prefix/ngrams_PY150_$MARKOV_MEMORY.csv".also { println("Writing CSV to $it")}).writeText(it.toCSV()) } }
  }.let { println("Trained Markov chain on ${it.value.counter.total.get()}" +
      "PY150 tokens in ${it.duration.inWholeSeconds}s"); it.value }
}

val P_BIFI_PY150: MarkovChain<Σᐩ> by lazy { P_BIFI + P_PY150 }

val topTokens by lazy { P_BIFI.topK(200).map { it.first } + "ε" - "BOS" - "EOS" }// + errDeck

typealias CooccurenceMatrix = List<List<Double>>
val pythonCooccurence: CooccurenceMatrix by lazy {
  readContents("parse_fixes.json").asStream().parallel()
    .map { "\n$it\n".lexToIntTypesAsPython() }
    .computeCooccurrenceProbs(pythonVocabBindex.size)
}

val mostCommonTokens by lazy {
  P_seq2parse.counter.rawCounts
    .getFrequentItems(ErrorType.NO_FALSE_NEGATIVES)
    .associate { it.item to it.estimate }
    .entries.sortedByDescending { it.value }.take(62)
    .onEach { println("${it.key} (${it.value})") }
    .map { it.key }.toSet()
}

/*
Compute Canada run command:

./gradlew shadowJar &&
scp build/libs/gym-fs-fat-1.0-SNAPSHOT.jar breandan@niagara.computecanada.ca:/home/b/bengioy/breandan/cstk &&
ssh breandan@niagara.computecanada.ca 'cd /home/b/bengioy/breandan/cstk && git pull && sbatch submit_job.sh'

Local run command:

./gradlew pythonSnippetRepair
*/

fun main() {
//  evaluateTidyparseOnSeq2Parse15k()
  evaluateTidyparseOnStackoverflow()
//  evaluateSeq2ParseOnStackOverflowDataset()
//  println(extractErrProbs().joinToString(", ", "listOf(", ")") { "\"${it.first}\" to ${it.second}" })
//  runSingleExample()
}

fun runSingleExample() {
  val clock = TimeSource.Monotonic.markNow()
  val example = "NAME = ( NAME . NAME ( NAME"
  parallelRepair(
    prompt = example.lexToStrTypesAsPython().joinToString(" ", "", " NEWLINE"),
    fillers = topTokens,
//        hints = pythonErrorLocations(humanError.lexToIntTypesAsPython()),
    maxEdits = 4,
    admissibilityFilter = { map { pythonVocabBindex.getUnsafe(it) ?: it.toInt() }.isValidPython() },
    // TODO: incorporate parseable segmentations into scoring mechanism to prioritize chokepoint repairs
    // TODO: only score the locations that are actually being modified to avoid redundant work
    scoreEdit = { P_BIFI.score(listOf("BOS") + it + "EOS") }
  ).onEach { println(prettyDiffNoFrills(example, it.resToStr().replace("'", ""))) }
    .also { println("Found ${it.size} total repairs in ${clock.elapsedNow().inWholeSeconds}s") }
}

fun evaluateSeq2ParseOnStackOverflowDataset() {
  class Seq2ParsePrecision {
    var syntaxPrecision = 0.0
    var humanFixPrecision = 0.0
    var chrMatchPrecision = 0.0
    var samples = 0
    var latency = 0.0
    fun update(seq2parseWasParseable: Boolean,
               seq2parseFixTks: List<Σᐩ>, minFixTks: List<Σᐩ>,
               humanFix: String, seq2parseFix: String, latency: Int) {
      samples += 1
      syntaxPrecision += if (seq2parseWasParseable) 1.0 else 0.0
      val avgSyntaxPrecision = (syntaxPrecision / (samples)).round(3)
      println("Average Syntactic precision@1: $avgSyntaxPrecision")
      humanFixPrecision += if (seq2parseFixTks == minFixTks) 1.0 else 0.0
      val avgHumanFixPrecision = (humanFixPrecision / (samples)).round(3)
      println("Average HumanEval precision@1: $avgHumanFixPrecision")
      chrMatchPrecision += if (seq2parseFix == humanFix) 1.0 else 0.0
      val avgChrMatchPrecision = (chrMatchPrecision / (samples)).round(3)
      println("Average CharMatch precision@1: $avgChrMatchPrecision")
      this.latency += latency
      val avgLatency = (this.latency / samples).round(3)
      println("Average latency to produce a single sample: ${avgLatency}ms\n")
    }
  }

  val totalPrecision = Seq2ParsePrecision()
  val editPrecision = (1..MAX_PATCH_SIZE).associateWith { Seq2ParsePrecision() }
  var latency: Int
  var percentageOfFixesShorterThanSeq2Parse = 0.0
  var percentageOfFixesLongerThanSeq2Parse = 0.0
  preprocessStackOverflow()
//    .filter { it.second != it.third }
    .forEachIndexed { i, (humanError, humanFix, minimumFix) ->
      val errTks = humanError.lexToStrTypesAsPython()
      val minFixTks = minimumFix.lexToStrTypesAsPython()

      val minFixSize = extractPatch(errTks, minFixTks).changedIndices().size

      val seq2parseFix = measureTimedValue { seq2parseFix(humanError) }.let {
        latency = it.duration.inWholeMilliseconds.toInt()
        it.value
      }

      val seq2parseFixTks = seq2parseFix.lexToStrTypesAsPython()
      val seq2parseEditSize = extractPatch(errTks, seq2parseFixTks).changedIndices().size

      val seq2parseWasParseable = seq2parseFix.isValidPython {
        val s2pDiffFullDiff = prettyDiffHorizontal(humanError, seq2parseFix, "human error", "seq2parse fix")
        println("\nSeq2Parse fix did NOT parse: $it!\n$s2pDiffFullDiff\n\n")
      }

      if (seq2parseFixTks == minFixTks)
        println("Abstract tokens matched but there was a character diff:\n" +
          prettyDiffHorizontal(minimumFix, seq2parseFix, "human fix", "seq2parse fix"))

      println("Original code error: ${errTks.joinToString(" ")}")
      val minDiff = prettyDiffNoFrills(errTks.joinToString(" "), minFixTks.joinToString(" "))
      println("Minimized human fix: $minDiff (parsed=${minimumFix.isValidPython()})")
      val s2pDiff = prettyDiffNoFrills(errTks.joinToString(" "), seq2parseFixTks.joinToString(" "))
      println("Seq2Parse tokenized: $s2pDiff (parsed=${seq2parseWasParseable}, matched=${seq2parseFixTks == minFixTks})\n")

      if (minFixSize > seq2parseEditSize) percentageOfFixesLongerThanSeq2Parse += 1.0
      else if (minFixSize < seq2parseEditSize) percentageOfFixesShorterThanSeq2Parse += 1.0
      println("Percentage of fixes shorter than Seq2Parse: ${(percentageOfFixesShorterThanSeq2Parse / (i + 1)).round(3)}")
      println("Percentage of fixes longer than Seq2Parse : ${(percentageOfFixesLongerThanSeq2Parse / (i + 1)).round(3)}")

      println("Ranking stats for $minFixSize-edit fixes (${editPrecision[minFixSize]!!.samples} samples):")
      editPrecision[minFixSize]!!.run { update(seq2parseWasParseable, seq2parseFixTks, minFixTks, humanFix, seq2parseFix, latency) }
      println("\nTotal ranking stats across all edit sizes (${totalPrecision.samples} samples):")
      totalPrecision.run { update(seq2parseWasParseable, seq2parseFixTks, minFixTks, humanFix, seq2parseFix, latency) }
    }
}

class RankStats(val name: String = "Total") {
  val upperBound = TIMEOUT_MS / 1000
  val time = (1000..TIMEOUT_MS step 1000).toSet() //(1..10).toSet()//setOf(2, 5, 10) + (20..upperBound step 20).toSet()
  // Mean Reciprocal Rank
  val timedMRR = time.associateWith { 0.0 }.toMutableMap()
  // Precision at K, first int is K, second is the time cutoff
  val timedPAK =
    (setOf(1, 5, 10, Int.MAX_VALUE) * (time.toSet()))
      .associateWith { 0.0 }.toMutableMap()

  val minAdmitSetSize = mutableMapOf<Π2A<Int>, MutableList<Int>>()

  val densities = mutableListOf<BigDecimal>()

  fun calcNormalizingConstantSize(n: Int, k: Int): BigDecimal =
    (1..k).sumOf { c ->
      ((c * n + n + 1) choose c).toBigDecimal() *
        (pythonVocabBindex.size + 1).pow(c).toBigDecimal()
    }

  fun updateDensity(repairs: List<Repair>) {
    if (repairs.isEmpty()) return
    val origStrSize = repairs.first().orig.size
    val minRepairDist = repairs.maxOf { it.edit.size }

    val density = repairs.size.toBigDecimal().divide(calcNormalizingConstantSize(origStrSize, minRepairDist), 15, RoundingMode.HALF_UP)
    densities.add(density)

    val kRepairsCount = repairs.filter { it.edit.size == minRepairDist }.size
    val key = origStrSize to minRepairDist
    minAdmitSetSize[key].let {
      if (it != null) it.add(kRepairsCount)
      else minAdmitSetSize[key] = mutableListOf(kRepairsCount)
    }

    fun List<Int>.meanVar(): String =
      if (isEmpty()) "N/A"
      else let { "(μ=" + it.average().round(1) + ", σ=" + it.map { it.toDouble() }.variance().round(1) + ")" }

    fun List<BigDecimal>.minMeanMax(): String =
      if (isEmpty()) "N/A" else let { "(μ=${it.mean()}, σ²=${it.variance()})" }

    val strBld = StringBuilder()
    strBld.append("$name density stats:" +
      " admits=${minAdmitSetSize.values.flatten().meanVar()}, density=${densities.minMeanMax()}, |Σ|=${pythonVocabBindex.size}\n\n")
    (1..3).forEach { edits ->
      strBld.append("Δ($edits) = ")
      // Buckets of size 10
      (20 until minAdmitSetSize.keys.maxOf { it.first }.coerceAtLeast(20) step 20)
        .forEach { len ->
          ((len - 10)..len).fold(listOf<Int>()) { a, it ->
            a + (minAdmitSetSize[it to edits] ?: emptyList())
          }.let { strBld.append("$len: ${it.meanVar()}, ") }
        }
      strBld.append("\n")
    }

    printInABox(strBld.toString())
  }

  var samplesEvaluated = 0

  fun update(repairProposals: List<Repair>, groundTruthRepair: String) {
    samplesEvaluated += 1
    (timedMRR.keys).forEach { ms ->
      repairProposals.filter { it.timeMS <= ms }
        .let {
          val mrr = it.indexOfFirst { it.matches(groundTruthRepair) }
            .let { if (it == -1) 0.0 else 1.0 / (it + 1) }
          timedMRR[ms] = (timedMRR[ms] ?: 0.0) + mrr
        }
    }

    (timedPAK.keys).forEach { (k, ms) ->
      repairProposals.filter { it.timeMS <= ms }
        .let {
          val pak = (if(k == Int.MAX_VALUE) it else it.take(k))
            .count { it.matches(groundTruthRepair) }.toDouble()
          timedPAK[k to ms] = (timedPAK[k to ms] ?: 0.0) + pak
        }
    }

    fun Int.roundToTenths() = (toDouble() / 1000).round(1)

    var summary = "$name ranking statistics across $samplesEvaluated samples...\n"
    val latestMRRs = timedMRR.entries.sortedByDescending { it.key }
      .joinToString(", ") { (k, v) ->
        "${(k.roundToTenths()).round(1)}s: ${(v / samplesEvaluated).round(3)}"
      }
    summary += "\nMRR=  $latestMRRs"

    val latestPAKs = timedPAK.entries.groupBy { it.key.first }
      .mapValues { (_, v) ->
        v.sortedByDescending { it.key.second }
          .joinToString(", ") { (p, v) ->
            "${p.second.roundToTenths()}s: ${(v / samplesEvaluated).round(3)}"
          }
      }.entries.joinToString("\n") { (k, v) ->
        "P@${if (k == Int.MAX_VALUE) "All" else k}=".padEnd(6) + v
      }
    summary += "\n$latestPAKs"
    printInABox(summary)

    updateDensity(repairProposals)
  }
}

var MAX_PATCH_SIZE = 3

// Tracks ranking statistics for each patch size and across all patch sizes
class MultiRankStats {
  // Total ranking statistics across all patch sizes
  val totalRankStats = RankStats()
  // Ranking statistics for each patch size
  val patchRankStats =
    (1..MAX_PATCH_SIZE).associateWith { RankStats("$it-edit") }.toMutableMap()
}

fun evaluateTidyparseOnStackoverflow() {
//  val errDeck = pythonErrProbs.expandByFrequency(10)
  println("Top tokens: $topTokens")

  val multiRankStats = MultiRankStats()

  preprocessStackOverflow()
    .forEach { (humanError, humanFix, minimumFix) ->
//      println("$a\n$b")
      val coarseBrokeTks = humanError.lexToStrTypesAsPython()
      val coarseFixedTks = minimumFix.lexToStrTypesAsPython()
      val coarseBrokeStr = coarseBrokeTks.joinToString(" ", "", " NEWLINE")
      val coarseFixedStr = coarseFixedTks.joinToString(" ", "", " NEWLINE")

      val patch: Patch = extractPatch(
        humanError.lexToStrTypesAsPython(),
        minimumFix.lexToStrTypesAsPython()
      )

      val patchSize = patch.changedIndices().size

      if (2 < patchSize) return@forEach

      println("Original vs. fixed source:\n${prettyDiffNoFrillsTrimAndAlignWithOriginal(coarseBrokeStr, coarseFixedStr)}")
      println("\n\n")

      val startTime = System.currentTimeMillis()
//      val segmentation = Segmentation.build(seq2parsePythonCFG, coarseBrokeStr)

      println("Repairing ($NUM_CORES cores): $coarseBrokeStr")

      parallelRepair(
        prompt = coarseBrokeStr,
        fillers = topTokens,
//        hints = pythonErrorLocations(humanError.lexToIntTypesAsPython()),
        maxEdits = 4,
        admissibilityFilter = { isValidPython() },
        // TODO: incorporate parseable segmentations into scoring mechanism to prioritize chokepoint repairs
        // TODO: only score the locations that are actually being modified to avoid redundant work
        scoreEdit = { P_BIFI.score(listOf("BOS") + it + "EOS") }
      )
//      seq2parsePythonCFG.metrizedRepair(coarseBrokeTks, P_BIFI)
//      seq2parsePythonCFG.ptreeRepair(coarseBrokeTks, { P_BIFI.score(listOf("BOS") + it + "EOS") })
      .also { repairs: List<Repair> ->
        repairs.take(20).apply { println("\nTop $size repairs:\n") }.forEach {
          println("Δ=${it.scoreStr()} repair (${it.elapsed()}): ${prettyDiffNoFrills(coarseBrokeStr, it.resToStr())}")
          //        println("(LATEX) Δ=${levenshtein(prompt, it)} repair: ${latexDiffSingleLOC(prompt, it)}")
        }

        val contained = repairs.any { coarseFixedStr == it.resToStr() }
        val elapsed = System.currentTimeMillis() - startTime

        println("\nFound ${repairs.size} valid repairs in ${elapsed}ms, or roughly " +
          "${(repairs.size / (elapsed/1000.0)).toString().take(5)} repairs per second.")

        val idx = repairs.indexOfFirst { it.resToStr() == coarseFixedStr }
        val minRepairState = if (!contained) "NOT" else "#$idx (${repairs[idx].timeMS}ms)"
        println("Minimized repair was $minRepairState in repair proposals!")

        if (contained) {
          println(prettyDiffHorizontal(humanError, minimumFix,
            "humanErr", "minFix") + "\n")
          println(prettyDiffNoFrills(coarseBrokeStr, coarseFixedStr) + "\n")
          latexDiffMultilineStrings(coarseBrokeStr, coarseFixedStr)
            .let { (a, b) -> println("$a\n\n$b\n") }
          latexDiffMultilineStrings(humanError, minimumFix)
            .let { (a, b) -> println("$a\n\n$b") }
        }

//      compareSeq2ParseFix(humanError, coarseBrokeStr, coarseFixedStr, repairs)
        updateRankingStats(repairs, coarseFixedStr, multiRankStats, patchSize)
      }
    }
  }

fun pythonErrorLocations(coarseBrokeTks: List<Int>): List<Int> =
  listOf(coarseBrokeTks.getIndexOfFirstPythonError())

fun Σᐩ.mapToUnquotedPythonTokens() =
  lexToStrTypesAsPython().joinToString(" ") {
    if (1 < it.length && it.startsWith("'") &&
      it.endsWith("'")) it.drop(1).dropLast(1)
    else if (it == "98") "INDENT"
    else if (it == "99") "DEDENT"
    else it
  }

// Returns a triple of: (1) the broken source, (2) the human fix, and (3) the minimized fix
fun preprocessStackOverflow(
  maxPatchSize: Int = MAX_PATCH_SIZE,
  lengthBounds: IntRange = 0..Int.MAX_VALUE,
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
  fixedSnippets: Sequence<String> = readContents("parse_fixes.json"),
): Sequence<Π3A<Σᐩ>> =
  brokeSnippets.zip(fixedSnippets)//.asStream().parallel()
    .filter { (broke, fixed) ->
//      '"' !in broke && '\'' !in broke &&
      (broke.lines().size - fixed.lines().size).absoluteValue <= maxPatchSize &&
        broke.mapToUnquotedPythonTokens().tokenizeByWhitespace().let {
          it.size in lengthBounds && it.all { it in seq2parsePythonCFG.terminals }
        } && (!broke.isValidPython() && fixed.isValidPython())
    }
    .distinct()
    .minimizeFix({ tokenizeAsPython(true) }, { isValidPython() })
    .filter { (broke, fixed, minfix) ->
      val mftks = minfix.mapToUnquotedPythonTokens()
      val bktks = broke.mapToUnquotedPythonTokens()

      levenshtein(bktks, mftks) <= maxPatchSize && minfix.isValidPython() &&
        "$mftks NEWLINE" in seq2parsePythonCFG.language
    }
    .filter { (broke, fixed, minfix) ->
//      val (brokeTokens, minFixedTokens) =
//        broke.lexToIntTypesAsPython() to minfix.lexToIntTypesAsPython()
//      (brokeTokens.size - fixedTokens.size).absoluteValue < 10 &&

      val minpatch =
        extractPatch(broke.lexToStrTypesAsPython(), minfix.lexToStrTypesAsPython())
      val (brokeVis, fixedVis, minfixVis) =
        broke.visibleChars() to fixed.visibleChars() to minfix.visibleChars()

      minpatch.changedIndices().size <= maxPatchSize &&
      brokeVis != fixedVis && minfixVis != brokeVis // && fixedVis != minfixVis
//      multisetManhattanDistance(brokeTokens, minFixedTokens).let { it in 1..5 }
    }.distinct().asSequence()
//    .map { (broke, fixed, minfix) ->
//      prettyDiffs(listOf(broke, fixed), listOf("original snippet", "human patch")).let { origDiff ->
//        prettyDiffs(listOf(broke, minfix), listOf("original snippet", "minimized patch")).let { minDiff ->
//          // Compare ASCII characters for a visible difference, if same do not print two
////          if (corrected.visibleChars() == minfix.visibleChars()) origDiff to "" else
//          origDiff to minDiff to broke to minfix
//        }
//      }
//    }
//    .shuffleOnline()

fun preprocessStackOverflowInParallel(
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
  fixedSnippets: Sequence<String> = readContents("parse_fixes.json"),
  take: Int
): Stream<Π3A<Σᐩ>> =
  brokeSnippets.zip(fixedSnippets).take(take).asStream().parallel()
    .filter { (broke, fixed) ->
//      '"' !in broke && '\'' !in broke &&
//      broke.tokenizeAsPython().size < 40 &&
        (!broke.isValidPython() && fixed.isValidPython()) &&
        (broke.lines().size - fixed.lines().size).absoluteValue < 4
    }
    .minimizeFix({ tokenizeAsPython(true) }, { isValidPython() })
    .filter { (broke, fixed, minfix) ->
      val minpatch = extractPatch(broke.lexToStrTypesAsPython(), minfix.lexToStrTypesAsPython())
      val (brokeVis, fixedVis, minfixVis) = broke.visibleChars() to fixed.visibleChars() to minfix.visibleChars()

      minfix.isValidPython() &&
        minpatch.changedIndices().size <= 5 &&
        brokeVis != fixedVis && brokeVis != minfixVis// && fixedVis != minfixVis
    }

private fun compareSeq2ParseFix(
  humanError: Σᐩ,
  coarseBrokeStr: String,
  coarseFixedStr: String,
  ourRepairs: List<Repair>
) {
  val seq2parseFix = seq2parseFix(humanError)
  val parseable = seq2parseFix.isValidPython()
  val seq2parseFixCoarse =
    seq2parseFix.lexToStrTypesAsPython().joinToString(" ", "", " NEWLINE")

  val idx = ourRepairs.indexOfFirst { it.matches(seq2parseFixCoarse) }
  println(
    "seq2parse fix (parseable=$parseable, idx=$idx, matches=${coarseFixedStr == seq2parseFixCoarse}): " +
      prettyDiffNoFrills(coarseBrokeStr, seq2parseFixCoarse)
  )
}

fun seq2parseFix(
  brokenCode: String,
  prefix: String = "http://127.0.0.1:5000/api/text?seq2parse="
) =
  try {
    URL("$prefix${URLEncoder.encode(brokenCode,"UTF-8")}").readText()
  } catch (e: Exception) { "ERROR (${e.message}):\n$brokenCode" }

private fun updateRankingStats(
  repairs: List<Repair>,
  coarseFixedStr: String,
  holeRankStats: MultiRankStats,
  editSize: Int
) {
  holeRankStats.totalRankStats.update(repairs, coarseFixedStr)
  holeRankStats.patchRankStats[editSize]!!.run { update(repairs, coarseFixedStr) }
}

// Draws a nice box around a multiline string using a single line box-drawing characters
// Rembering to pad the box by the length of the maximum line in the string
fun printInABox(s: String) =
  println(
    "\n┌" + "─".repeat(s.lines().maxBy { it.length }?.length?.plus(2) ?: 0) + "┐\n" +
      s.lines().joinToString("\n") { "│ $it" + " ".repeat((s.lines().maxBy { it.length }.length) - it.length) + " │" } + "\n" +
      "└" + "─".repeat(s.lines().maxBy { it.length }?.length?.plus(2) ?: 0) + "┘\n"
  )

fun evaluateTidyparseOnSeq2Parse15k() {
  brokenPythonSnippets.map {
      it.tokenizeByWhitespace()
        .joinToString(" ") { if (it in seq2parsePythonCFG.nonterminals) "<$it>" else it }
    }
    .filter { it.tokenizeByWhitespace().size < 50 }.distinct().take(300)
    .map { seq -> seq.tokenizeByWhitespace().joinToString(" ") { it.dropWhile { it == '_' }.dropLastWhile { it == '_' } } }
    .map { it.substringBefore(" ENDMARKER ") }
    .forEach { prompt ->
      val startTime = System.currentTimeMillis()
      val deck = seq2parsePythonCFG.terminals + "ε"
      val segmentation = Segmentation.build(seq2parsePythonCFG, prompt)

      println("Repairing: ${segmentation.toColorfulString()}\n")

      parallelRepair(
        prompt = prompt,
        fillers = deck,
        maxEdits = 4,
        admissibilityFilter = { this in seq2parsePythonCFG.language },
        // TODO: incorporate parseable segmentations into scoring mechanism to prioritize chokepoint repairs
        scoreEdit = { P_seq2parse.score(it) },
      ).also {
        it.take(20).apply { println("\nTop $size repairs:\n") }.forEach {
          println("Δ=${it.scoreStr()} repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.resToStr())}")
          //        println("(LATEX) Δ=${levenshtein(prompt, it)} repair: ${latexDiffSingleLOC(prompt, it)}")
        }

        val elapsed = System.currentTimeMillis() - startTime

        println("\nFound ${it.size} valid repairs in ${elapsed}ms, or roughly " +
          "${(it.size / (elapsed/1000.0)).toString().take(5)} repairs per second.")
      }
    }
}

val pythonErrProbs =
  listOf(
    "'-'" to 5, "'raise'" to 2, "'import'" to 40, "'None'" to 3, "')'" to 495,
    "'else'" to 3, "'in'" to 10, "'%'" to 2, "'pass'" to 14, "'True'" to 1,
    "'|'" to 4, "'=='" to 18, "'['" to 53, "':'" to 149, "'lambda'" to 5,
    "'...'" to 11, "98" to 108, "'.'" to 21, "99" to 105, "NUMBER" to 15,
    "'*'" to 1, "'yield'" to 1, "'is'" to 1, "NEWLINE" to 95, "'&'" to 16,
    "'from'" to 23, "'except'" to 8, "NAME" to 200, "'if'" to 6, "'}'" to 135,
    "';'" to 19, "'class'" to 13, "ε" to 2225, "'return'" to 9, "'as'" to 4,
    "'def'" to 17, "'/'" to 2, "'+'" to 18, "'~'" to 1, "']'" to 152,
    "'global'" to 1, "','" to 292, "'('" to 234, "'for'" to 5, "'='" to 49,
    "'**'" to 2, "'while'" to 1, "'{'" to 60, "'!='" to 1, "'del'" to 1,
    "STRING" to 221
  )

val s2pCFGStr =   """
START -> Stmts_Or_Newlines
Stmts_Or_Newlines -> Stmt_Or_Newline | Stmt_Or_Newline Stmts_Or_Newlines
Stmt_Or_Newline -> Stmt | Newline

Newline -> NEWLINE

Async_Funcdef -> Async_Keyword Funcdef
Funcdef -> Def_Keyword Simple_Name Parameters Colon Suite | Def_Keyword Simple_Name Parameters Arrow Test Colon Suite

Parameters -> Open_Paren Close_Paren | Open_Paren Typedargslist Close_Paren
Typedargslist -> Many_Tfpdef | Many_Tfpdef Comma | Many_Tfpdef Comma Star_Double_Star_Typed | Many_Tfpdef Comma Double_Star_Tfpdef | Star_Double_Star_Typed | Double_Star_Tfpdef
Star_Double_Star_Typed -> Star_Tfpdef | Star_Tfpdef Comma | Star_Tfpdef Comma Double_Star_Tfpdef
Star_Tfpdef_Comma -> Comma Tfpdef_Default | Comma Tfpdef_Default Star_Tfpdef_Comma
Star_Tfpdef -> Star_Op | Star_Op Star_Tfpdef_Comma | Star_Op Tfpdef | Star_Op Tfpdef Star_Tfpdef_Comma
Double_Star_Tfpdef -> Double_Star_Op Tfpdef | Double_Star_Op Tfpdef Comma
Many_Tfpdef -> Tfpdef_Default | Tfpdef_Default Comma Many_Tfpdef
Tfpdef_Default -> Tfpdef | Tfpdef Assign_Op Test

Varargslist -> Many_Vfpdef | Many_Vfpdef Comma | Many_Vfpdef Comma Star_Double_Star | Many_Vfpdef Comma Double_Star_Vfpdef | Star_Double_Star | Double_Star_Vfpdef
Star_Double_Star -> Star_Vfpdef | Star_Vfpdef Comma | Star_Vfpdef Comma Double_Star_Vfpdef
Star_Vfpdef_Comma -> Comma Vfpdef_Default | Comma Vfpdef_Default Star_Vfpdef_Comma
Star_Vfpdef -> Star_Op | Star_Op Star_Vfpdef_Comma | Star_Op Vfpdef | Star_Op Vfpdef Star_Vfpdef_Comma
Double_Star_Vfpdef -> Double_Star_Op Vfpdef | Double_Star_Op Vfpdef Comma
Many_Vfpdef -> Vfpdef_Default | Vfpdef_Default Comma Many_Vfpdef
Vfpdef_Default -> Vfpdef | Vfpdef Assign_Op Test

Tfpdef -> Vfpdef | Vfpdef Colon Test
Vfpdef -> NAME
Assign_Op -> =
Star_Op -> *
Double_Star_Op -> **
Arrow -> arrow

Stmt -> Simple_Stmt | Compound_Stmt
Simple_Stmt -> Small_Stmts Newline | Small_Stmts Semicolon Newline
Small_Stmts -> Small_Stmt | Small_Stmt Semicolon Small_Stmts
Small_Stmt -> Expr_Stmt | Del_Stmt | Pass_Stmt | Flow_Stmt | Import_Stmt | Global_Stmt | Nonlocal_Stmt | Assert_Stmt
Expr_Stmt -> Testlist_Star_Expr Annotated_Assign | Testlist_Star_Expr Aug_Assign Yield_Expr | Testlist_Star_Expr Aug_Assign Testlist_Endcomma | Testlist_Star_Exprs_Assign
Annotated_Assign -> Colon Test | Colon Test Assign_Op Test
Test_Or_Star_Expr -> Test | Star_Expr
Test_Or_Star_Exprs -> Test_Or_Star_Expr | Test_Or_Star_Expr Comma Test_Or_Star_Exprs
Testlist_Star_Expr -> Test_Or_Star_Exprs | Test_Or_Star_Exprs Comma
Yield_Testlist_Star_Assign_Exprs -> Assign_Op Yield_Expr | Assign_Op Testlist_Star_Expr | Assign_Op Yield_Expr Yield_Testlist_Star_Assign_Exprs | Assign_Op Testlist_Star_Expr Yield_Testlist_Star_Assign_Exprs
Testlist_Star_Exprs_Assign -> Testlist_Star_Expr | Testlist_Star_Expr Yield_Testlist_Star_Assign_Exprs
Del_Stmt -> Del_Keyword Exprlist
Flow_Stmt -> Break_Stmt | Continue_Stmt | Return_Stmt | Raise_Stmt | Yield_Stmt
Return_Stmt -> Return_Keyword | Return_Keyword Testlist_Endcomma
Yield_Stmt -> Yield_Expr
Raise_Stmt -> Raise_Keyword | Raise_Keyword Test | Raise_Keyword Test From_Keyword Test
Import_Stmt -> Import_name | Import_From
Import_name -> Import_Keyword Dotted_As_Names
Dots_Plus -> Dot_Or_Dots | Dot_Or_Dots Dots_Plus
Start_Dotted_Name -> Dotted_Name | Dots_Plus Dotted_Name
Import_From_Froms -> From_Keyword Start_Dotted_Name | From_Keyword Dots_Plus
Import_From_Imports -> Import_Keyword Star_Op | Import_Keyword Open_Paren Import_As_Names_Endcomma Close_Paren | Import_Keyword Import_As_Names_Endcomma
Import_From -> Import_From_Froms Import_From_Imports
Import_As_Name -> Simple_Name | Simple_Name As_Keyword Simple_Name
Dotted_As_Name -> Dotted_Name | Dotted_Name As_Keyword Simple_Name
Import_As_Names -> Import_As_Name | Import_As_Name Comma Import_As_Names_Endcomma
Import_As_Names_Endcomma -> Import_As_Names | Import_As_Name Comma
Dotted_As_Names -> Dotted_As_Name | Dotted_As_Name Comma Dotted_As_Names
Dotted_Name -> Simple_Name | Simple_Name Dot Dotted_Name
Many_Names -> Simple_Name | Simple_Name Comma Many_Names
Global_Stmt -> Global_Keyword Many_Names
Nonlocal_Stmt -> Nonlocal_Keyword Many_Names
Assert_Stmt -> Assert_Keyword Test | Assert_Keyword Test Comma Test

Aug_Assign -> += | -= | *= | @= | /= | %= | &= | |= | ^= | <<= | >>= | **= | //=
Del_Keyword -> del
Pass_Stmt -> pass
Break_Stmt -> break
Continue_Stmt -> continue
Return_Keyword -> return
Yield_Keyword -> yield
Raise_Keyword -> raise
From_Keyword -> from
Import_Keyword -> import
Dot_Or_Dots -> . | ...
As_Keyword -> as
Global_Keyword -> global
Nonlocal_Keyword -> nonlocal
Assert_Keyword -> assert
Def_Keyword -> def
Class_Keyword -> class

Compound_Stmt -> If_Stmt | While_Stmt | For_Stmt | Try_Stmt | With_Stmt | Funcdef | Classdef | Async_Stmt
Async_Stmt -> Async_Keyword Funcdef | Async_Keyword With_Stmt | Async_Keyword For_Stmt
Elif_Stmt -> Elif_Keyword Test Colon Suite | Elif_Keyword Test Colon Suite Elif_Stmt
Else_Stmt -> Else_Keyword Colon Suite
If_Stmt -> If_Keyword Test Colon Suite | If_Keyword Test Colon Suite Else_Stmt | If_Keyword Test Colon Suite Elif_Stmt | If_Keyword Test Colon Suite Elif_Stmt Else_Stmt
While_Stmt -> While_Keyword Test Colon Suite | While_Keyword Test Colon Suite Else_Stmt
For_Stmt -> For_Keyword Exprlist In_Keyword Testlist_Endcomma Colon Suite | For_Keyword Exprlist In_Keyword Testlist_Endcomma Colon Suite Else_Stmt
Finally_Stmt -> Finally_Keyword Colon Suite
Except_Stmt -> Except_Clause Colon Suite | Except_Clause Colon Suite Except_Stmt
Try_Stmt -> Try_Keyword Colon Suite Finally_Stmt | Try_Keyword Colon Suite Except_Stmt | Try_Keyword Colon Suite Except_Stmt Else_Stmt | Try_Keyword Colon Suite Except_Stmt Finally_Stmt | Try_Keyword Colon Suite Except_Stmt Else_Stmt Finally_Stmt
With_Stmt -> With_Keyword With_Items Colon Suite
With_Items -> With_Item | With_Item Comma With_Items
With_Item -> Test | Test As_Keyword Expr
Except_Clause -> Except_Keyword | Except_Keyword Test | Except_Keyword Test As_Keyword Simple_Name
Suite -> Simple_Stmt | Newline Indent Stmts_Or_Newlines Dedent

Async_Keyword -> async
Await_Keyword -> await
If_Keyword -> if
Elif_Keyword -> elif
Else_Keyword -> else
While_Keyword -> while
For_Keyword -> for
In_Keyword -> in
Finally_Keyword -> finally
Except_Keyword -> except
Try_Keyword -> try
With_Keyword -> with
Lambda_Keyword -> lambda
Indent -> INDENT
Dedent -> DEDENT
Colon -> :
Semicolon -> ;
Comma -> ,
Dot -> .
Open_Paren -> (
Close_Paren -> )
Open_Sq_Bracket -> [
Close_Sq_Bracket -> ]
Open_Curl_Bracket -> {
Close_Curl_Bracket -> }

Test -> Or_Test | Or_Test If_Keyword Or_Test Else_Keyword Test | Lambdef
Test_Nocond -> Or_Test | Lambdef_Nocond
Lambdef -> Lambda_Keyword Colon Test | Lambda_Keyword Varargslist Colon Test
Lambdef_Nocond -> Lambda_Keyword Colon Test_Nocond | Lambda_Keyword Varargslist Colon Test_Nocond
Or_Test -> And_Test | Or_Test Or_Bool_Op And_Test
And_Test -> Not_Test | And_Test And_Bool_Op Not_Test
Not_Test -> Not_Bool_Op Not_Test | Comparison
Comparison -> Expr | Comparison Comp_Op Expr
Star_Expr -> Star_Op Expr
Expr -> Xor_Expr | Expr Or_Op Xor_Expr
Xor_Expr -> And_Expr | Xor_Expr Xor_Op And_Expr
And_Expr -> Shift_Expr | And_Expr And_Op Shift_Expr
Shift_Expr -> Arith_Expr | Shift_Expr Shift_Op Arith_Expr
Arith_Expr -> Term | Arith_Expr Arith_Op Term
Term -> Factor | Term MulDiv_Op Factor
Factor -> Unary_Op Factor | Power
Power -> Atom_Expr | Atom_Expr Double_Star_Op Factor
Many_Trailers -> Trailer | Trailer Many_Trailers
Atom_Expr -> Atom | Atom Many_Trailers | Await_Keyword Atom | Await_Keyword Atom Many_Trailers
Atom -> Open_Paren Close_Paren | Open_Sq_Bracket Close_Sq_Bracket | Open_Curl_Bracket Close_Curl_Bracket | Open_Paren Yield_Expr Close_Paren | Open_Paren Testlist_Comp Close_Paren | Open_Sq_Bracket Testlist_Comp Close_Sq_Bracket | Open_Curl_Bracket Dict_Or_Set_Maker Close_Curl_Bracket | Literals
Testlist_Comp -> Test_Or_Star_Expr Comp_For | Testlist_Star_Expr
Trailer -> Open_Paren Close_Paren | Open_Paren Arglist Close_Paren | Open_Sq_Bracket Subscriptlist Close_Sq_Bracket | Dot Simple_Name
Subscripts -> Subscript | Subscript Comma Subscripts
Subscriptlist -> Subscripts | Subscripts Comma
Subscript -> Test | Colon | Test Colon | Colon Test | Colon Sliceop | Test Colon Test | Colon Test Sliceop | Test Colon Sliceop | Test Colon Test Sliceop
Sliceop -> Colon | Colon Test
Generic_Expr -> Expr | Star_Expr
Generic_Exprs -> Generic_Expr | Generic_Expr Comma Generic_Exprs
Exprlist -> Generic_Exprs | Generic_Exprs Comma
Testlist -> Test | Test Comma Testlist_Endcomma
Testlist_Endcomma -> Testlist | Test Comma
KeyVal_Or_Unpack -> Test Colon Test | Double_Star_Op Expr
Many_KeyVals_Or_Unpacks -> KeyVal_Or_Unpack | KeyVal_Or_Unpack Comma Many_KeyVals_Or_Unpacks
KeyVal_Or_Unpack_Setter -> KeyVal_Or_Unpack Comp_For | Many_KeyVals_Or_Unpacks | Many_KeyVals_Or_Unpacks Comma
Test_Or_Star_Expr_Setter -> Test_Or_Star_Expr Comp_For | Testlist_Star_Expr
Dict_Or_Set_Maker -> KeyVal_Or_Unpack_Setter | Test_Or_Star_Expr_Setter

Or_Bool_Op -> or
And_Bool_Op -> and
Not_Bool_Op -> not
Comp_Op -> < | > | == | >= | <= | <> | != | in | not_in | is | is_not
Or_Op -> OR
Xor_Op -> ^
And_Op -> &
Shift_Op -> << | >>
Arith_Op -> + | -
MulDiv_Op -> * | @ | / | % | //
Unary_Op -> + | - | ~
Literals -> NAME | NUMBER | STRING | ... | None | True | False
Simple_Name -> NAME

Classdef -> Class_Keyword Simple_Name Colon Suite | Class_Keyword Simple_Name Open_Paren Close_Paren Colon Suite | Class_Keyword Simple_Name Open_Paren Arglist Close_Paren Colon Suite

Arglist -> Arguments | Arguments Comma
Arguments -> Argument | Argument Comma Arguments
Argument -> Test | Test Comp_For | Test Assign_Op Test | Double_Star_Op Test | Star_Op Test

Comp_Iter -> Comp_For | Comp_If
Comp_For -> For_Keyword Exprlist In_Keyword Or_Test | For_Keyword Exprlist In_Keyword Or_Test Comp_Iter | Async_Keyword For_Keyword Exprlist In_Keyword Or_Test | Async_Keyword For_Keyword Exprlist In_Keyword Or_Test Comp_Iter
Comp_If -> If_Keyword Test_Nocond | If_Keyword Test_Nocond Comp_Iter

Yield_Expr -> Yield_Keyword | Yield_Keyword Yield_Arg
Yield_Arg -> From_Keyword Test | Testlist_Endcomma 
"""
val seq2ParseCFGNNTs = s2pCFGStr.parseCFG().subgrammar(PYMAP.keys.map { if (1 < it.length && it.startsWith("'") &&
  it.endsWith("'")) it.drop(1).dropLast(1) else it }.toSet()).noNonterminalStubs.freeze()

val vanillaS2PCFG = s2pCFGStr.parseCFG().noEpsilonOrNonterminalStubs.freeze()
val vanillaS2PCFGMinimized by lazy {
  vanillaS2PCFG.directSubgrammar(vanillaS2PCFG.symbols.filter { (symbolCounts[it] ?: 0) < 3 })
}

// Taken from seq2parse's Python grammar
val seq2parsePythonCFG: CFG by lazy {
  s2pCFGStr.parseCFG(normalize = false)
  /** TODO: remove this pain in the future, canonicalize [normalForm]s */
  .run {
    mutableListOf<CFG>().let { rewrites ->
      expandOr().freeze()
        .also { rewrites.add(it) }
        /** [originalForm] */
        .eliminateParametricityFromLHS()
        .also { rewrites.add(it) }
        /** [nonparametricForm] */
        .generateNonterminalStubs()
        .transformIntoCNF()
        .also { cnf -> rewriteHistory.put(cnf, rewrites) }
    }
  }.freeze().also {
    measureTimeMillis { println("UR:" + it.originalForm.unitReachability.size) }
      .also { println("Computed unit reachability in ${it}ms") }
  }
}

fun extractErrProbs(): List<Pair<Σᐩ, Int>> =
  preprocessStackOverflow().take(3000).asStream().parallel().flatMap { (b, _, m) ->
    val patch = extractPatch(b.lexToStrTypesAsPython(), m.lexToStrTypesAsPython())
    val changes = patch.changedIndices()
    changes.map { patch[it].new.let { it.ifEmpty { "ε" } } }
      .also { println(it) }
      .stream()
  }.collect(Collectors.groupingBy { it }).mapValues { it.value.size }.toList()

fun readContents(
  filename: String = "parse_errors.json",
  file: File = File(File("").absolutePath +
    "/src/main/resources/datasets/python/stack_overflow/$filename")
): Sequence<String> {
  val contentPattern = Pattern.compile("\"content\"\\s*:\\s*\"(.*?)\",\\s*\"length\"")

  return sequence {
    file.bufferedReader().use { reader ->
      val line = reader.readLine() ?: return@sequence

      val matcher = contentPattern.matcher(line)
      yieldAll(generateSequence {
        if (matcher.find()) {
          val json = "{\"content\":\"${matcher.group(1)}\"}"
          val parsedObject = Klaxon().parseJsonObject(json.reader())
          parsedObject.string("content")
        } else null
      })
    }
  }
}