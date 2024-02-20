package edu.mcgill.cstk.experiments.repair

import ConcurrentRankedProbabilisticSet
import NUM_CORES
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.types.to
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.absoluteValue
import kotlin.random.Random
import kotlin.streams.*
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds
import kotlin.to

/*
./gradlew pythonBarHillelRepair
 */
fun main() {
//  MAX_TOKENS = 20
//  MAX_RADIUS = 2
  evaluateBarHillelRepair()
//  evaluateSeq2ParseRepair()
}

fun readPCFGMap() =
  File(File("").absolutePath + "/src/main/resources/models/pcfg_BIFI.csv").readText()
  .lines().map { it.split(" ::: ") }.associate { Pair(it[0].split(" ").let { it[0] to it[1] to it[2] }, it[1].toInt()) }

fun evaluateBarHillelRepair() {
  // Perfect recall on first 20 repairs takes ~7 minutes on a 2019 MacBook Pro
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val s2pg = vanillaS2PCFG
  val parikhMap = s2pg.parikhMap
  val pcfgMap = readPCFGMap()
//  assert(validLexedPythonStatements.lines().all { it in s2pg.language })

  val dataset = balancedSmallRepairs.toList() // naturallySmallRepairs //pairwiseUniformAll
  println("Running Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  dataset.first().second.let { P_BIFI_PY150.score("BOS NEWLINE $it EOS".tokenizeByWhitespace()) }

  val latestCommitMessage = lastGitMessage().replace(" ", "_")
  val positiveHeader = "length, lev_dist, sample_ms, total_ms, " +
      "total_samples, lev_ball_arcs, productions, rank, edit1, edit2, edit3\n"
  val negativeHeader = "length, lev_dist, samples, productions, edit1, edit2, edit3\n"
  val positive = try { File("bar_hillel_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/bar_hillel_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
    .also { println("Writing positive CSV to: ${it.absolutePath}") }
  val negative = try { File("bar_hillel_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/bar_hillel_results_negative_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
    .also { println("Writing negative CSV to: ${it.absolutePath}") }
  println()

  dataset.shuffled(Random(1)).forEach { (invalid, valid) ->
    val allTime = TimeSource.Monotonic.markNow()
    val toRepair = "$invalid NEWLINE".tokenizeByWhitespace()
    val humanRepair = "$valid NEWLINE".tokenizeByWhitespace()
    val target = humanRepair.joinToString(" ")
    val source = toRepair.joinToString(" ")
    val levAlign = levenshteinAlign(toRepair, humanRepair)
    val levDist = levAlign.patchSize()

    var levBallSize = 1
    val humanRepairANSI = levenshteinAlign(toRepair, humanRepair).paintANSIColors()
    val intGram = try {
      s2pg.jvmIntersectLevFSA(
        makeLevFSA(toRepair, levDist).also { levBallSize = it.Q.size },
        parikhMap = parikhMap
      ).also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
    } catch (e: Exception) { println("Intersection error: ${e.stackTraceToString()}"); null }

    println("Constructed LEV($levDist, ${toRepair.size}, $levBallSize) " +
      "∩ CFG grammar with ${intGram?.size ?: 0} productions in ${allTime.elapsedNow()}")

    try {
      if (intGram == null) throw Exception("Exception while building grammar!")
      else if (100_000 < intGram.size) throw Exception("Int grammar was still too large!")
      else if (humanRepair !in intGram.language) throw Exception("Human repair is unrecognizable!")
      else println("Human repair is recognized by LEV ∩ CFG grammar")
    } catch (e: Exception) {
      println("Encountered error (${e.message}): $humanRepairANSI\n")
      allRate.error++; levRates.getOrPut(levDist) { LBHMetrics() }.error++
      println(allRate.toString())
      negative.appendText("${toRepair.size}, $levDist, 0, " +
        "${levBallSize}, ${intGram?.size ?: 0}, ${levAlign.summarize()}\n")
      return@forEach
    }

    allRate.total++; levRates.getOrPut(levDist) { LBHMetrics() }.total++
    println("Ground truth repair: $humanRepairANSI")
    val clock = TimeSource.Monotonic.markNow()
    val totalSamples = AtomicInteger(0)
    var matchFound = false
    val timeout = (TIMEOUT_MS / 1000).seconds
//    val results = mutableListOf<Σᐩ>()
    var elapsed = clock.elapsedNow().inWholeMilliseconds
    val results = ConcurrentRankedProbabilisticSet<Σᐩ>(10_000)
    val sampler = if (intGram.size < 10_000) {
      println("Small grammar, sampling without replacement from grammar...")
      intGram.sampleDirectlyWOR(stoppingCriterion = { clock.elapsedNow() < timeout })
    } else {
      println("Large grammar, sampling with replacement using PCFG...")
      intGram.sampleWithPCFG(pcfgMap, stoppingCriterion = { clock.elapsedNow() < timeout })
//        .map { println(levenshteinAlign(source, it).paintANSIColors()); it }
    }

    sampler.distinct().forEach {
      totalSamples.incrementAndGet()
      if (it == target) { matchFound = true; elapsed = clock.elapsedNow().inWholeMilliseconds }
      val repairDist = levenshtein(it.tokenizeByWhitespace(), humanRepair)
      val levModifier = when (repairDist) { 1 -> 0.58; 2 -> 0.34; else -> 0.08 }
      results.add(it, P_BIFI_PY150.score(it.mapToBIFIFmt()) * levModifier)
    }

    if (!matchFound) {
      println("Drew $totalSamples samples in $timeout," +
        " ${intGram.size} prods, length-$levDist human repair not found")
      negative.appendText("${toRepair.size}, $levDist, $totalSamples, " +
        "${levBallSize}, ${intGram.size}, ${levAlign.summarize()}\n")
    } else {
      val allElapsed = allTime.elapsedNow().inWholeMilliseconds
      val rankedResults = results.mostLikely.entries.map { it.value }
//        results.parallelStream().map {
//          val levDist = levenshtein(it.tokenizeByWhitespace(), humanRepair)
//          val levModifier = when (levDist) { 1 -> 0.58; 2 -> 0.34; else -> 0.08 }
//          it to P_BIFI.score(it.mapToBIFIFmt()) * levModifier
//        }.sorted(Comparator.comparingDouble { it.second }).map { it.first }.toList()
      // First sort by levenshtein distance, then by perplexity
//          .map { it to levenshtein(source, it) to P_BIFI.score(it.mapToBIFIFmt()) }
//          .sortedWith(compareBy({ it.second }, { it.third })).map { it.first }

      allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
      val indexOfTarget = rankedResults.indexOf(target)
        .also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
      println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
      println("Found length-$levDist repair in $elapsed ms, $allElapsed ms," +
        " $totalSamples samples, ${intGram.size} prods, $indexOfTarget rank")//, rank: ${rankedResults.indexOf(target) + 1} / ${rankedResults.size}")
      allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
      println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
      allTimeByLevDist[levDist] = allTimeByLevDist[levDist]!! + allElapsed
      println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
      samplesBeforeMatchByLevDist[levDist] = samplesBeforeMatchByLevDist[levDist]!! + totalSamples.get()
      println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
      positive.appendText("${toRepair.size}, $levDist, $elapsed, $allElapsed, " +
        "$totalSamples, ${levBallSize}, ${intGram.size}, $indexOfTarget, ${levAlign.summarize()}\n")
    }

    println()
  }
}

@JvmName("summarizeLBHMetrics")
fun Map<Int, LBHMetrics>.summarize() =
  entries.sortedBy { it.key }.joinToString("\n") { (k, v) -> "Lev($k): $v" }

data class LBHMetrics(var top1: Int = 0, var recall: Int = 0, var total: Int = 0, var error: Int = 0) {
  override fun toString() =
    "Top-1/rec/pos/total: $top1 / $recall / $total / ${total + error}, " +
      "errors: $error, P@1: ${top1.toDouble() / (total + error)}"
}

val naturallySmallRepairs: Sequence<Π2A<Σᐩ>> by lazy {
  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs.txt"
  val file = File(File("").absolutePath + path).readText()
  file.lines().asSequence().windowed(2, 2).map { it[0] to it[1] }
    .filter { (a, b) ->
      val broke = a.tokenizeByWhitespace()
      val fixed = b.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      broke.size in 3..MAX_TOKENS &&
        fixed.size in 3..MAX_TOKENS &&
        levDist <= MAX_RADIUS
    }
}

// Balanced number of repairs for each levenshtein distance
val balancedSmallRepairs: Sequence<Π2A<Σᐩ>> by lazy {
  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs.txt"
  val file = File(File("").absolutePath + path).readText()
  file.lines().asSequence().windowed(2, 2).map { it[0] to it[1] }
    .map { (a, b) ->
      val broke = a.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      a to b to levDist
    }.filter { (broke, fixed, levDist) ->
      broke.tokenizeByWhitespace().size in 3..MAX_TOKENS &&
        fixed.tokenizeByWhitespace().size in 3..MAX_TOKENS &&
        levDist <= MAX_RADIUS
    }
   .groupBy { it.third }.let { map ->
      val minSize = map.values.minOf { it.size }
      println("Size of smallest group: $minSize")
      map.mapValues { (_, v) -> v.shuffled().take(minSize) }
    }
    .values.asSequence().flatten()
    .map { it.first to it.second }
    .distinct().shuffled()
}

fun Σᐩ.mapToBIFIFmt() =
  "BOS NEWLINE $this EOS".tokenizeByWhitespace()

// Seq2Parse results:
// Lev(*): 0.29235695391897537
// Lev(1): Top-1/total: 1687 / 4219 = 0.3998577862052619
// Lev(2): Top-1/total: 362 / 2322 = 0.15590008613264428
// Lev(3): Top-1/total: 51 / 642 = 0.0794392523364486

fun evaluateSeq2ParseRepair() {
  val P_1ByLevDist = mutableMapOf<Int, S2PMetrics>()
  preprocessStackOverflowQuickly(lengthBounds = 0..MAX_TOKENS).forEach { (invalid, _, valid) ->
    val toRepair = invalid.mapToUnquotedPythonTokens().tokenizeByWhitespace()
    val humanRepair = valid.mapToUnquotedPythonTokens().tokenizeByWhitespace()
    val levDist = levenshtein(toRepair, humanRepair)
    val seq2parseFix = seq2parseFix(invalid)
    val s2pfTokens = seq2parseFix.mapToUnquotedPythonTokens().tokenizeByWhitespace()
    P_1ByLevDist.getOrPut(levDist) { S2PMetrics() }.total++
    if (s2pfTokens == humanRepair) { P_1ByLevDist.getOrPut(levDist) { S2PMetrics() }.top1++ }

    println("Ground truth : ${levenshteinAlign(toRepair, humanRepair).paintANSIColors()}")
    println("Seq2Parse fix: ${levenshteinAlign(toRepair, s2pfTokens).paintANSIColors()}")
    println(P_1ByLevDist.summarize())
    println()
  }
}

fun preprocessStackOverflowQuickly(
  maxPatchSize: Int = MAX_RADIUS,
  lengthBounds: IntRange = 0..Int.MAX_VALUE,
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
  fixedSnippets: Sequence<String> = readContents("parse_fixes.json"),
) =
  brokeSnippets.zip(fixedSnippets).asStream().parallel()
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
    }.distinct()
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

@JvmName("summarizeS2PMetrics")
fun Map<Int, S2PMetrics>.summarize() =
  "Lev(*): ${values.sumOf { it.top1 }.toDouble() / values.sumOf { it.total }}\n" +
  entries.sortedBy { it.key }.joinToString("\n") { (k, v) -> "Lev($k): $v" }

data class S2PMetrics(var top1: Int = 0, var total: Int = 0) {
  override fun toString() =
    "Top-1/total: $top1 / $total = ${top1.toDouble() / total}"
}