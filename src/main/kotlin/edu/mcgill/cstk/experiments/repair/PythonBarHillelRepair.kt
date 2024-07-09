package edu.mcgill.cstk.experiments.repair

import ConcurrentRankedProbabilisticSet
import NUM_CORES
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.types.to
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.*
import kotlin.streams.*
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds
import kotlin.to

/*
./gradlew pythonBarHillelRepair
 */
fun main() {
//  MAX_UNIQUE = 1_000
  TIMEOUT_MS = 30_000
  MIN_TOKENS = 3
  MAX_TOKENS = 80
  MAX_RADIUS = 3
  CFG_THRESH = 10_000
  evaluateBarHillelRepairOnStackOverflow()
//  evaluateSeq2ParseRepair()
//  evaluateBIFIRepair()
//  measureLevenshteinBlanketSize()
}

val LEN_BUCKET_INTERVAL = 10

fun readPCFG3(): Map<Π3A<Σᐩ>, Int> =
  File(File("").absolutePath + "/src/main/resources/models/pcfg3_BIFI.csv").readText()
  .lines().map { it.split(" ::: ") }.associate { Pair(it[0].split(" ").let { it[0] to it[1] to it[2] }, it[1].toInt()) }

fun readPCFG5(s2pg: CFG): Map<Int, Int> =
  File(File("").absolutePath + "/src/main/resources/models/pcfg5_BIFI.csv").readText()
    .lines().map { it.split(" ::: ") }
    .associate { Pair(it[0].split(" ")
      .map { if (it.endsWith('*') && it.length > 1) (31 * s2pg.ntMap[it.dropLast(1)]!!) else s2pg.ntMap[it] ?: Int.MAX_VALUE }
      /** See [Tree.quintuples] */
      .let { hash(it[0], it[1], it[2], it[3], it[4]) }, it[1].toInt()) }

fun evaluateBarHillelRepairOnStackOverflow() {
  val dataset = sizeAndDistBalancedRepairsUnminimized.toList()//corruptedBIFIGoodCode//sizeAndDistBalancedRepairsUnminimized.toList()
   // timeoutCases // corruptedBIFIGoodCode // balancedSmallRepairsUnminimized.toList() // naturallySmallRepairs //pairwiseUniformAll
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val s2pg = vanillaS2PCFG
  val lbc = s2pg.lengthBoundsCache
  val parikhMap = s2pg.parikhMap
  val pcfgMap = readPCFG3()
  val pcfgNorm = s2pg.nonterminals.associateWith { nt -> pcfgMap.filterKeys { it.first == nt }.values.sum() }

  println("Running Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  println("Sampling timeout: $TIMEOUT_MS ms, max tokens: $MAX_TOKENS, " +
      "max radius: $MAX_RADIUS, max unique: $MAX_UNIQUE, CFG threshold: $CFG_THRESH")
  dataset.first().second.let { P_BIFI_PY150.score(it.tokenizeByWhitespace()) }

  val latestCommitMessage = lastGitMessage().replace(Regex("[^A-Za-z0-9]"), "_")
//    .replace(" ", "_").replace("/", "_")
  val positiveHeader = "length, lev_dist, sample_ms, total_ms, " +
      "total_samples, lev_ball_arcs, productions, lang_size, dfa_states, dfa_transitions, rank, edit1, edit2, edit3\n"
  val negativeHeader = "length, lev_dist, samples, lev_states, productions, lang_size, dfa_states, dfa_transitions, edit1, edit2, edit3\n"
  val positive = try { File("bar_hillel_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/bar_hillel_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
    .also { println("Writing positive CSV to: ${it.absolutePath}") }
  val negative = try { File("bar_hillel_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/bar_hillel_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
    .also { println("Writing negative CSV to: ${it.absolutePath}") }
  println()

  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_AllByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val editLocationsByLenAndDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  dataset.forEach { (invalid, valid) ->
    val allTime = TimeSource.Monotonic.markNow()
    val toRepair = invalid.tokenizeByWhitespace()
    val humanRepair = valid.tokenizeByWhitespace()
    val target = humanRepair.joinToString(" ")
    val levAlign = levenshteinAlign(toRepair, humanRepair)
    val levDist = levAlign.patchSize()
    val lenBucket = (toRepair.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    var levBallSize = 1
    val humanRepairANSI = levenshteinAlign(toRepair, humanRepair).paintANSIColors()
    println("Source: ${toRepair.joinToString(" ")}")
    println("Repair: $humanRepairANSI")
    val intGram = try {
      s2pg.jvmIntersectLevFSAP(
        fsa = makeLevFSA(toRepair, levDist).also { levBallSize = it.Q.size },
        parikhMap = parikhMap,
        lbc = lbc
      ).also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
    } catch (e: Exception) { println("$humanRepairANSI\nIntersection error: ${e.stackTraceToString()}"); null }
      ?.freeze()

    println("Constructed LEV($levDist, ${toRepair.size}, $levBallSize) " +
      "∩ CFG grammar with ${intGram?.size ?: 0} productions in ${allTime.elapsedNow()}")

    try {
      if (intGram == null) throw Exception("Exception while building grammar!")
      else if (30_000 < intGram.size) throw Exception("Int grammar was still too large!")
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
    val pTree = measureTimedValue { intGram.toPTree(origCFG = s2pg) }
      .also { println("Constructed PTree in ${it.duration}") }.value
    val langSize = pTree.totalTreesStr
    val clock = TimeSource.Monotonic.markNow()
    val totalSamples = AtomicInteger(0)
    var matchFound = false
    val timeout = (TIMEOUT_MS / 1000).seconds
    var elapsed = clock.elapsedNow().inWholeMilliseconds

    val results = ConcurrentRankedProbabilisticSet<Σᐩ>(MAX_UNIQUE)
//      if (intGram.size < CFG_THRESH) {
//        println("Small grammar, sampling without replacement...")
        pTree
//          .sampleDirectlyWOR(stoppingCriterion = { clock.elapsedNow() < timeout })
          .sampleDirectlyWORAndScore(stoppingCriterion = { clock.elapsedNow() < timeout }, pcfgMap = pcfgMap, pcfgNorm = pcfgNorm)
//      } else {
//        println("Large grammar, sampling with replacement using PCFG...")
//        pTree.sampleWithPCFG(pcfgMap, stoppingCriterion = { clock.elapsedNow() < timeout })
  //        .map { println(levenshteinAlign(source, it).paintANSIColors()); it }
//      }
          .map {
            totalSamples.incrementAndGet()
            if (it.first == target) { matchFound = true; elapsed = clock.elapsedNow().inWholeMilliseconds }
//            results.add(it.first, P_BIFI_PY150.score(it.first.tokenizeByWhitespace()))
            // PCFG likelihood reranker
            results.add(it.first, -it.second + P_BIFI_PY150.score(it.first.tokenizeByWhitespace()))
          }
          .toList()

    println("Found $totalSamples samples in ${clock.elapsedNow()}")

    val rankedResults = results.mostLikely.entries.map { it.value }

    println("Ranked ${results.size} samples in ${clock.elapsedNow()}")

//    val dfa = pTree.toDFA(true)!!
//
//    val dfaRecognized = dfa.run(pTree.termDict.encode(humanRepair))
//    println("∩-DFA ${if (dfaRecognized) "accepted" else "rejected"} human repair!")
//
//    val rankedResults = dfa.decodeDFA(
//      mc = P_BIFI_PY150,
//      timeout = timeout,
//      dec = pTree.termDict,
//      parallelize = false,
//      callback = {
//        totalSamples.incrementAndGet()
//        if (it == target) {
//          matchFound = true
//          println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
//          elapsed = clock.elapsedNow().inWholeMilliseconds
//        }
//      }
//    )

    val indexOfTarget = rankedResults.indexOf(target).also {
      if (it == 0) P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      if (matchFound) P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
    }

    rankedResults.firstOrNull()?.tokenizeByWhitespace()
      ?.let { println("Top1 scoring repair: ${levenshteinAlign(toRepair, it).paintANSIColors()}") }

    if (indexOfTarget < 0) {
      println("Drew $totalSamples samples in $timeout with ${intGram.size} prods, " +
//        "${dfa.states.size} states, ${dfa.numberOfTransitions} transitions, " +
          "length-$levDist human repair not found")
      negative.appendText(
        "${toRepair.size}, $levDist, $totalSamples, ${levBallSize}, " +
          "${intGram.size}, $langSize, " +
//          "${dfa.states.size}, ${dfa.numberOfTransitions}, " +
          "${levAlign.summarize()}\n"
      )
    } else {
      val allElapsed = allTime.elapsedNow().inWholeMilliseconds
//        results.parallelStream().map {
//          val levDist = levenshtein(it.tokenizeByWhitespace(), humanRepair)
//          val levModifier = when (levDist) { 1 -> 0.58; 2 -> 0.34; else -> 0.08 }
//          it to P_BIFI.score(it.mapToBIFIFmt()) * levModifier
//        }.sorted(Comparator.comparingDouble { it.second }).map { it.first }.toList()
      // First sort by levenshtein distance, then by perplexity
//          .map { it to levenshtein(source, it) to P_BIFI.score(it.mapToBIFIFmt()) }
//          .sortedWith(compareBy({ it.second }, { it.third })).map { it.first }

      allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
      indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
      println("Found length-$levDist repair in $elapsed ms, $allElapsed ms," +
        " $totalSamples samples, ${intGram.size} prods, $langSize trees, $indexOfTarget rank")//, rank: ${rankedResults.indexOf(target) + 1} / ${rankedResults.size}")
      allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
//      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
      sampleTimeByLevDist[levDist] = (sampleTimeByLevDist[levDist] ?: 0.0) + elapsed
      println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
      allTimeByLevDist[levDist] = (allTimeByLevDist[levDist] ?: 0.0) + allElapsed
      println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
      samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + totalSamples.get()
      println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
      positive.appendText("${toRepair.size}, $levDist, $elapsed, $allElapsed, " +
        "$totalSamples, ${levBallSize}, ${intGram.size}, $langSize, " +
//          "${dfa.states.size}, ${dfa.numberOfTransitions}, " +
          "$indexOfTarget, ${levAlign.summarize()}\n")
    }

    println()
    println("Precision@1\n===========")
    println(P_1ByLevDist.summarizeLenAndDist())
    println("Precision@All\n=============")
    println(P_AllByLevDist.summarizeLenAndDist())
    println()

// Stability statistics collection (but it costs time to compute)
//    if (rankedResults.isNotEmpty()) {
//      editLocationsByLenAndDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
//      var levBlanket = rankedResults.first().tokenizeByWhitespace()
//      rankedResults.shuffled().parallelStream().forEach {
//        levBlanket = updateLevenshteinBlanket(levBlanket, it.tokenizeByWhitespace())
//      }
//
//      val stability = ((levBlanket.count { it != "_" }.toDouble() / toRepair.size) * 100).roundToInt()
//      editLocationsByLenAndDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1 += stability
//      println("Stability profile\n=============")
//      println(editLocationsByLenAndDist.summarizeLenAndDist())
//    }
  }
}

@JvmName("summarizeLBHMetrics")
fun Map<Int, LBHMetrics>.summarize() =
  entries.sortedBy { it.key }.joinToString("\n") { (k, v) -> "Lev($k): $v" }

data class LBHMetrics(var top1: Int = 0, var recall: Int = 0, var total: Int = 0, var error: Int = 0) {
  override fun toString() =
    "Top-1/rec/pos/total: $top1 / $recall / $total / ${total + error}, " +
      "errors: $error, P@1: ${top1.toDouble() / (total + error)}, P@All: ${recall.toDouble() / (total + error)}"
}

val naturallySmallRepairs: Sequence<Π2A<Σᐩ>> by lazy {
  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs.txt"
  val file = File(File("").absolutePath + path).readText()
  file.lines().asSequence().windowed(2, 2).map { it[0] to it[1] }
    .filter { (a, b) ->
      val broke = a.tokenizeByWhitespace()
      val fixed = b.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      broke.size in MIN_TOKENS..MAX_TOKENS &&
        fixed.size in MIN_TOKENS..MAX_TOKENS &&
        levDist <= MAX_RADIUS
    }
}

// Balanced number of repairs for each levenshtein distance
val levBalancedSmallRepairs: Sequence<Π2A<Σᐩ>> by lazy {
  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs.txt"
  val file = File(File("").absolutePath + path).readText()
  file.lines().asSequence().windowed(2, 2).map { it[0] to it[1] }
    .map { (a, b) ->
      val broke = a.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      a to b to levDist
    }.filter { (broke, fixed, levDist) ->
      broke.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
        fixed.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
        levDist <= MAX_RADIUS
    }
   .groupBy { it.third }.let { map ->
      val minSize = map.values.minOf { it.size }
      println("Rebalancing dataset, size of smallest bucket: $minSize")
      map.mapValues { (_, v) -> v.shuffled().take(minSize) }
    }
    .values.asSequence().flatten()
    .map { it.first to it.second }
    .distinct().shuffled()
}

val timeoutCases = listOf(
  "{ STRING : ... ... ... ... , STRING : { STRING : STRING , STRING : [ [ [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] ] ] } , STRING : { STRING : NUMBER , STRING : NUMBER } } NEWLINE" to
    "{ STRING : ... , STRING : { STRING : STRING , STRING : [ [ [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] , [ - NUMBER , NUMBER ] ] ] } , STRING : { STRING : NUMBER , STRING : NUMBER } } NEWLINE",
  "NAME = [ ( STRING , STRING ) , ( STRING , STRING ) , ( STRING , STRING ) , ( STRING , NAME ) , , ( NAME , NAME ) , , ( NAME , STRING ) , , ( NAME , NAME ) ]" to
    "NAME = [ ( STRING , STRING ) , ( STRING , STRING ) , ( STRING , STRING ) , ( STRING , NAME ) , ( NAME , NAME ) , ( NAME , STRING ) , ( NAME , NAME ) ]",
  "from NAME import NAME NEWLINE NAME = NAME ( NAME ) NEWLINE NAME = NAME . NAME . NAME . NAME . NAME . NAME ( ) NEWLINE NAME ( NAME ) NEWLINE >> STRING NEWLINE" to
    "from NAME import NAME NEWLINE NAME = NAME ( NAME ) NEWLINE NAME = NAME . NAME . NAME . NAME . NAME . NAME ( ) NEWLINE NAME ( NAME ) NEWLINE",
  "NAME : NEWLINE NAME = STRING NEWLINE NAME = NAME . NAME ( STRING ) NEWLINE NAME = NAME . NAME ( STRING , NAME ) NEWLINE" to
    "NAME = STRING NEWLINE NAME = NAME . NAME ( STRING ) NEWLINE NAME = NAME . NAME ( STRING , NAME ) NEWLINE"
)

val paperExamples = listOf("NAME = NAME . NAME ( NUMBER : , NUMBER : )" to "NAME = NAME . NAME [ NUMBER : , NUMBER : ]")

val sizeAndDistBalancedRepairsUnminimized: Sequence<Π2A<Σᐩ>> by lazy {
  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized.txt"
  val file = File(File("").absolutePath + path).readText()
  file.lines().asSequence().windowed(2, 2).map { it[0] to it[1] }
    .asStream().parallel()
    .map { (a, b) -> a.addNewLineIfMissing() to b.addNewLineIfMissing() }
    .map { (a, b) ->
      val broke = a.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      a to b to (broke.size / 10) * 10 to levDist
    }.filter { (broke, fixed, size, levDist) ->
      broke.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
          fixed.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
        levDist <= MAX_RADIUS
    }.toList()
    .groupBy { it.π3 to it.π4 }.let { map ->
      val minSize = map.entries.minBy { it.value.size }
      println("Size of smallest group: ${minSize.key}, ${minSize.value.size}")
      map.mapValues { (_, v) -> v.shuffled().take(100) }
    }
    .values.asSequence().flatten()
    .map { it.π1 to it.π2 }
    .distinct().shuffled()
}

val corruptedBIFIGoodCode by lazy {
  readBIFIContents()
    .map { it.mapToUnquotedPythonTokens().addNewLineIfMissing() }
    .filter {
      it.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
        it in vanillaS2PCFG.language
    }
    .flatMap { goodCode ->
      goodCode.nautralPythonCorruptions().distinct().filter {
        levenshtein(goodCode, it) <= MAX_RADIUS &&
            it !in vanillaS2PCFG.language
      }.take(10).map { it to goodCode }
    }.rebalancePrelexedOnlineByLenAndDist()
}

val balancedSmallRepairsUnminimized: Sequence<Π2A<Σᐩ>> by lazy {
  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized.txt"
  val file = File(File("").absolutePath + path).readText()
  file.lines().asSequence().windowed(2, 2).map { it[0] to it[1] }
    .map { (a, b) ->
      val broke = a.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      a to b to levDist
    }.filter { (broke, fixed, levDist) ->
      broke.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
          fixed.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
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

// Seq2Parse results:
// Lev(*): 0.29235695391897537
// Lev(1): Top-1/total: 1687 / 4219 = 0.3998577862052619
// Lev(2): Top-1/total: 362 / 2322 = 0.15590008613264428
// Lev(3): Top-1/total: 51 / 642 = 0.0794392523364486

fun evaluateSeq2ParseRepair() {
  MAX_TOKENS = 80
  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  preprocessStackOverflow(lengthBounds = 0..MAX_TOKENS)
    .rebalanceOnlineByLenAndDist()
    .forEach { (invalid, _, valid) ->
    val toRepair = invalid.mapToUnquotedPythonTokens().tokenizeByWhitespace()
    val humanRepair = valid.mapToUnquotedPythonTokens().tokenizeByWhitespace()
    val levDist = levenshtein(toRepair, humanRepair)
    val seq2parseFix = seq2parseFix(invalid)
    val s2pfTokens = seq2parseFix.mapToUnquotedPythonTokens().tokenizeByWhitespace()
    val length = (toRepair.size / 10) * 10

    P_1ByLevDist.getOrPut(length to levDist) { S2PMetrics() }.total++
    if (s2pfTokens == humanRepair) { P_1ByLevDist.getOrPut(length to levDist) { S2PMetrics() }.top1++ }

    println("Ground truth : ${levenshteinAlign(toRepair, humanRepair).paintANSIColors()}")
    println("Seq2Parse fix: ${levenshteinAlign(toRepair, s2pfTokens).paintANSIColors()}")
    println(P_1ByLevDist.summarizeLenAndDist())
    println()
  }
}

fun String.mapToBIFITokens(
  origToks: List<String> = tokenizeAsPython(),
  nameless: List<String> = mapToUnquotedPythonTokens().tokenizeByWhitespace()
) =
  nameless.mapIndexed { i, it ->
    when (it) {
      "STRING" -> "<STRING>"
      "NEWLINE" -> "<NEWLINE>"
      "INDENT" -> "<INDENT>"
      "DEDENT" -> "<DEDENT>"
      else -> origToks[i]
    }
  }.joinToString(" ")

fun Sequence<Π2A<Σᐩ>>.rebalancePrelexedOnlineByLenAndDist() =
  chunked(1000).map {
    it.map {
      it.π1 to it.π2 to
          it.π1.tokenizeByWhitespace().size to
            levenshtein(it.π1, it.π2)
    }.groupBy { it.π3 to it.π4 }.let { map ->
      val minSize = map.values.minOf { it.size }
      map.mapValues { (_, v) -> v.shuffled().take(minSize) }
    }.values.asSequence().flatten().map { it.π1 to it.π2 }
  }.flatten()

fun Sequence<Π3A<Σᐩ>>.rebalanceOnlineByLenAndDist() =
  chunked(1000).map {
    it.map {
      it.π1 to it.π2 to it.π3 to
        it.π1.mapToUnquotedPythonTokens().tokenizeByWhitespace().size to
        levenshtein(it.π1.mapToUnquotedPythonTokens(), it.π2.mapToUnquotedPythonTokens())
    }.groupBy { it.π4 to it.π5 }.let { map ->
      val minSize = map.values.minOf { it.size }
      println("Size of smallest group: $minSize")
      map.mapValues { (_, v) -> v.shuffled().take(minSize) }
    }.values.asSequence().flatten().map { it.π1 to it.π2 to it.π3 }
  }.flatten()

fun evaluateBIFIRepair() {
  MAX_TOKENS = 80
  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  preprocessStackOverflow(lengthBounds = 0..MAX_TOKENS)
    .rebalanceOnlineByLenAndDist()
    .forEach { (invalid, _, valid) ->
    val toRepair = bifiTokenize(invalid.mapToBIFITokens())
    // Incremental buckets of length 10
    val length = (toRepair.tokenizeByWhitespace().size / 10) * 10
    val humanRepair = bifiTokenize(valid.mapToBIFITokens())
    val levDist = levenshtein(toRepair, humanRepair)

    println("BROKEN: $toRepair")
    val bifiFixes = measureTimedValue { bifiFix(toRepair, MAX_UNIQUE) }
      .also { println("BIFI-$MAX_UNIQUE took: ${it.duration}") }.value

    bifiFixes.take(10).forEachIndexed { i, it ->
      println("BIFI-$i: ${levenshteinAlign(toRepair, it).paintANSIColors()}")
    }

    println("GROUND: ${levenshteinAlign(toRepair, humanRepair).paintANSIColors()}")

    P_1ByLevDist.getOrPut(length to levDist) { S2PMetrics() }.total++

    if (humanRepair in bifiFixes) { P_1ByLevDist.getOrPut(length to levDist) { S2PMetrics() }.top1++ }
    println(P_1ByLevDist.summarizeLenAndDist())
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

// How many unique edit locations are there for each length and Levenshtein distance?
fun measureLevenshteinBlanketSize() {
  val gram = vanillaS2PCFG
  val lbc = gram.lengthBoundsCache
  val parikhMap = gram.parikhMap
  val timeout = 10.seconds
  val pcfgMap = readPCFG5(gram)
  val editLocationsByLenAndDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  sizeAndDistBalancedRepairsUnminimized.toList()
    .map { it.first.addNewLineIfMissing() to it.second.addNewLineIfMissing() }
    .forEach {  (broke, fixed) ->
      val allTime = TimeSource.Monotonic.markNow()
      val brokeTokens = broke.tokenizeByWhitespace()
      val fixedTokens = fixed.tokenizeByWhitespace()
      val levDist = levenshtein(brokeTokens, fixedTokens)
      val lenBucket = (brokeTokens.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL

      editLocationsByLenAndDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

      val levBall = makeLevFSA(brokeTokens, levDist)
      val intGram = try {
        gram.jvmIntersectLevFSAP(
          levBall,
          parikhMap = parikhMap,
          lbc = lbc
        ).also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
      } catch (e: Exception) { null }?.freeze()

      try {
        if (intGram == null) throw Exception("Exception while building grammar!")
        else if (30_000 < intGram.size) throw Exception("Int grammar was still too large!")
        else if (fixedTokens !in intGram.language) throw Exception("Human repair is unrecognizable!")
        else println("Human repair is recognized by LEV ∩ CFG grammar")
      } catch (e: Exception) { return@forEach }

      println("Constructed LEV($levDist, ${brokeTokens.size}, ${levBall.Q.size}) " +
          "∩ CFG grammar with ${intGram.size} productions in ${allTime.elapsedNow()}")

      val pTree = intGram.toPTree()
      val clock = TimeSource.Monotonic.markNow()

      val sampler =
        if (intGram.size < CFG_THRESH) {
          println("Small grammar, sampling without replacement...")
          pTree.sampleDirectlyWOR(stoppingCriterion = { clock.elapsedNow() < timeout })
        } else {
          println("Large grammar, sampling with replacement using PCFG...")
          pTree.sampleWithPCFG(pcfgMap, stoppingCriterion = { clock.elapsedNow() < timeout })
          //        .map { println(levenshteinAlign(source, it).paintANSIColors()); it }
        }

      var levBlanket = brokeTokens
      sampler.distinct().limit(20000).forEach {
        levBlanket = updateLevenshteinBlanket(levBlanket, it.tokenizeByWhitespace())
      }

      val totalHoles = levBlanket.count { it == "_" }
      editLocationsByLenAndDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1 += totalHoles
      println(editLocationsByLenAndDist.summarizeLenAndDist())
  }
}

@JvmName("summarizeS2PMetrics")
fun Map<Int, S2PMetrics>.summarize() =
  "Lev(*): ${values.sumOf { it.top1 }.toDouble() / values.sumOf { it.total }}\n" +
  entries.sortedBy { it.key }.joinToString("\n") { (k, v) -> "Lev($k): $v" }

fun Map<Pair<Int, Int>, S2PMetrics>.summarizeLenAndDist() =
  // By distribution of lengths
  entries.groupBy({ it.key.first }, { it.value })
    .mapValues { (_, v) -> v.reduce { a, b -> a + b } }
    .toList().sortedBy { it.first }
    .joinToString("\n", "", "\n") { (k, v) -> "|σ|∈[$k, ${k+10}): $v" } +
  // By distribution of Levenshtein distances
  entries.groupBy({ it.key.second }, { it.value })
    .mapValues { (_, v) ->  v.reduce { a, b -> a + b } }
    .toList().sortedBy { it.first }
    .joinToString("\n", "", "\n") { (k, v) -> "Δ($k)= $v" } +
  // Joint distribution
      entries.sortedWith(compareBy({ it.key.first }, { it.key.second }))
        .joinToString("\n", "", "\n") { (k, v) -> "(|σ|∈[${k.first}, ${k.first+LEN_BUCKET_INTERVAL}), Δ=${k.second}): $v" }

data class S2PMetrics(var top1: Int = 0, var total: Int = 0) {
  operator fun plus(other: S2PMetrics) =
    S2PMetrics(top1 + other.top1, total + other.total)
  override fun toString() =
    "Top-1/total: $top1 / $total = ${top1.toDouble() / total}"
}

/*
Precision@1
===========
|σ|∈[0, 10): Top-1/total: 30 / 76 = 0.39473684210526316
|σ|∈[10, 20): Top-1/total: 48 / 142 = 0.3380281690140845
|σ|∈[20, 30): Top-1/total: 44 / 126 = 0.3492063492063492
|σ|∈[30, 40): Top-1/total: 31 / 131 = 0.2366412213740458
|σ|∈[40, 50): Top-1/total: 12 / 30 = 0.4
Δ(1)= Top-1/total: 91 / 180 = 0.5055555555555555
Δ(2)= Top-1/total: 44 / 178 = 0.24719101123595505
Δ(3)= Top-1/total: 30 / 147 = 0.20408163265306123
(|σ|∈[0, 10), Δ=1): Top-1/total: 17 / 30 = 0.5666666666666667
(|σ|∈[0, 10), Δ=2): Top-1/total: 9 / 27 = 0.3333333333333333
(|σ|∈[0, 10), Δ=3): Top-1/total: 4 / 19 = 0.21052631578947367
(|σ|∈[10, 20), Δ=1): Top-1/total: 27 / 51 = 0.5294117647058824
(|σ|∈[10, 20), Δ=2): Top-1/total: 13 / 50 = 0.26
(|σ|∈[10, 20), Δ=3): Top-1/total: 8 / 41 = 0.1951219512195122
(|σ|∈[20, 30), Δ=1): Top-1/total: 26 / 51 = 0.5098039215686274
(|σ|∈[20, 30), Δ=2): Top-1/total: 7 / 41 = 0.17073170731707318
(|σ|∈[20, 30), Δ=3): Top-1/total: 11 / 34 = 0.3235294117647059
(|σ|∈[30, 40), Δ=1): Top-1/total: 12 / 30 = 0.4
(|σ|∈[30, 40), Δ=2): Top-1/total: 13 / 52 = 0.25
(|σ|∈[30, 40), Δ=3): Top-1/total: 6 / 49 = 0.12244897959183673
(|σ|∈[40, 50), Δ=1): Top-1/total: 9 / 18 = 0.5
(|σ|∈[40, 50), Δ=2): Top-1/total: 2 / 8 = 0.25
(|σ|∈[40, 50), Δ=3): Top-1/total: 1 / 4 = 0.25

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 75 / 76 = 0.9868421052631579
|σ|∈[10, 20): Top-1/total: 142 / 142 = 1.0
|σ|∈[20, 30): Top-1/total: 125 / 126 = 0.9920634920634921
|σ|∈[30, 40): Top-1/total: 117 / 131 = 0.8931297709923665
|σ|∈[40, 50): Top-1/total: 16 / 30 = 0.5333333333333333
Δ(1)= Top-1/total: 172 / 180 = 0.9555555555555556
Δ(2)= Top-1/total: 174 / 178 = 0.9775280898876404
Δ(3)= Top-1/total: 129 / 147 = 0.8775510204081632
(|σ|∈[0, 10), Δ=1): Top-1/total: 30 / 30 = 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 27 / 27 = 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 18 / 19 = 0.9473684210526315
(|σ|∈[10, 20), Δ=1): Top-1/total: 51 / 51 = 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 50 / 50 = 1.0
(|σ|∈[10, 20), Δ=3): Top-1/total: 41 / 41 = 1.0
(|σ|∈[20, 30), Δ=1): Top-1/total: 51 / 51 = 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 41 / 41 = 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 33 / 34 = 0.9705882352941176
(|σ|∈[30, 40), Δ=1): Top-1/total: 30 / 30 = 1.0
(|σ|∈[30, 40), Δ=2): Top-1/total: 52 / 52 = 1.0
(|σ|∈[30, 40), Δ=3): Top-1/total: 35 / 49 = 0.7142857142857143
(|σ|∈[40, 50), Δ=1): Top-1/total: 10 / 18 = 0.5555555555555556
(|σ|∈[40, 50), Δ=2): Top-1/total: 4 / 8 = 0.5
(|σ|∈[40, 50), Δ=3): Top-1/total: 2 / 4 = 0.5
 */