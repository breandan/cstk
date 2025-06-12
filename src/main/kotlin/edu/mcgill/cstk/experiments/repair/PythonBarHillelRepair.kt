package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.types.to
import edu.mcgill.cstk.experiments.probing.MakeMore
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.*
import kotlin.math.*
import kotlin.sequences.toList
import kotlin.streams.*
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds
import kotlin.to

/*
./gradlew pythonBarHillelRepair
 */
fun main() {
  printMemoryUsage()
  LangCache.prepopPythonLangCache()
//  MAX_UNIQUE = 1_000
  TIMEOUT_MS = 30_000
  MIN_TOKENS = 3
  MAX_TOKENS = 80
  MAX_RADIUS = 3
  CFG_THRESH = 10_000
  evaluateRegexRepairOnStackOverflow()
//  evaluateMatrixBarHillelRepairOnStackOverflow()
//  evaluateBarHillelRepairOnStackOverflow()
//  evaluateSeq2ParseRepair()
//  evaluateBIFIRepair()
//  measureLevenshteinBlanketSize()
//  writeParikhMap()
}

val LEN_BUCKET_INTERVAL = 10

//fun writeParikhMap() {
//  val txt = ParikhMap.serialize(vanillaS2PCFG.parikhMap)
//  File("" + vanillaS2PCFG.hashCode() + ".cache")
//    .also { println("Wrote to ${it.absolutePath}") }
//    .writeText(txt)
//}

fun printMemoryUsage() {
  val runtime = Runtime.getRuntime()
  val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
  val maxMemory = runtime.maxMemory() / (1024 * 1024)
  val allocatedMemory = runtime.totalMemory() / (1024 * 1024)

  println("Used Memory: $usedMemory MB")
  println("Allocated Memory: $allocatedMemory MB")
  println("Max Memory: $maxMemory MB")
}

fun readResourceFile(path: String) = object {}.javaClass.classLoader.getResource(path)!!.readText()

fun readPCFG3(): Map<Π3A<Σᐩ>, Int> =
  readResourceFile("models/pcfg3_BIFI.csv").lines().map { it.split(" ::: ") }
  .associate { Pair(it[0].split(" ").let { it[0] to it[1] to it[2] }, it[1].toInt()) }

fun readPCFG5(s2pg: CFG): Map<Int, Int> =
  readResourceFile("models/pcfg5_BIFI.csv")
    .lines().map { it.split(" ::: ") }
    .associate { Pair(it[0].split(" ")
      .map { if (it.endsWith('*') && it.length > 1) (31 * s2pg.symMap[it.dropLast(1)]!!) else s2pg.symMap[it] ?: Int.MAX_VALUE }
      /** See [Tree.quintuples] */
      .let { hash(it[0], it[1], it[2], it[3], it[4]) }, it[1].toInt()) }

val s2pg by lazy { vanillaS2PCFG }
val parikhMap by lazy {
  LangCache.prepopPythonLangCache()
  s2pg.parikhMap }
val termDict by lazy { TermDict(s2pg.terminals) }

fun parallelPythonRepair(brokeStr: String): List<Σᐩ> {
  val brokeToks = brokeStr.tokenizeByWhitespace()

  val langEditDist = (1..MAX_RADIUS).firstOrNull {
    try {
      val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(brokeToks, pad = it)
      val fsa = makeLevFSA(brokeToks, it, monoEditBounds)
      s2pg.jvmIntersectLevFSAP(fsa = fsa, parikhMap = parikhMap).isNotEmpty()
    } catch (_: Exception) { println("Failed $it, increasing..."); false }
  } ?: MAX_RADIUS

  val levGuess = langEditDist + 2

  val intGram = try {
    val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(brokeToks)
    val fsa = makeLevFSA(brokeToks, levGuess, monoEditBounds)

    s2pg.jvmIntersectLevFSAP(fsa = fsa, parikhMap = parikhMap)
      .also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
  } catch (e: Exception) { return emptyList() }
  catch (e: Error) { return emptyList() }

  val pTree = measureTimedValue { intGram.toPTree(origCFG = s2pg) }
    .also { println("Constructed PTree in ${it.duration}") }.value
  val timeout = (TIMEOUT_MS / 1000).seconds

  val dfa = pTree.toDFA(minimize = false)!!

  val rankedResults = dfa.decodeDFA(mc = P_BIFI_PY150, timeout = timeout, dec = termDict)

  return rankedResults
}

fun evaluateRegexRepairOnStackOverflow() {
  val dataset = sizeAndDistBalancedRepairsUnminimized
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()

  println("Running Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  println("Sampling timeout: $TIMEOUT_MS ms, max tokens: $MAX_TOKENS, max radius: $MAX_RADIUS, max unique: $MAX_UNIQUE, CFG threshold: $CFG_THRESH")
  dataset.first().π2.let { P_BIFI_PY150.score(it.tokenizeByWhitespace()) }

  val latestCommitMessage = lastGitMessage().replace(Regex("[^A-Za-z0-9]"), "_")
    .let { if ("fatal: not a git repository" !in it) it else System.currentTimeMillis().toString() }
//    .replace(" ", "_").replace("/", "_")
  val positiveHeader = "length, lev_dist, sample_ms, total_ms, " +
      "total_samples, lev_ball_arcs, productions, lang_size, dfa_states, dfa_transitions, rank, edit1, edit2, edit3\n"
  val negativeHeader = "length, lev_dist, samples, lev_states, productions, lang_size, dfa_states, dfa_transitions, edit1, edit2, edit3\n"
  val title = "matrix_bar_hillel"
  val positive = try { File("data/${title}_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/${title}_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
    .also { println("Writing positive CSV to: ${it.absolutePath}") }
  val negative = try { File("data/${title}_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/${title}_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
    .also { println("Writing negative CSV to: ${it.absolutePath}") }
  println()

  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_10ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_100ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_AllByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  dataset.asStream().parallel().forEach { (brokeStr, fixedStr) ->
    val allTime = TimeSource.Monotonic.markNow()
    val brokeToks = brokeStr.tokenizeByWhitespace()
    val fixedToks = fixedStr.tokenizeByWhitespace()
    val levAlign = levenshteinAlign(brokeToks, fixedToks)
    val levDist = levAlign.patchSize() // True distance, only used for logging purposes

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_10ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_100ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    val humanRepairANSI = levenshteinAlign(brokeToks, fixedToks).paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")

    val clock = TimeSource.Monotonic.markNow()
    var totalSamples = 0
    var matchFound = false
    val timeout = (TIMEOUT_MS / 1000).seconds
    var elapsed = clock.elapsedNow().inWholeMilliseconds

    val rankedResults = sendCPU(brokeStr).lines().map { it.addNewLineIfMissing() }.onEach {
        totalSamples++
        if (it == fixedStr) {
          matchFound = true
          println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
          elapsed = clock.elapsedNow().inWholeMilliseconds
        }
      }

    val indexOfTarget = rankedResults.indexOf(fixedStr).also {
      if (it == 0) P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      if (it <= 10) P_10ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      if (it <= 100) P_100ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      if (matchFound) P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
    }

    rankedResults.firstOrNull()?.tokenizeByWhitespace()
      ?.let { println("Top1 scoring repair: ${levenshteinAlign(brokeToks, it).paintANSIColors()}") }

    if (indexOfTarget < 0) {
      println("Drew $totalSamples samples in ${clock.elapsedNow()}/$timeout with ? prods, " +
//        "${dfa.numStates} states, ${dfa.numberOfTransitions} transitions, " +
          "length-$levDist human repair not found")
      negative.appendText("${brokeToks.size}, $levDist, $totalSamples, ?, ?, ?, ${levAlign.summarize()}\n")
    } else {
      val allElapsed = allTime.elapsedNow().inWholeMilliseconds

      allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
      indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
      println("Found length-$levDist repair in $elapsed ms, $allElapsed ms," +
          " $totalSamples samples, ? prods, ? trees, $indexOfTarget rank")//, rank: ${rankedResults.indexOf(fixedTks) + 1} / ${rankedResults.size}")
      allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
//      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
      sampleTimeByLevDist[levDist] = (sampleTimeByLevDist[levDist] ?: 0.0) + elapsed
      println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
      allTimeByLevDist[levDist] = (allTimeByLevDist[levDist] ?: 0.0) + allElapsed
      println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
      samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + totalSamples
      println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
      positive.appendText("${brokeToks.size}, $levDist, $elapsed, $allElapsed, " +
          "0, 0" +
          "$indexOfTarget, ${levAlign.summarize()}\n")
    }

    println()
    println("Precision@1\n===========")
    println(P_1ByLevDist.summarizeLenAndDist())
    println("Precision@10\n===========")
    println(P_10ByLevDist.summarizeLenAndDist())
    println("Precision@100\n===========")
    println(P_100ByLevDist.summarizeLenAndDist())
    println("Precision@All\n=============")
    println(P_AllByLevDist.summarizeLenAndDist())
    println()
  }
}

fun evaluateBarHillelRepairOnStackOverflow() {
  val dataset = sizeAndDistBalancedRepairsUnminimized//corruptedBIFIGoodCode//sizeAndDistBalancedRepairsUnminimized.toList()
   // timeoutCases // corruptedBIFIGoodCode // balancedSmallRepairsUnminimized.toList() // naturallySmallRepairs //pairwiseUniformAll
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val s2pg = vanillaS2PCFG
  val termDict = TermDict(s2pg.terminals)
  val parikhMap = s2pg.parikhMap
  val pcfgMap = readPCFG3()
  val pcfgNorm = s2pg.nonterminals.associateWith { nt -> pcfgMap.filterKeys { it.first == nt }.values.sum() }

  println("Running Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  println("Sampling timeout: $TIMEOUT_MS ms, max tokens: $MAX_TOKENS, " +
      "max radius: $MAX_RADIUS, max unique: $MAX_UNIQUE, CFG threshold: $CFG_THRESH")
  dataset.first().π2.let { P_BIFI_PY150.score(it.tokenizeByWhitespace()) }

  val latestCommitMessage = lastGitMessage().replace(Regex("[^A-Za-z0-9]"), "_")
    .let { if ("fatal: not a git repository" !in it) it else System.currentTimeMillis().toString() }
//    .replace(" ", "_").replace("/", "_")
  val positiveHeader = "length, lev_dist, sample_ms, total_ms, " +
      "total_samples, lev_ball_arcs, productions, lang_size, dfa_states, dfa_transitions, rank, edit1, edit2, edit3\n"
  val negativeHeader = "length, lev_dist, samples, lev_states, productions, lang_size, dfa_states, dfa_transitions, edit1, edit2, edit3\n"
  val title = "matrix_bar_hillel"
  val positive = try { File("data/${title}_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/${title}_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
    .also { println("Writing positive CSV to: ${it.absolutePath}") }
  val negative = try { File("data/${title}_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/${title}_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
    .also { println("Writing negative CSV to: ${it.absolutePath}") }
  println()

  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_AllByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val editLocationsByLenAndDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  dataset.forEach { (brokeStr, fixedStr) ->
    val allTime = TimeSource.Monotonic.markNow()
    val brokeToks = brokeStr.tokenizeByWhitespace()
    val fixedToks = fixedStr.tokenizeByWhitespace()
    val encString = "|${MakeMore.encode(brokeStr)} "
    val levAlign = levenshteinAlign(brokeToks, fixedToks)

    // Declare the number of edits we are going to make up front
//    val predDist = MakeMore.predDist(encString)
    val predDist = 0
    val langEditDist = (1..MAX_RADIUS).firstOrNull {
        try {
          val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(brokeToks, pad = it)
          val fsa = makeLevFSA(brokeToks, it, monoEditBounds)
          s2pg.jvmIntersectLevFSAP(fsa = fsa, parikhMap = parikhMap).isNotEmpty()
        } catch (_: Exception) { println("Failed $it, increasing..."); false }
      } ?: MAX_RADIUS
    val levGuess = levAlign.patchSize() //min(predDist, langEditDist)

    val levDist = levAlign.patchSize() // True distance, only used for logging purposes
    println("Predicted edit dist: $predDist (true dist: $levDist, LED: $langEditDist)")

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    var levBallSize = 1
    val humanRepairANSI = levenshteinAlign(brokeToks, fixedToks).paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")

    val intGram = try {
      val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(brokeToks, pad = levGuess)
//    val multiEditBounds = vanillaS2PCFGWE.findMinimalMultiEditBounds(toRepair, monoEditBounds, levDist)
      val fsa = makeLevFSA(brokeToks, levGuess, monoEditBounds).also { levBallSize = it.Q.size }

      if (!fsa.recognizes(fixedToks))
        throw Exception("Human repair is unrecognizable! (Total time=${allTime.elapsedNow()})")
      else println("LEV-FSA recognizes human repair (Total time=${allTime.elapsedNow()})")

      s2pg.jvmIntersectLevFSAP(fsa = fsa, parikhMap = parikhMap)
        .also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
    } catch (e: Exception) { println("$humanRepairANSI\nIntersection exception: ${e.stackTraceToString()}"); null }
    catch (e: Error) { println("$humanRepairANSI\nIntersection error: ${e.stackTraceToString()}"); null }

    if (intGram != null) println("Constructed LEV($levGuess, ${brokeToks.size}, $levBallSize) " +
      "∩ CFG grammar with ${intGram.size} productions in ${allTime.elapsedNow()}")

    println("Implicated nonterminals: " +
        (intGram?.nonterminals?.map { if(it == "START") it else it.split("~")[1] }?.toSet()?.size ?: 0) +
        " / " + s2pg.nonterminals.size)

    allRate.total++; levRates.getOrPut(levDist) { LBHMetrics() }.total++
    try {
      if (intGram == null) throw Exception("Exception while building grammar!")
      else if (MAX_DFA_IN < intGram.size) throw Exception("Int grammar was still too large!")
      else if (fixedToks !in intGram.language) {
        println("Human repair recognized by original CFG: " + (fixedToks in vanillaS2PCFG.language))
        throw Exception("Human repair is unrecognizable by LEV ∩ CFG! (Total time=${allTime.elapsedNow()})")
      } else println("Human repair is recognized by LEV ∩ CFG! (Total time=${allTime.elapsedNow()})")
    } catch (e: Exception) {
      println("Encountered error ${e.message} ${allTime.elapsedNow()}):\n$humanRepairANSI\n${e.stackTraceToString()}")
      allRate.error++; levRates.getOrPut(levDist) { LBHMetrics() }.error++
      println(allRate.toString())
      negative.appendText("${brokeToks.size}, $levDist, 0, " +
        "${levBallSize}, ${intGram?.size ?: 0}, ${levAlign.summarize()}\n")

      println()
      println("Precision@1\n===========")
      println(P_1ByLevDist.summarizeLenAndDist())
      println("Precision@All\n=============")
      println(P_AllByLevDist.summarizeLenAndDist())
      println()
      return@forEach
    }

    val pTree = measureTimedValue { intGram.toPTree(origCFG = s2pg) }
      .also { println("Constructed PTree in ${it.duration}") }.value
    val langSize = pTree.totalTreesStr
    val clock = TimeSource.Monotonic.markNow()
    var totalSamples = 0
    var matchFound = false
    val timeout = (TIMEOUT_MS / 1000).seconds
    var elapsed = clock.elapsedNow().inWholeMilliseconds

    val dfa = pTree.toDFA(minimize = true)!!

//    println(dfa.toDot().replaceAll(vanillaS2PCFG.unicodeMap))

    val dfaRecognized = try { dfa.run(termDict.encode(fixedToks)) } catch (_: Exception) { false }
    println("∩-DFA ${if (dfaRecognized) "accepted" else "rejected"} human repair! (Total time=${allTime.elapsedNow()})")

    val rankedResults = dfa.decodeDFA(
      mc = P_BIFI_PY150,
      timeout = timeout,
      dec = termDict,
      callback = {
        totalSamples++
        if (it == fixedStr) {
          matchFound = true
          println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
          elapsed = clock.elapsedNow().inWholeMilliseconds
        }
      }
    )

//    val rankedResults = MakeMore.decodeDFA(
//      origStr = "$encString$levGuess ",
//      bAutomaton = dfa,
//      timeout = timeout,
//      dec = termDict,
//      callback = {
//        totalSamples++
//        if (it == fixedStr) {
//          matchFound = true
//          println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
//          elapsed = clock.elapsedNow().inWholeMilliseconds
//        }
//      }
//    )

//    rankedResults.take(100).forEach {
//      println("Sample: ${levenshteinAlign(humanRepair, it.tokenizeByWhitespace()).paintANSIColors()}")
//      println(it in vanillaS2PCFG.language)
//    }

    val indexOfTarget = rankedResults.indexOf(fixedStr).also {
      if (it == 0) P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      if (matchFound) P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
    }

    rankedResults.firstOrNull()?.tokenizeByWhitespace()
      ?.let { println("Top1 scoring repair: ${levenshteinAlign(brokeToks, it).paintANSIColors()}") }

    if (indexOfTarget < 0) {
      println("Drew $totalSamples samples in ${clock.elapsedNow()}/$timeout with ${intGram.size} prods, " +
//        "${dfa.numStates} states, ${dfa.numberOfTransitions} transitions, " +
          "length-$levDist human repair not found")
      negative.appendText(
        "${brokeToks.size}, $levDist, $totalSamples, ${levBallSize}, " +
          "${intGram.size}, $langSize, " +
//          "${dfa.numStates}, ${dfa.numberOfTransitions}, " +
          "${levAlign.summarize()}\n"
      )
    } else {
      val allElapsed = allTime.elapsedNow().inWholeMilliseconds

      allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
      indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
      println("Found length-$levDist repair in $elapsed ms, $allElapsed ms," +
        " $totalSamples samples, ${intGram.size} prods, $langSize trees, $indexOfTarget rank")//, rank: ${rankedResults.indexOf(fixedTks) + 1} / ${rankedResults.size}")
      allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
//      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
      sampleTimeByLevDist[levDist] = (sampleTimeByLevDist[levDist] ?: 0.0) + elapsed
      println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
      allTimeByLevDist[levDist] = (allTimeByLevDist[levDist] ?: 0.0) + allElapsed
      println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
      samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + totalSamples
      println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
      positive.appendText("${brokeToks.size}, $levDist, $elapsed, $allElapsed, " +
        "$totalSamples, ${levBallSize}, ${intGram.size}, $langSize, " +
          "${dfa.numberOfStates}, ${dfa.numberOfTransitions}, " +
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
    "Top-1/rec/pos/total: $top1 / $recall / ${total-error} / ${total}, " +
      "errors: $error, P@1: ${top1.toDouble() / total}, P@All: ${recall.toDouble() / total}"
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

val paperExamples = listOf(
//  "NAME = NAME . NAME ( NUMBER : , NUMBER : )" to "NAME = NAME . NAME [ NUMBER : , NUMBER : ]",
//  "{ STRING : [ STRING , STRING , STRING ] STRING : [ STRING , STRING , STRING ] STRING : [ STRING , STRING , STRING ] STRING : [ STRING , STRING , STRING ] } NEWLINE" to
//      "{ STRING : [ STRING , STRING , STRING ] , STRING : [ STRING , STRING , STRING ] , STRING : [ STRING , STRING , STRING ] , STRING : [ STRING , STRING , STRING ] } NEWLINE"
//  "NAME = NAME ( STRING ) NEWLINE while NAME . NAME ( ) != NAME ( STRING ) or NAME . NAME ( ) != NAME ( STRING ) NEWLINE INDENT NAME . NAME ( ) == NAME ( NAME ( STRING ) ) NEWLINE DEDENT NEWLINE" to
//  "NAME = NAME ( STRING ) NEWLINE while NAME . NAME ( ) != NAME ( STRING ) and NAME . NAME ( ) != NAME ( STRING ) : NEWLINE INDENT NAME . NAME ( ) == NAME ( NAME ( STRING ) ) NEWLINE DEDENT NEWLINE",
  "NAME = [ NUMBER NUMBER ] NEWLINE" to "NAME = [ NUMBER , NUMBER ] NEWLINE",
  "[ ( STRING : NUMBER ) , ( STRING : NUMBER ) , ( STRING : NUMBER ) , ( STRING : NUMBER ) ] NEWLINE" to
    "[ ( STRING , NUMBER ) , ( STRING , NUMBER ) , ( STRING , NUMBER ) , ( STRING , NUMBER ) ] NEWLINE",
)

val largeIntersectionInstances = listOf(
  "NAME ( STRING . NAME ( ( NAME & NAME ) ) or STRING NEWLINE" to
  "NAME ( STRING . NAME ( NAME & NAME ) or STRING ) NEWLINE",
  "import NAME NEWLINE def NAME ( NAME , NAME , NAME ) : NEWLINE INDENT for NAME , NAME in NAME ( NAME . NAME ( NAME , NAME ) ] ) : NEWLINE INDENT if NAME == NAME - NUMBER : NEWLINE INDENT return NAME . NAME ( ) NEWLINE DEDENT DEDENT DEDENT NAME = STRING NEWLINE NAME = STRING NEWLINE NAME ( NAME ( NAME , NAME , NUMBER ) ) NEWLINE" to
  "import NAME NEWLINE def NAME ( NAME , NAME , NAME ) : NEWLINE INDENT for NAME , NAME in NAME ( NAME . NAME ( NAME , NAME ) ] ) : NEWLINE INDENT if NAME == NAME - NUMBER : NEWLINE INDENT return NAME . NAME ( ) NEWLINE DEDENT DEDENT DEDENT NAME = STRING NEWLINE NAME = STRING NEWLINE NAME ( NAME ( NAME , NAME , NUMBER ) ) NEWLINE"
)

val shortcutTestcases: List<Pair<String, String>> = listOf(
  "STRING : { STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER } , NEWLINE STRING : { STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER } NEWLINE" to
  "{ STRING : { STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER } , STRING : { STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER , STRING : NUMBER } } NEWLINE",
    "NAME = { STRING : { STRING : [ { STRING : { STRING : STRING , STRING : STRING , STRING : STRING , STRING : STRING } } NEWLINE" to
    "NAME = { STRING : { STRING : [ { STRING : { STRING : STRING , STRING : STRING , STRING : STRING , STRING : STRING } } ] } } NEWLINE",
    "try : NEWLINE INDENT raise NAME ( STRING ) NEWLINE DEDENT except NAME , NAME : NEWLINE INDENT raise NAME ( STRING ) from NAME NEWLINE DEDENT NEWLINE" to
    "try : NEWLINE INDENT raise NAME ( STRING ) NEWLINE DEDENT except NAME as NAME : NEWLINE INDENT raise NAME ( STRING ) from NAME NEWLINE DEDENT NEWLINE",
    "NAME = { STRING : { STRING : { STRING : { STRING : STRING , STRING STRING } , STRING : { } , STRING : { } } } NEWLINE" to
    "NAME = { STRING : { STRING : { STRING : { STRING : STRING , STRING : STRING } , STRING : { } , STRING : { } } } } NEWLINE",
    "NEWLINE INDENT def NAME ( NAME ) : NEWLINE INDENT NAME = NAME ( STRING ) ; NEWLINE NAME = NAME ( ) ; NEWLINE NAME . NAME = NAME ; NEWLINE return NAME ; NEWLINE DEDENT DEDENT class NAME : NEWLINE INDENT pass NEWLINE DEDENT class NAME ( NAME ) : NEWLINE INDENT NAME = None NEWLINE DEDENT NAME = NAME ( ) ; NEWLINE NAME = NAME . NAME ( ) ; NEWLINE" to
    "class NAME : NEWLINE INDENT def NAME ( NAME ) : NEWLINE INDENT NAME = NAME ( STRING ) ; NEWLINE NAME = NAME ( ) ; NEWLINE NAME . NAME = NAME ; NEWLINE return NAME ; NEWLINE DEDENT DEDENT class NAME : NEWLINE INDENT pass NEWLINE DEDENT class NAME ( NAME ) : NEWLINE INDENT NAME = None NEWLINE DEDENT NAME = NAME ( ) ; NEWLINE NAME = NAME . NAME ( ) ; NEWLINE",
    "STRING : NUMBER , STRING : STRING , STRING : [ { STRING : STRING , STRING : NUMBER } , { STRING : STRING , STRING : NUMBER } , { STRING : STRING , STRING : NUMBER } ] NEWLINE" to
    "{ STRING : NUMBER , STRING : STRING , STRING : [ { STRING : STRING , STRING : NUMBER } , { STRING : STRING , STRING : NUMBER } , { STRING : STRING , STRING : NUMBER } ] } NEWLINE",
    "import NAME NEWLINE try : NEWLINE INDENT NUMBER / NUMBER NEWLINE DEDENT except NAME , NAME : NEWLINE INDENT NAME . NAME ( NAME ) NEWLINE DEDENT NEWLINE" to
    "import NAME NEWLINE try : NEWLINE INDENT NUMBER / NUMBER NEWLINE DEDENT except NAME as NAME : NEWLINE INDENT NAME . NAME ( STRING ) NEWLINE DEDENT NEWLINE",
    "NAME ( NAME ( NAME ( NAME ( NAME ( STRING ) . NAME ( ) , NAME = lambda NAME : ) NAME ( NAME ( NAME ) ) ) ) ) NEWLINE" to
    "NAME ( NAME ( NAME ( NAME ( NAME ( STRING ) . NAME ( ) , NAME = lambda NAME : NAME ( NAME ( NAME ) ) ) ) ) ) NEWLINE",

//  "NAME = [ { STRING : NUMBER , STRING : NUMBER } , { STRING : NUMBER , STRING : NUMBER } , { STRING : NUMBER , STRING : STRING ] NAME = [ { STRING : NUMBER , STRING : NUMBER } , { STRING : NUMBER , STRING : NUMBER } ] NEWLINE" to
//  "NAME = [ { STRING : NUMBER , STRING : NUMBER } , { STRING : NUMBER , STRING : NUMBER } , { STRING : NUMBER , STRING : STRING } ] NEWLINE NAME = [ { STRING : NUMBER , STRING : NUMBER } , { STRING : NUMBER , STRING : NUMBER } ] NEWLINE",
//  "def NAME ( NAME . NAME ) : NEWLINE INDENT NAME = NAME . NAME ( NAME = STRING ) NEWLINE DEDENT def NAME ( NAME . NAME ) : NEWLINE INDENT pass NEWLINE DEDENT def NAME ( NAME . NAME ) : NEWLINE INDENT NAME = NAME . NAME ( NAME ) NEWLINE NAME = NAME . NAME ( NAME ) NEWLINE DEDENT NEWLINE" to
//  "class NAME ( NAME . NAME ) : NEWLINE INDENT NAME = NAME . NAME ( NAME = STRING ) NEWLINE DEDENT class NAME ( NAME . NAME ) : NEWLINE INDENT pass NEWLINE DEDENT class NAME ( NAME . NAME ) : NEWLINE INDENT NAME = NAME . NAME ( NAME ) NEWLINE NAME = NAME . NAME ( NAME ) NEWLINE DEDENT NEWLINE",
//  "NAME . NAME ( STRING . NAME ( NAME [ STRING ) . NAME ( ) . NAME ( ) ) . NAME ( ) [ : NUMBER ] NEWLINE" to
//  "NAME . NAME ( STRING . NAME ( NAME [ STRING ] ) . NAME ( ) . NAME ( ) ) . NAME ( ) [ : NUMBER ] NEWLINE",
//  "from NAME import NAME NEWLINE from . NAME import NAME , NAME NEWLINE from NAME import * NEWLINE from NAME import NAME NEWLINE from NAME import NAME NEWLINE class NAME ( NAME . NAME ) : NEWLINE NAME = NAME . NAME . NAME ( ) NEWLINE NAME = NAME NEWLINE NAME = ( STRING , ) NEWLINE" to
//  "from NAME import NAME NEWLINE from . NAME import NAME , NAME NEWLINE from NAME import * NEWLINE from NAME import NAME NEWLINE from NAME import NAME NEWLINE class NAME ( NAME . NAME ) : NEWLINE INDENT NAME = NAME . NAME . NAME ( ) NEWLINE NAME = NAME NEWLINE NAME = ( STRING , ) NEWLINE DEDENT NEWLINE",
//  "NAME = NEWLINE STRING : { STRING : [ ] , STRING : [ ] , STRING : [ { STRING : [ NUMBER , NUMBER ] , STRING : STRING } , { STRING : [ NUMBER , NUMBER ] , STRING : STRING } ] , STRING : [ { STRING : STRING , STRING : [ NUMBER , NUMBER ] , STRING : STRING , STRING : STRING } ] } , NEWLINE" to
//  "NAME = { STRING : { STRING : [ ] , STRING : [ ] , STRING : [ { STRING : [ NUMBER , NUMBER ] , STRING : STRING } , { STRING : [ NUMBER , NUMBER ] , STRING : STRING } ] , STRING : [ { STRING : STRING , STRING : [ NUMBER , NUMBER ] , STRING : STRING , STRING : STRING } ] } } , NEWLINE",
//  "import NAME NEWLINE INDENT NAME = NAME . NAME ( NAME = STRING , NAME = NUMBER ) NEWLINE NAME ( NAME . NAME ) NEWLINE while True : NEWLINE INDENT NAME ( NAME . NAME ( NUMBER ) . NAME ( ) ) NEWLINE DEDENT NAME . NAME ( ) NEWLINE DEDENT NEWLINE" to
//  "import NAME NEWLINE NAME = NAME . NAME ( NAME = STRING , NAME = NUMBER ) NEWLINE NAME ( NAME . NAME ) NEWLINE while True : NEWLINE INDENT NAME ( NAME . NAME ( NUMBER ) . NAME ( ) ) NEWLINE DEDENT NAME . NAME ( ) NEWLINE",
//  "import NAME as NAME NEWLINE in = NAME . NAME ( [ NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER ] ) NEWLINE NAME = [ NAME . NAME ( in == NAME ) [ NUMBER ] . NAME ( ) for NAME in NAME . NAME ( in ) ] NEWLINE" to
//  "import NAME as NAME NEWLINE NAME = NAME . NAME ( [ NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER , NUMBER ] ) NEWLINE NAME = [ NAME . NAME ( NAME == NAME ) [ NUMBER ] . NAME ( ) for NAME in NAME . NAME ( NAME ) ] NEWLINE",
//  "NAME = NAME ( STRING ) NEWLINE NAME = NAME ( STRING ) NEWLINE NAME = { } NEWLINE NAME = { } NEWLINE for NAME in NAME : NEWLINE INDENT NAME [ NAME ] = NUMBER NEWLINE DEDENT for NAME in NAME : NEWLINE INDENT NAME [ NAME ] += NUMBER NEWLINE DEDENT for NAME in NAME NEWLINE INDENT NAME [ NAME ] = NUMBER NEWLINE DEDENT for NAME in NAME : NEWLINE INDENT NAME [ NAME ] += NUMBER NEWLINE if NAME >= NAME : NEWLINE INDENT NAME ( STRING ) NEWLINE DEDENT DEDENT NEWLINE" to
//  "NAME = NAME ( STRING ) NEWLINE NAME = NAME ( STRING ) NEWLINE NAME = { } NEWLINE NAME = { } NEWLINE for NAME in NAME : NEWLINE INDENT NAME [ NAME ] = NUMBER NEWLINE DEDENT for NAME in NAME : NEWLINE INDENT NAME [ NAME ] += NUMBER NEWLINE DEDENT for NAME in NAME : NEWLINE INDENT NAME [ NAME ] = NUMBER NEWLINE DEDENT for NAME in NAME : NEWLINE INDENT NAME [ NAME ] += NUMBER NEWLINE if NAME >= NAME : NEWLINE INDENT NAME ( STRING ) NEWLINE DEDENT NEWLINE",
//  "NAME = NAME . NAME . NAME ( STRING . NAME ( ) NAME = NUMBER NAME = NUMBER NEWLINE" to
//  "NAME = NAME . NAME . NAME ( STRING ) . NAME ( ) NEWLINE NAME = NUMBER NEWLINE NAME = NUMBER NEWLINE"
)

// Returns a quintuple of lexical (broke, fixed) and original code (broke, fixed) pairs
val sizeAndDistBalancedRepairsUnminimized: Sequence<Π4A<Σᐩ>> by lazy {
//  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
//  val file = File(File("").absolutePath + path).readText()
  val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64_tst.txt"
  val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()
  val decoder = Base64.getDecoder()
  contents.lines().asSequence().windowed(4, 4).map { it[0] to it[1] to it[2] to it[3] }
    .asStream().parallel()
    .map { (a, b, c, d) -> a.addNewLineIfMissing() to b.addNewLineIfMissing() to
        String(decoder.decode(c)) to String(decoder.decode(d)) }
    .map { (a, b, c, d) ->
      val broke = a.tokenizeByWhitespace()
      val levDist = levenshtein(broke, b.tokenizeByWhitespace())
      a to b to c to d to ((broke.size / 10) * 10 to levDist)
    }.filter { (broke, fixed, bc, oc, size) ->
      broke.tokenizeByWhitespace().size in MIN_TOKENS until MAX_TOKENS &&
          fixed.tokenizeByWhitespace().size in MIN_TOKENS until MAX_TOKENS &&
        size.second <= MAX_RADIUS
    }.toList()
    .groupBy { it.π5 }.let { map ->
      val minSize = map.entries.minBy { it.value.size }
      println("Size of smallest group: ${minSize.key}, ${minSize.value.size}")
      map.mapValues { (_, v) -> v.shuffled().take(300) }
    }
    .values.asSequence().flatten()
    .map { it.π1 to it.π2 to it.π3 to it.π4 }
    .distinct().shuffled()
    .filter { (broke, fixed, _, _) -> fixed in vanillaS2PCFG.language }
}

val corruptedBIFIGoodCode by lazy {
  readBIFIContents().asStream().parallel()
    .map { it.mapToUnquotedPythonTokens().addNewLineIfMissing() }
    .filter {
      it.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
        it in vanillaS2PCFG.language
    }
    .flatMap { goodCode ->
      goodCode.naturalPythonCorruptions().distinct().filter {
        levenshtein(goodCode, it) in 1..3 &&
            it !in vanillaS2PCFG.language
      }.take(10).map { it to goodCode }.asStream()
    }
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
    .filter { (broke, fixed) -> fixed in vanillaS2PCFG.language }
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
  val parikhMap = gram.parikhMap
  val timeout = 10.seconds
  val pcfgMap = readPCFG5(gram)
  val editLocationsByLenAndDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  sizeAndDistBalancedRepairsUnminimized.toList()
    .map { it.π1.addNewLineIfMissing() to it.π2.addNewLineIfMissing() }
    .forEach {  (broke, fixed) ->
      val allTime = TimeSource.Monotonic.markNow()
      val brokeTokens = broke.tokenizeByWhitespace()
      val fixedTokens = fixed.tokenizeByWhitespace()
      val levDist = levenshtein(brokeTokens, fixedTokens)
      val lenBucket = (brokeTokens.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL

      editLocationsByLenAndDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

      val levBall = makeLevFSA(brokeTokens, levDist)
      val intGram = try {
        gram.jvmIntersectLevFSAP(levBall, parikhMap = parikhMap)
          .also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
      } catch (e: Exception) { null }?.freeze()

      try {
        if (intGram == null) throw Exception("Exception while building grammar!")
        else if (MAX_DFA_IN < intGram.size) throw Exception("Int grammar was still too large!")
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
  operator fun plus(other: S2PMetrics) = S2PMetrics(top1 + other.top1, total + other.total)
  override fun toString() = "Top-1/total: $top1 / $total ≈ ${top1.toDouble() / total}"
}

fun profileRecognizer() {
  sizeAndDistBalancedRepairsUnminimized.toList().parallelStream().forEach { (invalid, valid) ->
    val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(invalid.tokenizeByWhitespace(), pad = 3)
    println(monoEditBounds)
    if(invalid.matches(vanillaS2PCFG)) println("!: $invalid")
    if(!valid.matches(vanillaS2PCFG)) println("!!: $valid")
  }
}

/*
w/ Matrix LBH, Markov Chain and exact distance

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 153 / 375 ≈ 0.408
|σ|∈[10, 20): Top-1/total: 249 / 769 ≈ 0.3237971391417425
|σ|∈[20, 30): Top-1/total: 263 / 766 ≈ 0.3433420365535248
|σ|∈[30, 40): Top-1/total: 240 / 774 ≈ 0.31007751937984496
|σ|∈[40, 50): Top-1/total: 220 / 714 ≈ 0.3081232492997199
|σ|∈[50, 60): Top-1/total: 220 / 684 ≈ 0.3216374269005848
|σ|∈[60, 70): Top-1/total: 224 / 584 ≈ 0.3835616438356164
|σ|∈[70, 80): Top-1/total: 198 / 467 ≈ 0.42398286937901497
Δ(1)= Top-1/total: 1128 / 2160 ≈ 0.5222222222222223
Δ(2)= Top-1/total: 466 / 1909 ≈ 0.24410686223153483
Δ(3)= Top-1/total: 173 / 1064 ≈ 0.162593984962406
(|σ|∈[0, 10), Δ=1): Top-1/total: 100 / 184 ≈ 0.5434782608695652
(|σ|∈[0, 10), Δ=2): Top-1/total: 44 / 133 ≈ 0.3308270676691729
(|σ|∈[0, 10), Δ=3): Top-1/total: 9 / 58 ≈ 0.15517241379310345
(|σ|∈[10, 20), Δ=1): Top-1/total: 135 / 298 ≈ 0.45302013422818793
(|σ|∈[10, 20), Δ=2): Top-1/total: 83 / 297 ≈ 0.27946127946127947
(|σ|∈[10, 20), Δ=3): Top-1/total: 31 / 174 ≈ 0.1781609195402299
(|σ|∈[20, 30), Δ=1): Top-1/total: 140 / 292 ≈ 0.4794520547945205
(|σ|∈[20, 30), Δ=2): Top-1/total: 86 / 292 ≈ 0.2945205479452055
(|σ|∈[20, 30), Δ=3): Top-1/total: 37 / 182 ≈ 0.2032967032967033
(|σ|∈[30, 40), Δ=1): Top-1/total: 141 / 290 ≈ 0.4862068965517241
(|σ|∈[30, 40), Δ=2): Top-1/total: 66 / 292 ≈ 0.22602739726027396
(|σ|∈[30, 40), Δ=3): Top-1/total: 33 / 192 ≈ 0.171875
(|σ|∈[40, 50), Δ=1): Top-1/total: 153 / 288 ≈ 0.53125
(|σ|∈[40, 50), Δ=2): Top-1/total: 49 / 288 ≈ 0.1701388888888889
(|σ|∈[40, 50), Δ=3): Top-1/total: 18 / 138 ≈ 0.13043478260869565
(|σ|∈[50, 60), Δ=1): Top-1/total: 141 / 287 ≈ 0.4912891986062718
(|σ|∈[50, 60), Δ=2): Top-1/total: 62 / 265 ≈ 0.2339622641509434
(|σ|∈[50, 60), Δ=3): Top-1/total: 17 / 132 ≈ 0.12878787878787878
(|σ|∈[60, 70), Δ=1): Top-1/total: 166 / 277 ≈ 0.5992779783393501
(|σ|∈[60, 70), Δ=2): Top-1/total: 42 / 197 ≈ 0.2131979695431472
(|σ|∈[60, 70), Δ=3): Top-1/total: 16 / 110 ≈ 0.14545454545454545
(|σ|∈[70, 80), Δ=1): Top-1/total: 152 / 244 ≈ 0.6229508196721312
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 145 ≈ 0.23448275862068965
(|σ|∈[70, 80), Δ=3): Top-1/total: 12 / 78 ≈ 0.15384615384615385

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 375 / 375 ≈ 1.0
|σ|∈[10, 20): Top-1/total: 759 / 769 ≈ 0.9869960988296489
|σ|∈[20, 30): Top-1/total: 749 / 766 ≈ 0.9778067885117493
|σ|∈[30, 40): Top-1/total: 757 / 774 ≈ 0.9780361757105943
|σ|∈[40, 50): Top-1/total: 699 / 714 ≈ 0.9789915966386554
|σ|∈[50, 60): Top-1/total: 658 / 684 ≈ 0.9619883040935673
|σ|∈[60, 70): Top-1/total: 564 / 584 ≈ 0.9657534246575342
|σ|∈[70, 80): Top-1/total: 456 / 467 ≈ 0.9764453961456103
Δ(1)= Top-1/total: 2160 / 2160 ≈ 1.0
Δ(2)= Top-1/total: 1909 / 1909 ≈ 1.0
Δ(3)= Top-1/total: 948 / 1064 ≈ 0.8909774436090225
(|σ|∈[0, 10), Δ=1): Top-1/total: 184 / 184 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 133 / 133 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 58 / 58 ≈ 1.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 298 / 298 ≈ 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 297 / 297 ≈ 1.0
(|σ|∈[10, 20), Δ=3): Top-1/total: 164 / 174 ≈ 0.9425287356321839
(|σ|∈[20, 30), Δ=1): Top-1/total: 292 / 292 ≈ 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 292 / 292 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 165 / 182 ≈ 0.9065934065934066
(|σ|∈[30, 40), Δ=1): Top-1/total: 290 / 290 ≈ 1.0
(|σ|∈[30, 40), Δ=2): Top-1/total: 292 / 292 ≈ 1.0
(|σ|∈[30, 40), Δ=3): Top-1/total: 175 / 192 ≈ 0.9114583333333334
(|σ|∈[40, 50), Δ=1): Top-1/total: 288 / 288 ≈ 1.0
(|σ|∈[40, 50), Δ=2): Top-1/total: 288 / 288 ≈ 1.0
(|σ|∈[40, 50), Δ=3): Top-1/total: 123 / 138 ≈ 0.8913043478260869
(|σ|∈[50, 60), Δ=1): Top-1/total: 287 / 287 ≈ 1.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 265 / 265 ≈ 1.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 106 / 132 ≈ 0.803030303030303
(|σ|∈[60, 70), Δ=1): Top-1/total: 277 / 277 ≈ 1.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 197 / 197 ≈ 1.0
(|σ|∈[60, 70), Δ=3): Top-1/total: 90 / 110 ≈ 0.8181818181818182
(|σ|∈[70, 80), Δ=1): Top-1/total: 244 / 244 ≈ 1.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 145 / 145 ≈ 1.0
(|σ|∈[70, 80), Δ=3): Top-1/total: 67 / 78 ≈ 0.8589743589743589

w/ LLM pairwise denoiser

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 38 / 191 ≈ 0.19895287958115182
|σ|∈[10, 20): Top-1/total: 47 / 363 ≈ 0.12947658402203857
|σ|∈[20, 30): Top-1/total: 55 / 362 ≈ 0.15193370165745856
|σ|∈[30, 40): Top-1/total: 61 / 393 ≈ 0.15521628498727735
|σ|∈[40, 50): Top-1/total: 64 / 342 ≈ 0.1871345029239766
|σ|∈[50, 60): Top-1/total: 79 / 337 ≈ 0.2344213649851632
|σ|∈[60, 70): Top-1/total: 55 / 255 ≈ 0.21568627450980393
|σ|∈[70, 80): Top-1/total: 49 / 216 ≈ 0.22685185185185186
Δ(1)= Top-1/total: 390 / 1007 ≈ 0.3872889771598808
Δ(2)= Top-1/total: 48 / 930 ≈ 0.05161290322580645
Δ(3)= Top-1/total: 10 / 522 ≈ 0.019157088122605363
(|σ|∈[0, 10), Δ=1): Top-1/total: 37 / 91 ≈ 0.4065934065934066
(|σ|∈[0, 10), Δ=2): Top-1/total: 1 / 68 ≈ 0.014705882352941176
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 32 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 41 / 138 ≈ 0.2971014492753623
(|σ|∈[10, 20), Δ=2): Top-1/total: 5 / 141 ≈ 0.03546099290780142
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 84 ≈ 0.011904761904761904
(|σ|∈[20, 30), Δ=1): Top-1/total: 48 / 135 ≈ 0.35555555555555557
(|σ|∈[20, 30), Δ=2): Top-1/total: 7 / 142 ≈ 0.04929577464788732
(|σ|∈[20, 30), Δ=3): Top-1/total: 0 / 85 ≈ 0.0
(|σ|∈[30, 40), Δ=1): Top-1/total: 52 / 139 ≈ 0.37410071942446044
(|σ|∈[30, 40), Δ=2): Top-1/total: 8 / 146 ≈ 0.0547945205479452
(|σ|∈[30, 40), Δ=3): Top-1/total: 1 / 108 ≈ 0.009259259259259259
(|σ|∈[40, 50), Δ=1): Top-1/total: 57 / 141 ≈ 0.40425531914893614
(|σ|∈[40, 50), Δ=2): Top-1/total: 5 / 140 ≈ 0.03571428571428571
(|σ|∈[40, 50), Δ=3): Top-1/total: 2 / 61 ≈ 0.03278688524590164
(|σ|∈[50, 60), Δ=1): Top-1/total: 60 / 131 ≈ 0.4580152671755725
(|σ|∈[50, 60), Δ=2): Top-1/total: 14 / 139 ≈ 0.10071942446043165
(|σ|∈[50, 60), Δ=3): Top-1/total: 5 / 67 ≈ 0.07462686567164178
(|σ|∈[60, 70), Δ=1): Top-1/total: 51 / 123 ≈ 0.4146341463414634
(|σ|∈[60, 70), Δ=2): Top-1/total: 4 / 81 ≈ 0.04938271604938271
(|σ|∈[60, 70), Δ=3): Top-1/total: 0 / 51 ≈ 0.0
(|σ|∈[70, 80), Δ=1): Top-1/total: 44 / 109 ≈ 0.4036697247706422
(|σ|∈[70, 80), Δ=2): Top-1/total: 4 / 73 ≈ 0.0547945205479452
(|σ|∈[70, 80), Δ=3): Top-1/total: 1 / 34 ≈ 0.029411764705882353

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 75 / 191 ≈ 0.39267015706806285
|σ|∈[10, 20): Top-1/total: 155 / 363 ≈ 0.42699724517906334
|σ|∈[20, 30): Top-1/total: 141 / 362 ≈ 0.38950276243093923
|σ|∈[30, 40): Top-1/total: 141 / 393 ≈ 0.35877862595419846
|σ|∈[40, 50): Top-1/total: 140 / 342 ≈ 0.4093567251461988
|σ|∈[50, 60): Top-1/total: 143 / 337 ≈ 0.42433234421364985
|σ|∈[60, 70): Top-1/total: 111 / 255 ≈ 0.43529411764705883
|σ|∈[70, 80): Top-1/total: 105 / 216 ≈ 0.4861111111111111
Δ(1)= Top-1/total: 757 / 1007 ≈ 0.7517378351539226
Δ(2)= Top-1/total: 208 / 930 ≈ 0.22365591397849463
Δ(3)= Top-1/total: 46 / 522 ≈ 0.08812260536398467
(|σ|∈[0, 10), Δ=1): Top-1/total: 56 / 91 ≈ 0.6153846153846154
(|σ|∈[0, 10), Δ=2): Top-1/total: 16 / 68 ≈ 0.23529411764705882
(|σ|∈[0, 10), Δ=3): Top-1/total: 3 / 32 ≈ 0.09375
(|σ|∈[10, 20), Δ=1): Top-1/total: 104 / 138 ≈ 0.7536231884057971
(|σ|∈[10, 20), Δ=2): Top-1/total: 43 / 141 ≈ 0.3049645390070922
(|σ|∈[10, 20), Δ=3): Top-1/total: 8 / 84 ≈ 0.09523809523809523
(|σ|∈[20, 30), Δ=1): Top-1/total: 105 / 135 ≈ 0.7777777777777778
(|σ|∈[20, 30), Δ=2): Top-1/total: 29 / 142 ≈ 0.20422535211267606
(|σ|∈[20, 30), Δ=3): Top-1/total: 7 / 85 ≈ 0.08235294117647059
(|σ|∈[30, 40), Δ=1): Top-1/total: 106 / 139 ≈ 0.762589928057554
(|σ|∈[30, 40), Δ=2): Top-1/total: 29 / 146 ≈ 0.19863013698630136
(|σ|∈[30, 40), Δ=3): Top-1/total: 6 / 108 ≈ 0.05555555555555555
(|σ|∈[40, 50), Δ=1): Top-1/total: 103 / 141 ≈ 0.7304964539007093
(|σ|∈[40, 50), Δ=2): Top-1/total: 33 / 140 ≈ 0.2357142857142857
(|σ|∈[40, 50), Δ=3): Top-1/total: 4 / 61 ≈ 0.06557377049180328
(|σ|∈[50, 60), Δ=1): Top-1/total: 104 / 131 ≈ 0.7938931297709924
(|σ|∈[50, 60), Δ=2): Top-1/total: 28 / 139 ≈ 0.2014388489208633
(|σ|∈[50, 60), Δ=3): Top-1/total: 11 / 67 ≈ 0.16417910447761194
(|σ|∈[60, 70), Δ=1): Top-1/total: 92 / 123 ≈ 0.7479674796747967
(|σ|∈[60, 70), Δ=2): Top-1/total: 15 / 81 ≈ 0.18518518518518517
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 51 ≈ 0.0784313725490196
(|σ|∈[70, 80), Δ=1): Top-1/total: 87 / 109 ≈ 0.7981651376146789
(|σ|∈[70, 80), Δ=2): Top-1/total: 15 / 73 ≈ 0.2054794520547945
(|σ|∈[70, 80), Δ=3): Top-1/total: 3 / 34 ≈ 0.08823529411764706


/**
             Unsupervised                                      Supervised

 Precision@1                                   Precision@1
 Δ(1)= Top-1/total: 579 / 1340 ≈ 0.4320895     Δ(1)= Top-1/total: 390 / 1007 ≈ 0.38728
 Δ(2)= Top-1/total: 71 / 1224 ≈ 0.05800653     Δ(2)= Top-1/total: 48 / 930 ≈ 0.0516129
 Δ(3)= Top-1/total: 21 / 654 ≈ 0.032110091     Δ(3)= Top-1/total: 10 / 522 ≈ 0.0191570

 Precision@All                                 Precision@All
 Δ(1)= Top-1/total: 1174 / 1340 ≈ 0.876119     Δ(1)= Top-N/total: 757 / 1007 ≈ 0.75173
 Δ(2)= Top-1/total: 377 / 1224 ≈ 0.3080065     Δ(2)= Top-N/total: 208 / 930 ≈ 0.223655
 Δ(3)= Top-1/total: 75 / 654 ≈ 0.114678899     Δ(3)= Top-N/total: 46 / 522 ≈ 0.0881226
 */

w/ LLM unsupervised denoiser

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 54 / 246 ≈ 0.21951219512195122
|σ|∈[10, 20): Top-1/total: 94 / 494 ≈ 0.1902834008097166
|σ|∈[20, 30): Top-1/total: 81 / 473 ≈ 0.17124735729386892
|σ|∈[30, 40): Top-1/total: 97 / 472 ≈ 0.2055084745762712
|σ|∈[40, 50): Top-1/total: 87 / 457 ≈ 0.19037199124726478
|σ|∈[50, 60): Top-1/total: 102 / 437 ≈ 0.2334096109839817
|σ|∈[60, 70): Top-1/total: 76 / 349 ≈ 0.2177650429799427
|σ|∈[70, 80): Top-1/total: 80 / 290 ≈ 0.27586206896551724
Δ(1)= Top-1/total: 579 / 1340 ≈ 0.43208955223880596
Δ(2)= Top-1/total: 71 / 1224 ≈ 0.05800653594771242
Δ(3)= Top-1/total: 21 / 654 ≈ 0.03211009174311927
(|σ|∈[0, 10), Δ=1): Top-1/total: 51 / 116 ≈ 0.4396551724137931
(|σ|∈[0, 10), Δ=2): Top-1/total: 2 / 97 ≈ 0.020618556701030927
(|σ|∈[0, 10), Δ=3): Top-1/total: 1 / 33 ≈ 0.030303030303030304
(|σ|∈[10, 20), Δ=1): Top-1/total: 82 / 192 ≈ 0.4270833333333333
(|σ|∈[10, 20), Δ=2): Top-1/total: 9 / 187 ≈ 0.0481283422459893
(|σ|∈[10, 20), Δ=3): Top-1/total: 3 / 115 ≈ 0.02608695652173913
(|σ|∈[20, 30), Δ=1): Top-1/total: 71 / 177 ≈ 0.4011299435028249
(|σ|∈[20, 30), Δ=2): Top-1/total: 8 / 185 ≈ 0.043243243243243246
(|σ|∈[20, 30), Δ=3): Top-1/total: 2 / 111 ≈ 0.018018018018018018
(|σ|∈[30, 40), Δ=1): Top-1/total: 82 / 174 ≈ 0.47126436781609193
(|σ|∈[30, 40), Δ=2): Top-1/total: 14 / 185 ≈ 0.07567567567567568
(|σ|∈[30, 40), Δ=3): Top-1/total: 1 / 113 ≈ 0.008849557522123894
(|σ|∈[40, 50), Δ=1): Top-1/total: 77 / 185 ≈ 0.41621621621621624
(|σ|∈[40, 50), Δ=2): Top-1/total: 10 / 188 ≈ 0.05319148936170213
(|σ|∈[40, 50), Δ=3): Top-1/total: 0 / 84 ≈ 0.0
(|σ|∈[50, 60), Δ=1): Top-1/total: 75 / 184 ≈ 0.4076086956521739
(|σ|∈[50, 60), Δ=2): Top-1/total: 17 / 164 ≈ 0.10365853658536585
(|σ|∈[50, 60), Δ=3): Top-1/total: 10 / 89 ≈ 0.11235955056179775
(|σ|∈[60, 70), Δ=1): Top-1/total: 67 / 157 ≈ 0.4267515923566879
(|σ|∈[60, 70), Δ=2): Top-1/total: 7 / 127 ≈ 0.05511811023622047
(|σ|∈[60, 70), Δ=3): Top-1/total: 2 / 65 ≈ 0.03076923076923077
(|σ|∈[70, 80), Δ=1): Top-1/total: 74 / 155 ≈ 0.4774193548387097
(|σ|∈[70, 80), Δ=2): Top-1/total: 4 / 91 ≈ 0.04395604395604396
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 44 ≈ 0.045454545454545456

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 160 / 246 ≈ 0.6504065040650406
|σ|∈[10, 20): Top-1/total: 262 / 494 ≈ 0.5303643724696356
|σ|∈[20, 30): Top-1/total: 215 / 473 ≈ 0.45454545454545453
|σ|∈[30, 40): Top-1/total: 211 / 472 ≈ 0.4470338983050847
|σ|∈[40, 50): Top-1/total: 215 / 457 ≈ 0.47045951859956237
|σ|∈[50, 60): Top-1/total: 222 / 437 ≈ 0.5080091533180778
|σ|∈[60, 70): Top-1/total: 174 / 349 ≈ 0.498567335243553
|σ|∈[70, 80): Top-1/total: 167 / 290 ≈ 0.5758620689655173
Δ(1)= Top-1/total: 1174 / 1340 ≈ 0.8761194029850746
Δ(2)= Top-1/total: 377 / 1224 ≈ 0.30800653594771243
Δ(3)= Top-1/total: 75 / 654 ≈ 0.11467889908256881
(|σ|∈[0, 10), Δ=1): Top-1/total: 99 / 116 ≈ 0.853448275862069
(|σ|∈[0, 10), Δ=2): Top-1/total: 50 / 97 ≈ 0.5154639175257731
(|σ|∈[0, 10), Δ=3): Top-1/total: 11 / 33 ≈ 0.3333333333333333
(|σ|∈[10, 20), Δ=1): Top-1/total: 180 / 192 ≈ 0.9375
(|σ|∈[10, 20), Δ=2): Top-1/total: 70 / 187 ≈ 0.37433155080213903
(|σ|∈[10, 20), Δ=3): Top-1/total: 12 / 115 ≈ 0.10434782608695652
(|σ|∈[20, 30), Δ=1): Top-1/total: 154 / 177 ≈ 0.8700564971751412
(|σ|∈[20, 30), Δ=2): Top-1/total: 52 / 185 ≈ 0.2810810810810811
(|σ|∈[20, 30), Δ=3): Top-1/total: 9 / 111 ≈ 0.08108108108108109
(|σ|∈[30, 40), Δ=1): Top-1/total: 151 / 174 ≈ 0.867816091954023
(|σ|∈[30, 40), Δ=2): Top-1/total: 54 / 185 ≈ 0.2918918918918919
(|σ|∈[30, 40), Δ=3): Top-1/total: 6 / 113 ≈ 0.05309734513274336
(|σ|∈[40, 50), Δ=1): Top-1/total: 162 / 185 ≈ 0.8756756756756757
(|σ|∈[40, 50), Δ=2): Top-1/total: 47 / 188 ≈ 0.25
(|σ|∈[40, 50), Δ=3): Top-1/total: 6 / 84 ≈ 0.07142857142857142
(|σ|∈[50, 60), Δ=1): Top-1/total: 157 / 184 ≈ 0.8532608695652174
(|σ|∈[50, 60), Δ=2): Top-1/total: 48 / 164 ≈ 0.2926829268292683
(|σ|∈[50, 60), Δ=3): Top-1/total: 17 / 89 ≈ 0.19101123595505617
(|σ|∈[60, 70), Δ=1): Top-1/total: 131 / 157 ≈ 0.8343949044585988
(|σ|∈[60, 70), Δ=2): Top-1/total: 33 / 127 ≈ 0.25984251968503935
(|σ|∈[60, 70), Δ=3): Top-1/total: 10 / 65 ≈ 0.15384615384615385
(|σ|∈[70, 80), Δ=1): Top-1/total: 140 / 155 ≈ 0.9032258064516129
(|σ|∈[70, 80), Δ=2): Top-1/total: 23 / 91 ≈ 0.25274725274725274
(|σ|∈[70, 80), Δ=3): Top-1/total: 4 / 44 ≈ 0.09090909090909091

w/ parallel beam search

Precision@1
===========
(|σ|∈[0, 10), Δ=1): Top-1/total: 99 / 184 ≈ 0.5380434782608695
(|σ|∈[0, 10), Δ=2): Top-1/total: 44 / 133 ≈ 0.3308270676691729
(|σ|∈[0, 10), Δ=3): Top-1/total: 9 / 58 ≈ 0.15517241379310345
(|σ|∈[10, 20), Δ=1): Top-1/total: 145 / 300 ≈ 0.48333333333333334
(|σ|∈[10, 20), Δ=2): Top-1/total: 88 / 298 ≈ 0.2953020134228188
(|σ|∈[10, 20), Δ=3): Top-1/total: 31 / 174 ≈ 0.1781609195402299
(|σ|∈[20, 30), Δ=1): Top-1/total: 120 / 289 ≈ 0.41522491349480967
(|σ|∈[20, 30), Δ=2): Top-1/total: 81 / 290 ≈ 0.2793103448275862
(|σ|∈[20, 30), Δ=3): Top-1/total: 37 / 182 ≈ 0.2032967032967033
(|σ|∈[30, 40), Δ=1): Top-1/total: 143 / 289 ≈ 0.49480968858131485
(|σ|∈[30, 40), Δ=2): Top-1/total: 67 / 290 ≈ 0.23103448275862068
(|σ|∈[30, 40), Δ=3): Top-1/total: 33 / 192 ≈ 0.171875
(|σ|∈[40, 50), Δ=1): Top-1/total: 151 / 289 ≈ 0.5224913494809689
(|σ|∈[40, 50), Δ=2): Top-1/total: 52 / 288 ≈ 0.18055555555555555
(|σ|∈[40, 50), Δ=3): Top-1/total: 18 / 138 ≈ 0.13043478260869565
(|σ|∈[50, 60), Δ=1): Top-1/total: 140 / 287 ≈ 0.4878048780487805
(|σ|∈[50, 60), Δ=2): Top-1/total: 62 / 265 ≈ 0.2339622641509434
(|σ|∈[50, 60), Δ=3): Top-1/total: 17 / 132 ≈ 0.12878787878787878
(|σ|∈[60, 70), Δ=1): Top-1/total: 167 / 274 ≈ 0.6094890510948905
(|σ|∈[60, 70), Δ=2): Top-1/total: 42 / 197 ≈ 0.2131979695431472
(|σ|∈[60, 70), Δ=3): Top-1/total: 16 / 110 ≈ 0.14545454545454545
(|σ|∈[70, 80), Δ=1): Top-1/total: 152 / 244 ≈ 0.6229508196721312
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 145 ≈ 0.23448275862068965
(|σ|∈[70, 80), Δ=3): Top-1/total: 12 / 78 ≈ 0.15384615384615385

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 374 / 375 ≈ 0.9973333333333333
|σ|∈[10, 20): Top-1/total: 760 / 772 ≈ 0.9844559585492227
|σ|∈[20, 30): Top-1/total: 740 / 761 ≈ 0.9724047306176085
|σ|∈[30, 40): Top-1/total: 749 / 771 ≈ 0.9714656290531777
|σ|∈[40, 50): Top-1/total: 694 / 715 ≈ 0.9706293706293706
|σ|∈[50, 60): Top-1/total: 653 / 684 ≈ 0.9546783625730995
|σ|∈[60, 70): Top-1/total: 556 / 581 ≈ 0.9569707401032702
|σ|∈[70, 80): Top-1/total: 450 / 467 ≈ 0.9635974304068522
Δ(1)= Top-1/total: 2134 / 2156 ≈ 0.9897959183673469
Δ(2)= Top-1/total: 1904 / 1906 ≈ 0.9989506820566632
Δ(3)= Top-1/total: 938 / 1064 ≈ 0.881578947368421
(|σ|∈[0, 10), Δ=1): Top-1/total: 183 / 184 ≈ 0.9945652173913043
(|σ|∈[0, 10), Δ=2): Top-1/total: 133 / 133 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 58 / 58 ≈ 1.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 299 / 300 ≈ 0.9966666666666667
(|σ|∈[10, 20), Δ=2): Top-1/total: 297 / 298 ≈ 0.9966442953020134
(|σ|∈[10, 20), Δ=3): Top-1/total: 164 / 174 ≈ 0.9425287356321839
(|σ|∈[20, 30), Δ=1): Top-1/total: 288 / 289 ≈ 0.9965397923875432
(|σ|∈[20, 30), Δ=2): Top-1/total: 290 / 290 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 162 / 182 ≈ 0.8901098901098901
(|σ|∈[30, 40), Δ=1): Top-1/total: 287 / 289 ≈ 0.9930795847750865
(|σ|∈[30, 40), Δ=2): Top-1/total: 290 / 290 ≈ 1.0
(|σ|∈[30, 40), Δ=3): Top-1/total: 172 / 192 ≈ 0.8958333333333334
(|σ|∈[40, 50), Δ=1): Top-1/total: 283 / 289 ≈ 0.9792387543252595
(|σ|∈[40, 50), Δ=2): Top-1/total: 288 / 288 ≈ 1.0
(|σ|∈[40, 50), Δ=3): Top-1/total: 123 / 138 ≈ 0.8913043478260869
(|σ|∈[50, 60), Δ=1): Top-1/total: 283 / 287 ≈ 0.9860627177700348
(|σ|∈[50, 60), Δ=2): Top-1/total: 265 / 265 ≈ 1.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 105 / 132 ≈ 0.7954545454545454
(|σ|∈[60, 70), Δ=1): Top-1/total: 270 / 274 ≈ 0.9854014598540146
(|σ|∈[60, 70), Δ=2): Top-1/total: 196 / 197 ≈ 0.9949238578680203
(|σ|∈[60, 70), Δ=3): Top-1/total: 90 / 110 ≈ 0.8181818181818182
(|σ|∈[70, 80), Δ=1): Top-1/total: 241 / 244 ≈ 0.9877049180327869
(|σ|∈[70, 80), Δ=2): Top-1/total: 145 / 145 ≈ 1.0
(|σ|∈[70, 80), Δ=3): Top-1/total: 64 / 78 ≈ 0.8205128205128205

w/ horizontal states + NT compatibility filter

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 152 / 375 ≈ 0.4053333333333333
|σ|∈[10, 20): Top-1/total: 245 / 769 ≈ 0.31859557867360205
|σ|∈[20, 30): Top-1/total: 245 / 763 ≈ 0.3211009174311927
|σ|∈[30, 40): Top-1/total: 228 / 776 ≈ 0.29381443298969073
|σ|∈[40, 50): Top-1/total: 221 / 720 ≈ 0.30694444444444446
|σ|∈[50, 60): Top-1/total: 221 / 684 ≈ 0.3230994152046784
|σ|∈[60, 70): Top-1/total: 217 / 582 ≈ 0.37285223367697595
|σ|∈[70, 80): Top-1/total: 196 / 467 ≈ 0.4197002141327623
Δ(1)= Top-1/total: 1103 / 2159 ≈ 0.5108846688281612
Δ(2)= Top-1/total: 461 / 1913 ≈ 0.24098274960794563
Δ(3)= Top-1/total: 161 / 1064 ≈ 0.1513157894736842
(|σ|∈[0, 10), Δ=1): Top-1/total: 99 / 184 ≈ 0.5380434782608695
(|σ|∈[0, 10), Δ=2): Top-1/total: 44 / 133 ≈ 0.3308270676691729
(|σ|∈[0, 10), Δ=3): Top-1/total: 9 / 58 ≈ 0.15517241379310345
(|σ|∈[10, 20), Δ=1): Top-1/total: 123 / 297 ≈ 0.41414141414141414
(|σ|∈[10, 20), Δ=2): Top-1/total: 91 / 298 ≈ 0.3053691275167785
(|σ|∈[10, 20), Δ=3): Top-1/total: 31 / 174 ≈ 0.1781609195402299
(|σ|∈[20, 30), Δ=1): Top-1/total: 135 / 288 ≈ 0.46875
(|σ|∈[20, 30), Δ=2): Top-1/total: 75 / 293 ≈ 0.25597269624573377
(|σ|∈[20, 30), Δ=3): Top-1/total: 35 / 182 ≈ 0.19230769230769232
(|σ|∈[30, 40), Δ=1): Top-1/total: 133 / 293 ≈ 0.4539249146757679
(|σ|∈[30, 40), Δ=2): Top-1/total: 64 / 291 ≈ 0.21993127147766323
(|σ|∈[30, 40), Δ=3): Top-1/total: 31 / 192 ≈ 0.16145833333333334
(|σ|∈[40, 50), Δ=1): Top-1/total: 156 / 291 ≈ 0.5360824742268041
(|σ|∈[40, 50), Δ=2): Top-1/total: 49 / 291 ≈ 0.16838487972508592
(|σ|∈[40, 50), Δ=3): Top-1/total: 16 / 138 ≈ 0.11594202898550725
(|σ|∈[50, 60), Δ=1): Top-1/total: 144 / 287 ≈ 0.5017421602787456
(|σ|∈[50, 60), Δ=2): Top-1/total: 62 / 265 ≈ 0.2339622641509434
(|σ|∈[50, 60), Δ=3): Top-1/total: 15 / 132 ≈ 0.11363636363636363
(|σ|∈[60, 70), Δ=1): Top-1/total: 161 / 275 ≈ 0.5854545454545454
(|σ|∈[60, 70), Δ=2): Top-1/total: 42 / 197 ≈ 0.2131979695431472
(|σ|∈[60, 70), Δ=3): Top-1/total: 14 / 110 ≈ 0.12727272727272726
(|σ|∈[70, 80), Δ=1): Top-1/total: 152 / 244 ≈ 0.6229508196721312
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 145 ≈ 0.23448275862068965
(|σ|∈[70, 80), Δ=3): Top-1/total: 10 / 78 ≈ 0.1282051282051282

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 374 / 375 ≈ 0.9973333333333333
|σ|∈[10, 20): Top-1/total: 763 / 769 ≈ 0.9921976592977894
|σ|∈[20, 30): Top-1/total: 740 / 763 ≈ 0.9698558322411533
|σ|∈[30, 40): Top-1/total: 731 / 776 ≈ 0.9420103092783505
|σ|∈[40, 50): Top-1/total: 664 / 720 ≈ 0.9222222222222223
|σ|∈[50, 60): Top-1/total: 620 / 684 ≈ 0.9064327485380117
|σ|∈[60, 70): Top-1/total: 525 / 582 ≈ 0.9020618556701031
|σ|∈[70, 80): Top-1/total: 421 / 467 ≈ 0.9014989293361885
Δ(1)= Top-1/total: 2138 / 2159 ≈ 0.9902732746641963
Δ(2)= Top-1/total: 1909 / 1913 ≈ 0.9979090433873498
Δ(3)= Top-1/total: 791 / 1064 ≈ 0.743421052631579
(|σ|∈[0, 10), Δ=1): Top-1/total: 183 / 184 ≈ 0.9945652173913043
(|σ|∈[0, 10), Δ=2): Top-1/total: 133 / 133 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 58 / 58 ≈ 1.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 296 / 297 ≈ 0.9966329966329966
(|σ|∈[10, 20), Δ=2): Top-1/total: 297 / 298 ≈ 0.9966442953020134
(|σ|∈[10, 20), Δ=3): Top-1/total: 170 / 174 ≈ 0.9770114942528736
(|σ|∈[20, 30), Δ=1): Top-1/total: 286 / 288 ≈ 0.9930555555555556
(|σ|∈[20, 30), Δ=2): Top-1/total: 293 / 293 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 161 / 182 ≈ 0.8846153846153846
(|σ|∈[30, 40), Δ=1): Top-1/total: 291 / 293 ≈ 0.9931740614334471
(|σ|∈[30, 40), Δ=2): Top-1/total: 291 / 291 ≈ 1.0
(|σ|∈[30, 40), Δ=3): Top-1/total: 149 / 192 ≈ 0.7760416666666666
(|σ|∈[40, 50), Δ=1): Top-1/total: 287 / 291 ≈ 0.9862542955326461
(|σ|∈[40, 50), Δ=2): Top-1/total: 291 / 291 ≈ 1.0
(|σ|∈[40, 50), Δ=3): Top-1/total: 86 / 138 ≈ 0.6231884057971014
(|σ|∈[50, 60), Δ=1): Top-1/total: 283 / 287 ≈ 0.9860627177700348
(|σ|∈[50, 60), Δ=2): Top-1/total: 265 / 265 ≈ 1.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 72 / 132 ≈ 0.5454545454545454
(|σ|∈[60, 70), Δ=1): Top-1/total: 271 / 275 ≈ 0.9854545454545455
(|σ|∈[60, 70), Δ=2): Top-1/total: 196 / 197 ≈ 0.9949238578680203
(|σ|∈[60, 70), Δ=3): Top-1/total: 58 / 110 ≈ 0.5272727272727272
(|σ|∈[70, 80), Δ=1): Top-1/total: 241 / 244 ≈ 0.9877049180327869
(|σ|∈[70, 80), Δ=2): Top-1/total: 143 / 145 ≈ 0.9862068965517241
(|σ|∈[70, 80), Δ=3): Top-1/total: 37 / 78 ≈ 0.47435897435897434

w/ Lev-NFA multiedit pruning

Precision@1
===========
(|σ|∈[0, 10), Δ=1): Top-1/total: 54 / 98 ≈ 0.5510204081632653
(|σ|∈[0, 10), Δ=2): Top-1/total: 31 / 100 ≈ 0.31
(|σ|∈[0, 10), Δ=3): Top-1/total: 9 / 58 ≈ 0.15517241379310345
(|σ|∈[10, 20), Δ=1): Top-1/total: 49 / 98 ≈ 0.5
(|σ|∈[10, 20), Δ=2): Top-1/total: 28 / 100 ≈ 0.28
(|σ|∈[10, 20), Δ=3): Top-1/total: 15 / 99 ≈ 0.15151515151515152
(|σ|∈[20, 30), Δ=1): Top-1/total: 44 / 98 ≈ 0.4489795918367347
(|σ|∈[20, 30), Δ=2): Top-1/total: 19 / 97 ≈ 0.1958762886597938
(|σ|∈[20, 30), Δ=3): Top-1/total: 20 / 100 ≈ 0.2
(|σ|∈[30, 40), Δ=1): Top-1/total: 52 / 97 ≈ 0.5360824742268041
(|σ|∈[30, 40), Δ=2): Top-1/total: 24 / 98 ≈ 0.24489795918367346
(|σ|∈[30, 40), Δ=3): Top-1/total: 16 / 99 ≈ 0.16161616161616163
(|σ|∈[40, 50), Δ=1): Top-1/total: 43 / 97 ≈ 0.44329896907216493
(|σ|∈[40, 50), Δ=2): Top-1/total: 13 / 98 ≈ 0.1326530612244898
(|σ|∈[40, 50), Δ=3): Top-1/total: 12 / 95 ≈ 0.12631578947368421
(|σ|∈[50, 60), Δ=1): Top-1/total: 43 / 97 ≈ 0.44329896907216493
(|σ|∈[50, 60), Δ=2): Top-1/total: 24 / 94 ≈ 0.2553191489361702
(|σ|∈[50, 60), Δ=3): Top-1/total: 13 / 100 ≈ 0.13
(|σ|∈[60, 70), Δ=1): Top-1/total: 61 / 92 ≈ 0.6630434782608695
(|σ|∈[60, 70), Δ=2): Top-1/total: 17 / 96 ≈ 0.17708333333333334
(|σ|∈[60, 70), Δ=3): Top-1/total: 13 / 93 ≈ 0.13978494623655913
(|σ|∈[70, 80), Δ=1): Top-1/total: 54 / 90 ≈ 0.6
(|σ|∈[70, 80), Δ=2): Top-1/total: 20 / 94 ≈ 0.2127659574468085
(|σ|∈[70, 80), Δ=3): Top-1/total: 9 / 81 ≈ 0.1111111111111111

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 254 / 256 ≈ 0.9921875
|σ|∈[10, 20): Top-1/total: 294 / 297 ≈ 0.98989898989899
|σ|∈[20, 30): Top-1/total: 280 / 295 ≈ 0.9491525423728814
|σ|∈[30, 40): Top-1/total: 262 / 294 ≈ 0.891156462585034
|σ|∈[40, 50): Top-1/total: 247 / 290 ≈ 0.8517241379310345
|σ|∈[50, 60): Top-1/total: 242 / 291 ≈ 0.8316151202749141
|σ|∈[60, 70): Top-1/total: 231 / 281 ≈ 0.8220640569395018
|σ|∈[70, 80): Top-1/total: 208 / 265 ≈ 0.7849056603773585
|σ|∈[80, 90): Top-1/total: 17 / 20 ≈ 0.85
Δ(1)= Top-1/total: 769 / 778 ≈ 0.9884318766066839
Δ(2)= Top-1/total: 774 / 782 ≈ 0.989769820971867
Δ(3)= Top-1/total: 492 / 729 ≈ 0.6748971193415638
(|σ|∈[0, 10), Δ=1): Top-1/total: 97 / 98 ≈ 0.9897959183673469
(|σ|∈[0, 10), Δ=2): Top-1/total: 99 / 100 ≈ 0.99
(|σ|∈[0, 10), Δ=3): Top-1/total: 58 / 58 ≈ 1.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 98 / 98 ≈ 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 100 / 100 ≈ 1.0
(|σ|∈[10, 20), Δ=3): Top-1/total: 96 / 99 ≈ 0.9696969696969697
(|σ|∈[20, 30), Δ=1): Top-1/total: 98 / 98 ≈ 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 97 / 97 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 85 / 100 ≈ 0.85
(|σ|∈[30, 40), Δ=1): Top-1/total: 95 / 97 ≈ 0.979381443298969
(|σ|∈[30, 40), Δ=2): Top-1/total: 97 / 98 ≈ 0.9897959183673469
(|σ|∈[30, 40), Δ=3): Top-1/total: 70 / 99 ≈ 0.7070707070707071
(|σ|∈[40, 50), Δ=1): Top-1/total: 96 / 97 ≈ 0.9896907216494846
(|σ|∈[40, 50), Δ=2): Top-1/total: 98 / 98 ≈ 1.0
(|σ|∈[40, 50), Δ=3): Top-1/total: 53 / 95 ≈ 0.5578947368421052
(|σ|∈[50, 60), Δ=1): Top-1/total: 95 / 97 ≈ 0.979381443298969
(|σ|∈[50, 60), Δ=2): Top-1/total: 93 / 94 ≈ 0.9893617021276596
(|σ|∈[50, 60), Δ=3): Top-1/total: 54 / 100 ≈ 0.54
(|σ|∈[60, 70), Δ=1): Top-1/total: 92 / 92 ≈ 1.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 94 / 96 ≈ 0.9791666666666666
(|σ|∈[60, 70), Δ=3): Top-1/total: 45 / 93 ≈ 0.4838709677419355
(|σ|∈[70, 80), Δ=1): Top-1/total: 88 / 90 ≈ 0.9777777777777777
(|σ|∈[70, 80), Δ=2): Top-1/total: 91 / 94 ≈ 0.9680851063829787
(|σ|∈[70, 80), Δ=3): Top-1/total: 29 / 81 ≈ 0.35802469135802467

// w/ language edit distance radius + markov chain decoder

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 98 / 341 ≈ 0.2873900293255132
|σ|∈[10, 20): Top-1/total: 170 / 697 ≈ 0.24390243902439024
|σ|∈[20, 30): Top-1/total: 179 / 692 ≈ 0.2586705202312139
|σ|∈[30, 40): Top-1/total: 184 / 711 ≈ 0.2587904360056259
|σ|∈[40, 50): Top-1/total: 182 / 655 ≈ 0.2778625954198473
|σ|∈[50, 60): Top-1/total: 185 / 630 ≈ 0.29365079365079366
|σ|∈[60, 70): Top-1/total: 177 / 527 ≈ 0.33586337760910817
|σ|∈[70, 80): Top-1/total: 160 / 414 ≈ 0.3864734299516908
Δ(1)= Top-1/total: 1044 / 1959 ≈ 0.5329249617151608
Δ(2)= Top-1/total: 242 / 1738 ≈ 0.13924050632911392
Δ(3)= Top-1/total: 49 / 970 ≈ 0.050515463917525774
(|σ|∈[0, 10), Δ=1): Top-1/total: 90 / 167 ≈ 0.5389221556886228
(|σ|∈[0, 10), Δ=2): Top-1/total: 8 / 119 ≈ 0.06722689075630252
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 55 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 129 / 276 ≈ 0.4673913043478261
(|σ|∈[10, 20), Δ=2): Top-1/total: 34 / 266 ≈ 0.12781954887218044
(|σ|∈[10, 20), Δ=3): Top-1/total: 7 / 155 ≈ 0.04516129032258064
(|σ|∈[20, 30), Δ=1): Top-1/total: 129 / 262 ≈ 0.49236641221374045
(|σ|∈[20, 30), Δ=2): Top-1/total: 40 / 266 ≈ 0.15037593984962405
(|σ|∈[20, 30), Δ=3): Top-1/total: 10 / 164 ≈ 0.06097560975609756
(|σ|∈[30, 40), Δ=1): Top-1/total: 132 / 270 ≈ 0.4888888888888889
(|σ|∈[30, 40), Δ=2): Top-1/total: 43 / 262 ≈ 0.16412213740458015
(|σ|∈[30, 40), Δ=3): Top-1/total: 9 / 179 ≈ 0.05027932960893855
(|σ|∈[40, 50), Δ=1): Top-1/total: 142 / 259 ≈ 0.5482625482625483
(|σ|∈[40, 50), Δ=2): Top-1/total: 32 / 274 ≈ 0.11678832116788321
(|σ|∈[40, 50), Δ=3): Top-1/total: 8 / 122 ≈ 0.06557377049180328
(|σ|∈[50, 60), Δ=1): Top-1/total: 135 / 263 ≈ 0.5133079847908745
(|σ|∈[50, 60), Δ=2): Top-1/total: 41 / 244 ≈ 0.1680327868852459
(|σ|∈[50, 60), Δ=3): Top-1/total: 9 / 123 ≈ 0.07317073170731707
(|σ|∈[60, 70), Δ=1): Top-1/total: 149 / 245 ≈ 0.6081632653061224
(|σ|∈[60, 70), Δ=2): Top-1/total: 24 / 180 ≈ 0.13333333333333333
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 102 ≈ 0.0392156862745098
(|σ|∈[70, 80), Δ=1): Top-1/total: 138 / 217 ≈ 0.6359447004608295
(|σ|∈[70, 80), Δ=2): Top-1/total: 20 / 127 ≈ 0.15748031496062992
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 70 ≈ 0.02857142857142857

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 178 / 341 ≈ 0.5219941348973607
|σ|∈[10, 20): Top-1/total: 342 / 697 ≈ 0.49067431850789095
|σ|∈[20, 30): Top-1/total: 342 / 692 ≈ 0.49421965317919075
|σ|∈[30, 40): Top-1/total: 367 / 711 ≈ 0.5161744022503516
|σ|∈[40, 50): Top-1/total: 347 / 655 ≈ 0.5297709923664122
|σ|∈[50, 60): Top-1/total: 362 / 630 ≈ 0.5746031746031746
|σ|∈[60, 70): Top-1/total: 307 / 527 ≈ 0.5825426944971537
|σ|∈[70, 80): Top-1/total: 256 / 414 ≈ 0.6183574879227053
Δ(1)= Top-1/total: 1941 / 1959 ≈ 0.9908116385911179
Δ(2)= Top-1/total: 444 / 1738 ≈ 0.2554660529344074
Δ(3)= Top-1/total: 116 / 970 ≈ 0.11958762886597939
(|σ|∈[0, 10), Δ=1): Top-1/total: 167 / 167 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 10 / 119 ≈ 0.08403361344537816
(|σ|∈[0, 10), Δ=3): Top-1/total: 1 / 55 ≈ 0.01818181818181818
(|σ|∈[10, 20), Δ=1): Top-1/total: 275 / 276 ≈ 0.9963768115942029
(|σ|∈[10, 20), Δ=2): Top-1/total: 54 / 266 ≈ 0.20300751879699247
(|σ|∈[10, 20), Δ=3): Top-1/total: 13 / 155 ≈ 0.08387096774193549
(|σ|∈[20, 30), Δ=1): Top-1/total: 261 / 262 ≈ 0.9961832061068703
(|σ|∈[20, 30), Δ=2): Top-1/total: 64 / 266 ≈ 0.24060150375939848
(|σ|∈[20, 30), Δ=3): Top-1/total: 17 / 164 ≈ 0.10365853658536585
(|σ|∈[30, 40), Δ=1): Top-1/total: 268 / 270 ≈ 0.9925925925925926
(|σ|∈[30, 40), Δ=2): Top-1/total: 78 / 262 ≈ 0.29770992366412213
(|σ|∈[30, 40), Δ=3): Top-1/total: 21 / 179 ≈ 0.11731843575418995
(|σ|∈[40, 50), Δ=1): Top-1/total: 255 / 259 ≈ 0.9845559845559846
(|σ|∈[40, 50), Δ=2): Top-1/total: 74 / 274 ≈ 0.27007299270072993
(|σ|∈[40, 50), Δ=3): Top-1/total: 18 / 122 ≈ 0.14754098360655737
(|σ|∈[50, 60), Δ=1): Top-1/total: 261 / 263 ≈ 0.9923954372623575
(|σ|∈[50, 60), Δ=2): Top-1/total: 78 / 244 ≈ 0.319672131147541
(|σ|∈[50, 60), Δ=3): Top-1/total: 23 / 123 ≈ 0.18699186991869918
(|σ|∈[60, 70), Δ=1): Top-1/total: 241 / 245 ≈ 0.9836734693877551
(|σ|∈[60, 70), Δ=2): Top-1/total: 51 / 180 ≈ 0.2833333333333333
(|σ|∈[60, 70), Δ=3): Top-1/total: 15 / 102 ≈ 0.14705882352941177
(|σ|∈[70, 80), Δ=1): Top-1/total: 213 / 217 ≈ 0.9815668202764977
(|σ|∈[70, 80), Δ=2): Top-1/total: 35 / 127 ≈ 0.2755905511811024
(|σ|∈[70, 80), Δ=3): Top-1/total: 8 / 70 ≈ 0.11428571428571428
 */