package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.automata.decodeDFA
import ai.hypergraph.kaliningraph.automata.toDFA
import ai.hypergraph.kaliningraph.parsing.NUM_CORES
import ai.hypergraph.kaliningraph.parsing.TermDict
import ai.hypergraph.kaliningraph.parsing.levenshteinAlign
import ai.hypergraph.kaliningraph.parsing.makeLevFSA
import ai.hypergraph.kaliningraph.parsing.maxParsableFragmentB
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.parsing.paintANSIColors
import ai.hypergraph.kaliningraph.parsing.parikhMap
import ai.hypergraph.kaliningraph.parsing.patchSize
import ai.hypergraph.kaliningraph.parsing.sampleDirectlyWOR
import ai.hypergraph.kaliningraph.parsing.sampleDirectlyWORAndScore
import ai.hypergraph.kaliningraph.parsing.summarize
import ai.hypergraph.kaliningraph.parsing.terminals
import ai.hypergraph.kaliningraph.repair.CFG_THRESH
import ai.hypergraph.kaliningraph.repair.MAX_RADIUS
import ai.hypergraph.kaliningraph.repair.MAX_TOKENS
import ai.hypergraph.kaliningraph.repair.MAX_UNIQUE
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.repair.vanillaS2PCFG
import ai.hypergraph.kaliningraph.repair.vanillaS2PCFGWE
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import edu.mcgill.cstk.experiments.probing.MakeMore
import edu.mcgill.cstk.utils.lastGitMessage
import java.io.File
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource
import kotlin.time.measureTimedValue

fun evaluateMatrixBarHillelRepairOnStackOverflow() {
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

  println("Running Matrix Bar-Hillel repair on Python snippets with $NUM_CORES cores")
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
    val langEditDist = FSA.LED(s2pg, brokeStr)
    val levGuess = langEditDist + 1
    val predDist = levGuess

    val levDist = levAlign.patchSize() // True distance, only used for logging purposes
    println("Predicted edit dist: $predDist (true dist: $levDist, LED: $langEditDist)")

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    var levBallSize = 1
    val humanRepairANSI = levenshteinAlign(brokeToks, fixedToks).paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")
    allRate.total++; levRates.getOrPut(levDist) { LBHMetrics() }.total++

    val timedTree = try {
      val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(brokeToks, pad = levGuess)
//    val multiEditBounds = vanillaS2PCFGWE.findMinimalMultiEditBounds(toRepair, monoEditBounds, levDist)
      val fsa = makeLevFSA(brokeToks, levGuess, monoEditBounds).also { levBallSize = it.Q.size }
      measureTimedValue { FSA.intersectPTree(brokeStr, s2pg, levGuess, fsa) }
    }
    catch (e: Exception) {
      println("Encountered error ${e.message} ${allTime.elapsedNow()}):\n$humanRepairANSI\n${e.stackTraceToString()}")
      allRate.error++; levRates.getOrPut(levDist) { LBHMetrics() }.error++
      println(allRate.toString())
      negative.appendText("${brokeToks.size}, $levDist, 0, " +
          "${levBallSize}, 0, ${levAlign.summarize()}\n")

      println()
      println("Precision@1\n===========")
      println(P_1ByLevDist.summarizeLenAndDist())
      println("Precision@All\n=============")
      println(P_AllByLevDist.summarizeLenAndDist())
      println()
      null
    }

    if (timedTree == null) return@forEach
    val pTree = timedTree.value!!
    val intGramSize = pTree.totalProds
    val langSize = pTree.totalTreesStr
    // TODO: Why so many productions? Need to implement PTree.toCFG()
    println("Constructed PTree in ${timedTree.duration} with $intGramSize productions and $langSize words")
    val clock = TimeSource.Monotonic.markNow()
    var totalSamples = 0
    var matchFound = false
    val timeout = (TIMEOUT_MS / 1000).seconds
    var elapsed = clock.elapsedNow().inWholeMilliseconds

//    val dfa = pTree.toDFA(minimize = true)!!
//
//    val dfaRecognized = try { dfa.run(termDict.encode(fixedToks)) } catch (_: Exception) { false }
//    println("∩-DFA ${if (dfaRecognized) "accepted" else "rejected"} human repair! (Total time=${allTime.elapsedNow()})")

//    val rankedResults = dfa.decodeDFA(
//      mc = P_BIFI_PY150,
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
    val rankedResults = pTree.sampleDirectlyWOR { clock.elapsedNow() < timeout }.distinct().map {
      totalSamples++
      if (it == fixedStr) {
        matchFound = true
        println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
        elapsed = clock.elapsedNow().inWholeMilliseconds
      }
      it to P_BIFI_PY150.score(it.tokenizeByWhitespace())
    }
    .toList().sortedBy { it.second }.map { it.first }

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
      println("Drew $totalSamples samples in ${clock.elapsedNow()}/$timeout with $intGramSize prods, " +
//        "${dfa.numStates} states, ${dfa.numberOfTransitions} transitions, " +
          "length-$levDist human repair not found")
      negative.appendText(
        "${brokeToks.size}, $levDist, $totalSamples, ${levBallSize}, " +
            "$intGramSize, $langSize, " +
//          "${dfa.numStates}, ${dfa.numberOfTransitions}, " +
            "${levAlign.summarize()}\n"
      )
    } else {
      val allElapsed = allTime.elapsedNow().inWholeMilliseconds

      allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
      indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
      println("Found length-$levDist repair in $elapsed ms, $allElapsed ms," +
          " $totalSamples samples, $intGramSize prods, $langSize trees, $indexOfTarget rank")//, rank: ${rankedResults.indexOf(fixedTks) + 1} / ${rankedResults.size}")
      allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
//      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
      sampleTimeByLevDist[levDist] = (sampleTimeByLevDist[levDist] ?: 0.0) + elapsed
      println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
      allTimeByLevDist[levDist] = (allTimeByLevDist[levDist] ?: 0.0) + allElapsed
      println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
      samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + totalSamples
      println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
      positive.appendText("${brokeToks.size}, $levDist, $elapsed, $allElapsed, " +
          "$totalSamples, $levBallSize, $intGramSize, $langSize, " +
//          "${dfa.numberOfStates}, ${dfa.numberOfTransitions}, " +
          "$indexOfTarget, ${levAlign.summarize()}\n")
    }

    println()
    println("Precision@1\n===========")
    println(P_1ByLevDist.summarizeLenAndDist())
    println("Precision@All\n=============")
    println(P_AllByLevDist.summarizeLenAndDist())
    println()
  }
}