package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.experiments.probing.charify
import edu.mcgill.cstk.utils.getOutput
import edu.mcgill.cstk.utils.lastGitMessage
import java.io.File
import kotlin.streams.asStream
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

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

//  startWGPUServer()
  evaluateRegexRepairOnStackOverflow()
//  stopWGPUServer()
//  evaluateMatrixBarHillelRepairOnStackOverflow()
//  evaluateBarHillelRepairOnStackOverflow()
//  evaluateSeq2ParseRepair()
//  evaluateBIFIRepair()
//  measureLevenshteinBlanketSize()
//  writeParikhMap()
}

fun evaluateRegexRepairOnStackOverflow() {
  val dataset = sizeAndDistBalancedRepairsUnminimized
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val termDict = TermDict(s2pg.terminals)

  println("Running Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  println("Sampling timeout: $TIMEOUT_MS ms, max tokens: $MAX_TOKENS, max radius: $MAX_RADIUS, max unique: $MAX_UNIQUE, CFG threshold: $CFG_THRESH")
  dataset.first().π2.let { P_BIFI_PY150.score(it.tokenizeByWhitespace()) }

  val latestCommitMessage = lastGitMessage().replace(Regex("[^A-Za-z0-9]"), "_")
    .let { if ("fatal: not a git repository" !in it) it else System.currentTimeMillis().toString() }
  val positiveHeader = "length, lev_dist, led, sample_ms, sample_rerank_ms, cpu_time, gpu_time, total_samples, lang_size, final_rank, orig_rank, gpu_rank\n"
  val title = "regex_bar_hillel"
  val csv = File("data/${title}_results_$latestCommitMessage.csv").also { it.appendText(positiveHeader) }
  println()

  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_10ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_100ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_1000ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_AllByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  fun summarizeRunningStats() {
    println()
    println("Precision@1\n===========")
    println(P_1ByLevDist.summarizeLenAndDist())
    println("Precision@10\n===========")
    println(P_10ByLevDist.summarizeLenAndDist())
    println("Precision@100\n===========")
    println(P_100ByLevDist.summarizeLenAndDist())
    println("Precision@1000\n=============")
    println(P_1000ByLevDist.summarizeLenAndDist())
    println("Precision@All\n=============")
    println(P_AllByLevDist.summarizeLenAndDist())
    println()
  }

  dataset.asStream().forEach { (brokeStr, fixedStr) ->
    val allTime = TimeSource.Monotonic.markNow()
    val brokeToks = brokeStr.tokenizeByWhitespace()
    val fixedToks = fixedStr.tokenizeByWhitespace()
    val levAlign = levenshteinAlign(brokeToks, fixedToks)
    val trueLevDist = levAlign.patchSize() // True distance, only used for logging purposes

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.total++
    P_10ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.total++
    P_100ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.total++
    P_1000ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.total++

    val humanRepairANSI = levenshteinAlign(brokeToks, fixedToks).paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")

    allRate.total++; levRates.getOrPut(trueLevDist) { LBHMetrics() }.total++

    val clock = TimeSource.Monotonic.markNow()
    var totalSamples = 0
    var matchFound = false
    val timeout = (TIMEOUT_MS / 1000).seconds

    var gpuResults = emptyList<Σᐩ>()
//  gpuResults = sendGPU(brokeStr).lines()
//      .map { it to P_BIFI_PY150.score(it.tokenizeByWhitespace()) }
//      .sortedBy { it.second }.map { it.first }.map { it.addNewLineIfMissing() }

    val gpuRank = gpuResults.indexOf(fixedStr)
    val gpuTime = clock.elapsedNow().inWholeMilliseconds
//    println("GPU returned ${gpuResults.size} results in $gpuTime ms, rank of true repair: $gpuRank")

    val cpuClock = TimeSource.Monotonic.markNow()
    var cpuTime: Long

//    val langSize = 0

    val dfa = sendCPU(brokeStr)
    val dfaRecognized = dfa?.recognizes(fixedToks, s2pg.tmLst) ?: false
    val langSize = 0

    val radius = (latestLangEditDistance + LED_BUFFER).coerceAtMost(MAX_RADIUS)
    println("∩-DFA ${if (dfaRecognized) "accepted" else "rejected"} human repair! (Total time=${allTime.elapsedNow()}, $trueLevDist/$radius)")
    if (!dfaRecognized) {
      if (trueLevDist <= radius)
      System.err.println("True Levenshtein distance ($trueLevDist) was <=${latestLangEditDistance + LED_BUFFER}, but true repair rejected!")
      allRate.error++; levRates.getOrPut(trueLevDist) { LBHMetrics() }.error++
    }

    var origRank = -1
    val unrankedResults =
      (dfa?.decodeDFA(mc = FAST_MC, timeout = timeout, dec = termDict) ?: emptyList())
        .parallelStream().map { it to it.charify().scoreWithPDFA() }
        .sorted { p1, p2 -> p1.second.compareTo(p2.second) }
        .map { it.first.addNewLineIfMissing() }.toList()
      .let {
        cpuTime = cpuClock.elapsedNow().inWholeMilliseconds
        origRank = it.indexOf(fixedStr)
        totalSamples = it.size
        println("CPU returned $totalSamples results in $cpuTime ms")
//        if ("Error" in getOutput(fixedStr)) it else it.filterErrors(s2pg, clock)
        it
      }

//    println()
//    measureTimeMillis { val dfa2 =dfa!!.toWFA(s2pg.tmLst); println("dFa2: ${dfa2.summary()}"); pythonPDFA.intersectOther(dfa2) }
//      .also { println("Took: ${it}ms to intersect ${pythonPDFA.summary()}") }

    val elapsed = clock.elapsedNow().inWholeMilliseconds
//    val rankedResults = unrankedResults
//      .parallelStream().map { it to it.charify().scoreWithPDFA() }
//      .sorted { p1, p2 -> p1.second.compareTo(p2.second) }.map { it.first }.toList()
    val rankedResults = if (unrankedResults.isEmpty()) emptyList()
    else (rerankGPU(brokeStr, unrankedResults.take(RERANK_THR).joinToString("\n")) + unrankedResults.drop(RERANK_THR))
        .onEachIndexed { i, it ->
          if (it == fixedStr) {
            matchFound = true
            println("Found human repair ((rank: $i, orig: $origRank) ${clock.elapsedNow()}):\n$humanRepairANSI")
          }
        }

    val allElapsed = clock.elapsedNow().inWholeMilliseconds
    println("Repairs fetched in $elapsed ms, reranking completed in ${allElapsed - elapsed} ms")

    val indexOfTarget = rankedResults.indexOf(fixedStr).also {
      if (matchFound) {
        P_AllByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.top1++
        if (it == 0) P_1ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.top1++
        if (it <= 10) P_10ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.top1++
        if (it <= 100) P_100ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.top1++
        if (it <= 1000) P_1000ByLevDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }.top1++
      }
    }

    rankedResults.firstOrNull()?.tokenizeByWhitespace()
      ?.let { println("Top-1 scoring repair:\n${levenshteinAlign(brokeToks, it).paintANSIColors()}") }

    if (indexOfTarget < 0) {
//      println("Drew $totalSamples samples in ${clock.elapsedNow()}/$timeout, Δ=$levDist human repair not found")
//      negative.appendText("${brokeToks.size}, $levDist, $latestLangEditDistance, $elapsed, $allElapsed, $totalSamples, $langSize\n")
    } else {
      allRate.recall++; levRates.getOrPut(trueLevDist) { LBHMetrics() }.recall++
      indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(trueLevDist) { LBHMetrics() }.top1++ } }

      println("Found Δ=$trueLevDist repair in $allElapsed ms, samp=${totalSamples}/$langSize, $indexOfTarget rank, $origRank orig")
      allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
//      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
      sampleTimeByLevDist[trueLevDist] = (sampleTimeByLevDist[trueLevDist] ?: 0.0) + elapsed
      println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
      allTimeByLevDist[trueLevDist] = (allTimeByLevDist[trueLevDist] ?: 0.0) + allElapsed
      println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
      samplesBeforeMatchByLevDist[trueLevDist] = (samplesBeforeMatchByLevDist[trueLevDist] ?: 0.0) + totalSamples
      println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
    }
    csv.appendText("${brokeToks.size}, $trueLevDist, $latestLangEditDistance, $elapsed, $allElapsed, $gpuTime, $cpuTime, $totalSamples, $langSize, $indexOfTarget, $origRank, $gpuRank\n")

    if (allRate.total % 100 == 0) summarizeRunningStats()
    println()
  }

  summarizeRunningStats()
}

/*
w/ Neural reranker

Lev(*): Top-1/rec/pos/total: 677 / 1338 / 2211 / 2211, errors: 0, P@1: 0.30619629127091813, P@All: 0.6051560379918589
Lev(1): Top-1/rec/pos/total: 310 / 673 / 747 / 747, errors: 0, P@1: 0.4149933065595716, P@All: 0.9009370816599732
Lev(2): Top-1/rec/pos/total: 223 / 349 / 456 / 456, errors: 0, P@1: 0.48903508771929827, P@All: 0.7653508771929824
Lev(3): Top-1/rec/pos/total: 144 / 316 / 1008 / 1008, errors: 0, P@1: 0.14285714285714285, P@All: 0.3134920634920635
Draw timings (ms): {1=3344.9177877428997, 2=2464.9372197309417, 3=4467.380418535127}
Full timings (ms): {1=5093.895366218236, 2=3083.681614349776, 3=5107.357997010464}
Avg samples drawn: {1=521.2847533632287, 2=273.39985052316894, 3=362.1188340807175}

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 46 / 127 ≈ 0.36220472440944884
|σ|∈[10, 20): Top-1/total: 144 / 381 ≈ 0.3779527559055118
|σ|∈[20, 30): Top-1/total: 143 / 429 ≈ 0.3333333333333333
|σ|∈[30, 40): Top-1/total: 107 / 360 ≈ 0.2972222222222222
|σ|∈[40, 50): Top-1/total: 79 / 291 ≈ 0.27147766323024053
|σ|∈[50, 60): Top-1/total: 68 / 237 ≈ 0.2869198312236287
|σ|∈[60, 70): Top-1/total: 52 / 207 ≈ 0.25120772946859904
|σ|∈[70, 80): Top-1/total: 38 / 179 ≈ 0.2122905027932961
Δ(1)= Top-1/total: 310 / 747 ≈ 0.4149933065595716
Δ(2)= Top-1/total: 223 / 456 ≈ 0.48903508771929827
Δ(3)= Top-1/total: 144 / 1008 ≈ 0.14285714285714285
(|σ|∈[0, 10), Δ=1): Top-1/total: 14 / 38 ≈ 0.3684210526315789
(|σ|∈[0, 10), Δ=2): Top-1/total: 20 / 31 ≈ 0.6451612903225806
(|σ|∈[0, 10), Δ=3): Top-1/total: 12 / 58 ≈ 0.20689655172413793
(|σ|∈[10, 20), Δ=1): Top-1/total: 75 / 144 ≈ 0.5208333333333334
(|σ|∈[10, 20), Δ=2): Top-1/total: 43 / 67 ≈ 0.6417910447761194
(|σ|∈[10, 20), Δ=3): Top-1/total: 26 / 170 ≈ 0.15294117647058825
(|σ|∈[20, 30), Δ=1): Top-1/total: 73 / 167 ≈ 0.437125748502994
(|σ|∈[20, 30), Δ=2): Top-1/total: 49 / 88 ≈ 0.5568181818181818
(|σ|∈[20, 30), Δ=3): Top-1/total: 21 / 174 ≈ 0.1206896551724138
(|σ|∈[30, 40), Δ=1): Top-1/total: 52 / 131 ≈ 0.3969465648854962
(|σ|∈[30, 40), Δ=2): Top-1/total: 34 / 68 ≈ 0.5
(|σ|∈[30, 40), Δ=3): Top-1/total: 21 / 161 ≈ 0.13043478260869565
(|σ|∈[40, 50), Δ=1): Top-1/total: 41 / 108 ≈ 0.37962962962962965
(|σ|∈[40, 50), Δ=2): Top-1/total: 21 / 50 ≈ 0.42
(|σ|∈[40, 50), Δ=3): Top-1/total: 17 / 133 ≈ 0.12781954887218044
(|σ|∈[50, 60), Δ=1): Top-1/total: 21 / 61 ≈ 0.3442622950819672
(|σ|∈[50, 60), Δ=2): Top-1/total: 25 / 52 ≈ 0.4807692307692308
(|σ|∈[50, 60), Δ=3): Top-1/total: 22 / 124 ≈ 0.1774193548387097
(|σ|∈[60, 70), Δ=1): Top-1/total: 20 / 47 ≈ 0.425531914893617
(|σ|∈[60, 70), Δ=2): Top-1/total: 15 / 50 ≈ 0.3
(|σ|∈[60, 70), Δ=3): Top-1/total: 17 / 110 ≈ 0.15454545454545454
(|σ|∈[70, 80), Δ=1): Top-1/total: 14 / 51 ≈ 0.27450980392156865
(|σ|∈[70, 80), Δ=2): Top-1/total: 16 / 50 ≈ 0.32
(|σ|∈[70, 80), Δ=3): Top-1/total: 8 / 78 ≈ 0.10256410256410256

Precision@10
===========
|σ|∈[0, 10): Top-1/total: 80 / 127 ≈ 0.6299212598425197
|σ|∈[10, 20): Top-1/total: 224 / 381 ≈ 0.5879265091863517
|σ|∈[20, 30): Top-1/total: 249 / 429 ≈ 0.5804195804195804
|σ|∈[30, 40): Top-1/total: 192 / 360 ≈ 0.5333333333333333
|σ|∈[40, 50): Top-1/total: 153 / 291 ≈ 0.5257731958762887
|σ|∈[50, 60): Top-1/total: 123 / 237 ≈ 0.5189873417721519
|σ|∈[60, 70): Top-1/total: 107 / 207 ≈ 0.5169082125603864
|σ|∈[70, 80): Top-1/total: 87 / 179 ≈ 0.4860335195530726
Δ(1)= Top-1/total: 621 / 747 ≈ 0.8313253012048193
Δ(2)= Top-1/total: 317 / 456 ≈ 0.6951754385964912
Δ(3)= Top-1/total: 277 / 1008 ≈ 0.2748015873015873
(|σ|∈[0, 10), Δ=1): Top-1/total: 32 / 38 ≈ 0.8421052631578947
(|σ|∈[0, 10), Δ=2): Top-1/total: 29 / 31 ≈ 0.9354838709677419
(|σ|∈[0, 10), Δ=3): Top-1/total: 19 / 58 ≈ 0.3275862068965517
(|σ|∈[10, 20), Δ=1): Top-1/total: 128 / 144 ≈ 0.8888888888888888
(|σ|∈[10, 20), Δ=2): Top-1/total: 55 / 67 ≈ 0.8208955223880597
(|σ|∈[10, 20), Δ=3): Top-1/total: 41 / 170 ≈ 0.2411764705882353
(|σ|∈[20, 30), Δ=1): Top-1/total: 144 / 167 ≈ 0.8622754491017964
(|σ|∈[20, 30), Δ=2): Top-1/total: 64 / 88 ≈ 0.7272727272727273
(|σ|∈[20, 30), Δ=3): Top-1/total: 41 / 174 ≈ 0.23563218390804597
(|σ|∈[30, 40), Δ=1): Top-1/total: 109 / 131 ≈ 0.8320610687022901
(|σ|∈[30, 40), Δ=2): Top-1/total: 45 / 68 ≈ 0.6617647058823529
(|σ|∈[30, 40), Δ=3): Top-1/total: 38 / 161 ≈ 0.2360248447204969
(|σ|∈[40, 50), Δ=1): Top-1/total: 84 / 108 ≈ 0.7777777777777778
(|σ|∈[40, 50), Δ=2): Top-1/total: 30 / 50 ≈ 0.6
(|σ|∈[40, 50), Δ=3): Top-1/total: 39 / 133 ≈ 0.2932330827067669
(|σ|∈[50, 60), Δ=1): Top-1/total: 48 / 61 ≈ 0.7868852459016393
(|σ|∈[50, 60), Δ=2): Top-1/total: 31 / 52 ≈ 0.5961538461538461
(|σ|∈[50, 60), Δ=3): Top-1/total: 44 / 124 ≈ 0.3548387096774194
(|σ|∈[60, 70), Δ=1): Top-1/total: 36 / 47 ≈ 0.7659574468085106
(|σ|∈[60, 70), Δ=2): Top-1/total: 35 / 50 ≈ 0.7
(|σ|∈[60, 70), Δ=3): Top-1/total: 36 / 110 ≈ 0.32727272727272727
(|σ|∈[70, 80), Δ=1): Top-1/total: 40 / 51 ≈ 0.7843137254901961
(|σ|∈[70, 80), Δ=2): Top-1/total: 28 / 50 ≈ 0.56
(|σ|∈[70, 80), Δ=3): Top-1/total: 19 / 78 ≈ 0.24358974358974358

Precision@100
===========
|σ|∈[0, 10): Top-1/total: 84 / 127 ≈ 0.6614173228346457
|σ|∈[10, 20): Top-1/total: 239 / 381 ≈ 0.6272965879265092
|σ|∈[20, 30): Top-1/total: 267 / 429 ≈ 0.6223776223776224
|σ|∈[30, 40): Top-1/total: 203 / 360 ≈ 0.5638888888888889
|σ|∈[40, 50): Top-1/total: 167 / 291 ≈ 0.5738831615120275
|σ|∈[50, 60): Top-1/total: 136 / 237 ≈ 0.5738396624472574
|σ|∈[60, 70): Top-1/total: 116 / 207 ≈ 0.5603864734299517
|σ|∈[70, 80): Top-1/total: 101 / 179 ≈ 0.5642458100558659
Δ(1)= Top-1/total: 663 / 747 ≈ 0.8875502008032129
Δ(2)= Top-1/total: 341 / 456 ≈ 0.7478070175438597
Δ(3)= Top-1/total: 309 / 1008 ≈ 0.30654761904761907
(|σ|∈[0, 10), Δ=1): Top-1/total: 33 / 38 ≈ 0.868421052631579
(|σ|∈[0, 10), Δ=2): Top-1/total: 29 / 31 ≈ 0.9354838709677419
(|σ|∈[0, 10), Δ=3): Top-1/total: 22 / 58 ≈ 0.3793103448275862
(|σ|∈[10, 20), Δ=1): Top-1/total: 135 / 144 ≈ 0.9375
(|σ|∈[10, 20), Δ=2): Top-1/total: 59 / 67 ≈ 0.8805970149253731
(|σ|∈[10, 20), Δ=3): Top-1/total: 45 / 170 ≈ 0.2647058823529412
(|σ|∈[20, 30), Δ=1): Top-1/total: 149 / 167 ≈ 0.8922155688622755
(|σ|∈[20, 30), Δ=2): Top-1/total: 71 / 88 ≈ 0.8068181818181818
(|σ|∈[20, 30), Δ=3): Top-1/total: 47 / 174 ≈ 0.27011494252873564
(|σ|∈[30, 40), Δ=1): Top-1/total: 116 / 131 ≈ 0.8854961832061069
(|σ|∈[30, 40), Δ=2): Top-1/total: 45 / 68 ≈ 0.6617647058823529
(|σ|∈[30, 40), Δ=3): Top-1/total: 42 / 161 ≈ 0.2608695652173913
(|σ|∈[40, 50), Δ=1): Top-1/total: 92 / 108 ≈ 0.8518518518518519
(|σ|∈[40, 50), Δ=2): Top-1/total: 32 / 50 ≈ 0.64
(|σ|∈[40, 50), Δ=3): Top-1/total: 43 / 133 ≈ 0.3233082706766917
(|σ|∈[50, 60), Δ=1): Top-1/total: 53 / 61 ≈ 0.8688524590163934
(|σ|∈[50, 60), Δ=2): Top-1/total: 34 / 52 ≈ 0.6538461538461539
(|σ|∈[50, 60), Δ=3): Top-1/total: 49 / 124 ≈ 0.3951612903225806
(|σ|∈[60, 70), Δ=1): Top-1/total: 42 / 47 ≈ 0.8936170212765957
(|σ|∈[60, 70), Δ=2): Top-1/total: 37 / 50 ≈ 0.74
(|σ|∈[60, 70), Δ=3): Top-1/total: 37 / 110 ≈ 0.33636363636363636
(|σ|∈[70, 80), Δ=1): Top-1/total: 43 / 51 ≈ 0.8431372549019608
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 50 ≈ 0.68
(|σ|∈[70, 80), Δ=3): Top-1/total: 24 / 78 ≈ 0.3076923076923077

Precision@1000
=============
|σ|∈[0, 10): Top-1/total: 85 / 127 ≈ 0.6692913385826772
|σ|∈[10, 20): Top-1/total: 244 / 381 ≈ 0.6404199475065617
|σ|∈[20, 30): Top-1/total: 269 / 429 ≈ 0.627039627039627
|σ|∈[30, 40): Top-1/total: 204 / 360 ≈ 0.5666666666666667
|σ|∈[40, 50): Top-1/total: 168 / 291 ≈ 0.5773195876288659
|σ|∈[50, 60): Top-1/total: 140 / 237 ≈ 0.5907172995780591
|σ|∈[60, 70): Top-1/total: 119 / 207 ≈ 0.5748792270531401
|σ|∈[70, 80): Top-1/total: 105 / 179 ≈ 0.5865921787709497
Δ(1)= Top-1/total: 671 / 747 ≈ 0.8982597054886211
Δ(2)= Top-1/total: 348 / 456 ≈ 0.7631578947368421
Δ(3)= Top-1/total: 315 / 1008 ≈ 0.3125
(|σ|∈[0, 10), Δ=1): Top-1/total: 34 / 38 ≈ 0.8947368421052632
(|σ|∈[0, 10), Δ=2): Top-1/total: 29 / 31 ≈ 0.9354838709677419
(|σ|∈[0, 10), Δ=3): Top-1/total: 22 / 58 ≈ 0.3793103448275862
(|σ|∈[10, 20), Δ=1): Top-1/total: 137 / 144 ≈ 0.9513888888888888
(|σ|∈[10, 20), Δ=2): Top-1/total: 61 / 67 ≈ 0.9104477611940298
(|σ|∈[10, 20), Δ=3): Top-1/total: 46 / 170 ≈ 0.27058823529411763
(|σ|∈[20, 30), Δ=1): Top-1/total: 149 / 167 ≈ 0.8922155688622755
(|σ|∈[20, 30), Δ=2): Top-1/total: 73 / 88 ≈ 0.8295454545454546
(|σ|∈[20, 30), Δ=3): Top-1/total: 47 / 174 ≈ 0.27011494252873564
(|σ|∈[30, 40), Δ=1): Top-1/total: 117 / 131 ≈ 0.8931297709923665
(|σ|∈[30, 40), Δ=2): Top-1/total: 45 / 68 ≈ 0.6617647058823529
(|σ|∈[30, 40), Δ=3): Top-1/total: 42 / 161 ≈ 0.2608695652173913
(|σ|∈[40, 50), Δ=1): Top-1/total: 93 / 108 ≈ 0.8611111111111112
(|σ|∈[40, 50), Δ=2): Top-1/total: 32 / 50 ≈ 0.64
(|σ|∈[40, 50), Δ=3): Top-1/total: 43 / 133 ≈ 0.3233082706766917
(|σ|∈[50, 60), Δ=1): Top-1/total: 54 / 61 ≈ 0.8852459016393442
(|σ|∈[50, 60), Δ=2): Top-1/total: 36 / 52 ≈ 0.6923076923076923
(|σ|∈[50, 60), Δ=3): Top-1/total: 50 / 124 ≈ 0.4032258064516129
(|σ|∈[60, 70), Δ=1): Top-1/total: 42 / 47 ≈ 0.8936170212765957
(|σ|∈[60, 70), Δ=2): Top-1/total: 38 / 50 ≈ 0.76
(|σ|∈[60, 70), Δ=3): Top-1/total: 39 / 110 ≈ 0.35454545454545455
(|σ|∈[70, 80), Δ=1): Top-1/total: 45 / 51 ≈ 0.8823529411764706
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 50 ≈ 0.68
(|σ|∈[70, 80), Δ=3): Top-1/total: 26 / 78 ≈ 0.3333333333333333

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 86 / 127 ≈ 0.6771653543307087
|σ|∈[10, 20): Top-1/total: 244 / 381 ≈ 0.6404199475065617
|σ|∈[20, 30): Top-1/total: 270 / 429 ≈ 0.6293706293706294
|σ|∈[30, 40): Top-1/total: 204 / 360 ≈ 0.5666666666666667
|σ|∈[40, 50): Top-1/total: 169 / 291 ≈ 0.5807560137457045
|σ|∈[50, 60): Top-1/total: 141 / 237 ≈ 0.5949367088607594
|σ|∈[60, 70): Top-1/total: 119 / 207 ≈ 0.5748792270531401
|σ|∈[70, 80): Top-1/total: 105 / 179 ≈ 0.5865921787709497
Δ(1)= Top-1/total: 673 / 747 ≈ 0.9009370816599732
Δ(2)= Top-1/total: 349 / 456 ≈ 0.7653508771929824
Δ(3)= Top-1/total: 316 / 1008 ≈ 0.3134920634920635
(|σ|∈[0, 10), Δ=1): Top-1/total: 35 / 38 ≈ 0.9210526315789473
(|σ|∈[0, 10), Δ=2): Top-1/total: 29 / 31 ≈ 0.9354838709677419
(|σ|∈[0, 10), Δ=3): Top-1/total: 22 / 58 ≈ 0.3793103448275862
(|σ|∈[10, 20), Δ=1): Top-1/total: 137 / 144 ≈ 0.9513888888888888
(|σ|∈[10, 20), Δ=2): Top-1/total: 61 / 67 ≈ 0.9104477611940298
(|σ|∈[10, 20), Δ=3): Top-1/total: 46 / 170 ≈ 0.27058823529411763
(|σ|∈[20, 30), Δ=1): Top-1/total: 149 / 167 ≈ 0.8922155688622755
(|σ|∈[20, 30), Δ=2): Top-1/total: 74 / 88 ≈ 0.8409090909090909
(|σ|∈[20, 30), Δ=3): Top-1/total: 47 / 174 ≈ 0.27011494252873564
(|σ|∈[30, 40), Δ=1): Top-1/total: 117 / 131 ≈ 0.8931297709923665
(|σ|∈[30, 40), Δ=2): Top-1/total: 45 / 68 ≈ 0.6617647058823529
(|σ|∈[30, 40), Δ=3): Top-1/total: 42 / 161 ≈ 0.2608695652173913
(|σ|∈[40, 50), Δ=1): Top-1/total: 93 / 108 ≈ 0.8611111111111112
(|σ|∈[40, 50), Δ=2): Top-1/total: 32 / 50 ≈ 0.64
(|σ|∈[40, 50), Δ=3): Top-1/total: 44 / 133 ≈ 0.3308270676691729
(|σ|∈[50, 60), Δ=1): Top-1/total: 55 / 61 ≈ 0.9016393442622951
(|σ|∈[50, 60), Δ=2): Top-1/total: 36 / 52 ≈ 0.6923076923076923
(|σ|∈[50, 60), Δ=3): Top-1/total: 50 / 124 ≈ 0.4032258064516129
(|σ|∈[60, 70), Δ=1): Top-1/total: 42 / 47 ≈ 0.8936170212765957
(|σ|∈[60, 70), Δ=2): Top-1/total: 38 / 50 ≈ 0.76
(|σ|∈[60, 70), Δ=3): Top-1/total: 39 / 110 ≈ 0.35454545454545455
(|σ|∈[70, 80), Δ=1): Top-1/total: 45 / 51 ≈ 0.8823529411764706
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 50 ≈ 0.68
(|σ|∈[70, 80), Δ=3): Top-1/total: 26 / 78 ≈ 0.3333333333333333

w/ DFA-NGRAM + No reranking

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 28 / 311 ≈ 0.09003215434083602
|σ|∈[10, 20): Top-1/total: 42 / 742 ≈ 0.05660377358490566
|σ|∈[20, 30): Top-1/total: 33 / 729 ≈ 0.04526748971193416
|σ|∈[30, 40): Top-1/total: 33 / 733 ≈ 0.045020463847203276
|σ|∈[40, 50): Top-1/total: 29 / 695 ≈ 0.041726618705035974
|σ|∈[50, 60): Top-1/total: 25 / 610 ≈ 0.040983606557377046
|σ|∈[60, 70): Top-1/total: 12 / 482 ≈ 0.024896265560165973
|σ|∈[70, 80): Top-1/total: 8 / 349 ≈ 0.022922636103151862
Δ(1)= Top-1/total: 30 / 2024 ≈ 0.014822134387351778
Δ(2)= Top-1/total: 104 / 1731 ≈ 0.06008087810514154
Δ(3)= Top-1/total: 76 / 896 ≈ 0.08482142857142858
(|σ|∈[0, 10), Δ=1): Top-1/total: 5 / 152 ≈ 0.03289473684210526
(|σ|∈[0, 10), Δ=2): Top-1/total: 19 / 105 ≈ 0.18095238095238095
(|σ|∈[0, 10), Δ=3): Top-1/total: 4 / 54 ≈ 0.07407407407407407
(|σ|∈[10, 20), Δ=1): Top-1/total: 10 / 299 ≈ 0.033444816053511704
(|σ|∈[10, 20), Δ=2): Top-1/total: 19 / 297 ≈ 0.06397306397306397
(|σ|∈[10, 20), Δ=3): Top-1/total: 13 / 146 ≈ 0.08904109589041095
(|σ|∈[20, 30), Δ=1): Top-1/total: 3 / 290 ≈ 0.010344827586206896
(|σ|∈[20, 30), Δ=2): Top-1/total: 17 / 289 ≈ 0.058823529411764705
(|σ|∈[20, 30), Δ=3): Top-1/total: 13 / 150 ≈ 0.08666666666666667
(|σ|∈[30, 40), Δ=1): Top-1/total: 4 / 290 ≈ 0.013793103448275862
(|σ|∈[30, 40), Δ=2): Top-1/total: 15 / 289 ≈ 0.05190311418685121
(|σ|∈[30, 40), Δ=3): Top-1/total: 14 / 154 ≈ 0.09090909090909091
(|σ|∈[40, 50), Δ=1): Top-1/total: 3 / 289 ≈ 0.010380622837370242
(|σ|∈[40, 50), Δ=2): Top-1/total: 17 / 292 ≈ 0.05821917808219178
(|σ|∈[40, 50), Δ=3): Top-1/total: 9 / 114 ≈ 0.07894736842105263
(|σ|∈[50, 60), Δ=1): Top-1/total: 3 / 288 ≈ 0.010416666666666666
(|σ|∈[50, 60), Δ=2): Top-1/total: 9 / 202 ≈ 0.04455445544554455
(|σ|∈[50, 60), Δ=3): Top-1/total: 13 / 120 ≈ 0.10833333333333334
(|σ|∈[60, 70), Δ=1): Top-1/total: 2 / 240 ≈ 0.008333333333333333
(|σ|∈[60, 70), Δ=2): Top-1/total: 4 / 147 ≈ 0.027210884353741496
(|σ|∈[60, 70), Δ=3): Top-1/total: 6 / 95 ≈ 0.06315789473684211
(|σ|∈[70, 80), Δ=1): Top-1/total: 0 / 176 ≈ 0.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 4 / 110 ≈ 0.03636363636363636
(|σ|∈[70, 80), Δ=3): Top-1/total: 4 / 63 ≈ 0.06349206349206349

Precision@10
===========
|σ|∈[0, 10): Top-1/total: 103 / 311 ≈ 0.3311897106109325
|σ|∈[10, 20): Top-1/total: 186 / 744 ≈ 0.25
|σ|∈[20, 30): Top-1/total: 158 / 732 ≈ 0.21584699453551912
|σ|∈[30, 40): Top-1/total: 162 / 733 ≈ 0.22100954979536153
|σ|∈[40, 50): Top-1/total: 137 / 695 ≈ 0.1971223021582734
|σ|∈[50, 60): Top-1/total: 135 / 610 ≈ 0.22131147540983606
|σ|∈[60, 70): Top-1/total: 79 / 482 ≈ 0.16390041493775934
|σ|∈[70, 80): Top-1/total: 59 / 349 ≈ 0.16905444126074498
Δ(1)= Top-1/total: 511 / 2027 ≈ 0.252096694622595
Δ(2)= Top-1/total: 380 / 1733 ≈ 0.2192729371032891
Δ(3)= Top-1/total: 128 / 896 ≈ 0.14285714285714285
(|σ|∈[0, 10), Δ=1): Top-1/total: 52 / 152 ≈ 0.34210526315789475
(|σ|∈[0, 10), Δ=2): Top-1/total: 41 / 105 ≈ 0.3904761904761905
(|σ|∈[0, 10), Δ=3): Top-1/total: 10 / 54 ≈ 0.18518518518518517
(|σ|∈[10, 20), Δ=1): Top-1/total: 86 / 299 ≈ 0.28762541806020064
(|σ|∈[10, 20), Δ=2): Top-1/total: 83 / 299 ≈ 0.27759197324414714
(|σ|∈[10, 20), Δ=3): Top-1/total: 17 / 146 ≈ 0.11643835616438356
(|σ|∈[20, 30), Δ=1): Top-1/total: 63 / 293 ≈ 0.2150170648464164
(|σ|∈[20, 30), Δ=2): Top-1/total: 74 / 289 ≈ 0.2560553633217993
(|σ|∈[20, 30), Δ=3): Top-1/total: 21 / 150 ≈ 0.14
(|σ|∈[30, 40), Δ=1): Top-1/total: 80 / 290 ≈ 0.27586206896551724
(|σ|∈[30, 40), Δ=2): Top-1/total: 63 / 289 ≈ 0.2179930795847751
(|σ|∈[30, 40), Δ=3): Top-1/total: 19 / 154 ≈ 0.12337662337662338
(|σ|∈[40, 50), Δ=1): Top-1/total: 68 / 289 ≈ 0.23529411764705882
(|σ|∈[40, 50), Δ=2): Top-1/total: 51 / 292 ≈ 0.17465753424657535
(|σ|∈[40, 50), Δ=3): Top-1/total: 18 / 114 ≈ 0.15789473684210525
(|σ|∈[50, 60), Δ=1): Top-1/total: 76 / 288 ≈ 0.2638888888888889
(|σ|∈[50, 60), Δ=2): Top-1/total: 38 / 202 ≈ 0.18811881188118812
(|σ|∈[50, 60), Δ=3): Top-1/total: 21 / 120 ≈ 0.175
(|σ|∈[60, 70), Δ=1): Top-1/total: 45 / 240 ≈ 0.1875
(|σ|∈[60, 70), Δ=2): Top-1/total: 19 / 147 ≈ 0.1292517006802721
(|σ|∈[60, 70), Δ=3): Top-1/total: 15 / 95 ≈ 0.15789473684210525
(|σ|∈[70, 80), Δ=1): Top-1/total: 41 / 176 ≈ 0.23295454545454544
(|σ|∈[70, 80), Δ=2): Top-1/total: 11 / 110 ≈ 0.1
(|σ|∈[70, 80), Δ=3): Top-1/total: 7 / 63 ≈ 0.1111111111111111

Precision@100
===========
|σ|∈[0, 10): Top-1/total: 194 / 311 ≈ 0.6237942122186495
|σ|∈[10, 20): Top-1/total: 389 / 744 ≈ 0.5228494623655914
|σ|∈[20, 30): Top-1/total: 334 / 732 ≈ 0.4562841530054645
|σ|∈[30, 40): Top-1/total: 339 / 733 ≈ 0.46248294679399726
|σ|∈[40, 50): Top-1/total: 310 / 695 ≈ 0.4460431654676259
|σ|∈[50, 60): Top-1/total: 289 / 610 ≈ 0.4737704918032787
|σ|∈[60, 70): Top-1/total: 215 / 482 ≈ 0.4460580912863071
|σ|∈[70, 80): Top-1/total: 153 / 349 ≈ 0.4383954154727794
Δ(1)= Top-1/total: 1204 / 2027 ≈ 0.5939812530833745
Δ(2)= Top-1/total: 815 / 1733 ≈ 0.4702827466820542
Δ(3)= Top-1/total: 204 / 896 ≈ 0.22767857142857142
(|σ|∈[0, 10), Δ=1): Top-1/total: 108 / 152 ≈ 0.7105263157894737
(|σ|∈[0, 10), Δ=2): Top-1/total: 74 / 105 ≈ 0.7047619047619048
(|σ|∈[0, 10), Δ=3): Top-1/total: 12 / 54 ≈ 0.2222222222222222
(|σ|∈[10, 20), Δ=1): Top-1/total: 194 / 299 ≈ 0.6488294314381271
(|σ|∈[10, 20), Δ=2): Top-1/total: 166 / 299 ≈ 0.5551839464882943
(|σ|∈[10, 20), Δ=3): Top-1/total: 29 / 146 ≈ 0.19863013698630136
(|σ|∈[20, 30), Δ=1): Top-1/total: 156 / 293 ≈ 0.5324232081911263
(|σ|∈[20, 30), Δ=2): Top-1/total: 149 / 289 ≈ 0.5155709342560554
(|σ|∈[20, 30), Δ=3): Top-1/total: 29 / 150 ≈ 0.19333333333333333
(|σ|∈[30, 40), Δ=1): Top-1/total: 172 / 290 ≈ 0.593103448275862
(|σ|∈[30, 40), Δ=2): Top-1/total: 133 / 289 ≈ 0.4602076124567474
(|σ|∈[30, 40), Δ=3): Top-1/total: 34 / 154 ≈ 0.22077922077922077
(|σ|∈[40, 50), Δ=1): Top-1/total: 158 / 289 ≈ 0.5467128027681661
(|σ|∈[40, 50), Δ=2): Top-1/total: 125 / 292 ≈ 0.4280821917808219
(|σ|∈[40, 50), Δ=3): Top-1/total: 27 / 114 ≈ 0.23684210526315788
(|σ|∈[50, 60), Δ=1): Top-1/total: 164 / 288 ≈ 0.5694444444444444
(|σ|∈[50, 60), Δ=2): Top-1/total: 87 / 202 ≈ 0.4306930693069307
(|σ|∈[50, 60), Δ=3): Top-1/total: 38 / 120 ≈ 0.31666666666666665
(|σ|∈[60, 70), Δ=1): Top-1/total: 145 / 240 ≈ 0.6041666666666666
(|σ|∈[60, 70), Δ=2): Top-1/total: 47 / 147 ≈ 0.3197278911564626
(|σ|∈[60, 70), Δ=3): Top-1/total: 23 / 95 ≈ 0.24210526315789474
(|σ|∈[70, 80), Δ=1): Top-1/total: 107 / 176 ≈ 0.6079545454545454
(|σ|∈[70, 80), Δ=2): Top-1/total: 34 / 110 ≈ 0.3090909090909091
(|σ|∈[70, 80), Δ=3): Top-1/total: 12 / 63 ≈ 0.19047619047619047

Precision@1000
=============
|σ|∈[0, 10): Top-1/total: 261 / 311 ≈ 0.8392282958199357
|σ|∈[10, 20): Top-1/total: 594 / 744 ≈ 0.7983870967741935
|σ|∈[20, 30): Top-1/total: 532 / 732 ≈ 0.726775956284153
|σ|∈[30, 40): Top-1/total: 522 / 733 ≈ 0.7121418826739427
|σ|∈[40, 50): Top-1/total: 498 / 695 ≈ 0.7165467625899281
|σ|∈[50, 60): Top-1/total: 434 / 610 ≈ 0.7114754098360656
|σ|∈[60, 70): Top-1/total: 344 / 482 ≈ 0.7136929460580913
|σ|∈[70, 80): Top-1/total: 250 / 349 ≈ 0.7163323782234957
Δ(1)= Top-1/total: 1818 / 2027 ≈ 0.8968919585594475
Δ(2)= Top-1/total: 1335 / 1733 ≈ 0.7703404500865552
Δ(3)= Top-1/total: 282 / 896 ≈ 0.31473214285714285
(|σ|∈[0, 10), Δ=1): Top-1/total: 141 / 152 ≈ 0.9276315789473685
(|σ|∈[0, 10), Δ=2): Top-1/total: 100 / 105 ≈ 0.9523809523809523
(|σ|∈[0, 10), Δ=3): Top-1/total: 20 / 54 ≈ 0.37037037037037035
(|σ|∈[10, 20), Δ=1): Top-1/total: 286 / 299 ≈ 0.9565217391304348
(|σ|∈[10, 20), Δ=2): Top-1/total: 266 / 299 ≈ 0.8896321070234113
(|σ|∈[10, 20), Δ=3): Top-1/total: 42 / 146 ≈ 0.2876712328767123
(|σ|∈[20, 30), Δ=1): Top-1/total: 264 / 293 ≈ 0.9010238907849829
(|σ|∈[20, 30), Δ=2): Top-1/total: 228 / 289 ≈ 0.7889273356401384
(|σ|∈[20, 30), Δ=3): Top-1/total: 40 / 150 ≈ 0.26666666666666666
(|σ|∈[30, 40), Δ=1): Top-1/total: 253 / 290 ≈ 0.8724137931034482
(|σ|∈[30, 40), Δ=2): Top-1/total: 218 / 289 ≈ 0.754325259515571
(|σ|∈[30, 40), Δ=3): Top-1/total: 51 / 154 ≈ 0.33116883116883117
(|σ|∈[40, 50), Δ=1): Top-1/total: 259 / 289 ≈ 0.8961937716262975
(|σ|∈[40, 50), Δ=2): Top-1/total: 205 / 292 ≈ 0.702054794520548
(|σ|∈[40, 50), Δ=3): Top-1/total: 34 / 114 ≈ 0.2982456140350877
(|σ|∈[50, 60), Δ=1): Top-1/total: 244 / 288 ≈ 0.8472222222222222
(|σ|∈[50, 60), Δ=2): Top-1/total: 144 / 202 ≈ 0.7128712871287128
(|σ|∈[50, 60), Δ=3): Top-1/total: 46 / 120 ≈ 0.38333333333333336
(|σ|∈[60, 70), Δ=1): Top-1/total: 213 / 240 ≈ 0.8875
(|σ|∈[60, 70), Δ=2): Top-1/total: 101 / 147 ≈ 0.6870748299319728
(|σ|∈[60, 70), Δ=3): Top-1/total: 30 / 95 ≈ 0.3157894736842105
(|σ|∈[70, 80), Δ=1): Top-1/total: 158 / 176 ≈ 0.8977272727272727
(|σ|∈[70, 80), Δ=2): Top-1/total: 73 / 110 ≈ 0.6636363636363637
(|σ|∈[70, 80), Δ=3): Top-1/total: 19 / 63 ≈ 0.30158730158730157

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 262 / 311 ≈ 0.842443729903537
|σ|∈[10, 20): Top-1/total: 595 / 744 ≈ 0.7997311827956989
|σ|∈[20, 30): Top-1/total: 534 / 732 ≈ 0.7295081967213115
|σ|∈[30, 40): Top-1/total: 524 / 733 ≈ 0.7148703956343793
|σ|∈[40, 50): Top-1/total: 502 / 695 ≈ 0.7223021582733813
|σ|∈[50, 60): Top-1/total: 434 / 610 ≈ 0.7114754098360656
|σ|∈[60, 70): Top-1/total: 345 / 482 ≈ 0.7157676348547718
|σ|∈[70, 80): Top-1/total: 251 / 349 ≈ 0.7191977077363897
Δ(1)= Top-1/total: 1824 / 2027 ≈ 0.8998519980266404
Δ(2)= Top-1/total: 1340 / 1733 ≈ 0.7732256203115984
Δ(3)= Top-1/total: 283 / 896 ≈ 0.3158482142857143
(|σ|∈[0, 10), Δ=1): Top-1/total: 142 / 152 ≈ 0.9342105263157895
(|σ|∈[0, 10), Δ=2): Top-1/total: 100 / 105 ≈ 0.9523809523809523
(|σ|∈[0, 10), Δ=3): Top-1/total: 20 / 54 ≈ 0.37037037037037035
(|σ|∈[10, 20), Δ=1): Top-1/total: 286 / 299 ≈ 0.9565217391304348
(|σ|∈[10, 20), Δ=2): Top-1/total: 267 / 299 ≈ 0.8929765886287625
(|σ|∈[10, 20), Δ=3): Top-1/total: 42 / 146 ≈ 0.2876712328767123
(|σ|∈[20, 30), Δ=1): Top-1/total: 266 / 293 ≈ 0.9078498293515358
(|σ|∈[20, 30), Δ=2): Top-1/total: 228 / 289 ≈ 0.7889273356401384
(|σ|∈[20, 30), Δ=3): Top-1/total: 40 / 150 ≈ 0.26666666666666666
(|σ|∈[30, 40), Δ=1): Top-1/total: 255 / 290 ≈ 0.8793103448275862
(|σ|∈[30, 40), Δ=2): Top-1/total: 218 / 289 ≈ 0.754325259515571
(|σ|∈[30, 40), Δ=3): Top-1/total: 51 / 154 ≈ 0.33116883116883117
(|σ|∈[40, 50), Δ=1): Top-1/total: 259 / 289 ≈ 0.8961937716262975
(|σ|∈[40, 50), Δ=2): Top-1/total: 208 / 292 ≈ 0.7123287671232876
(|σ|∈[40, 50), Δ=3): Top-1/total: 35 / 114 ≈ 0.30701754385964913
(|σ|∈[50, 60), Δ=1): Top-1/total: 244 / 288 ≈ 0.8472222222222222
(|σ|∈[50, 60), Δ=2): Top-1/total: 144 / 202 ≈ 0.7128712871287128
(|σ|∈[50, 60), Δ=3): Top-1/total: 46 / 120 ≈ 0.38333333333333336
(|σ|∈[60, 70), Δ=1): Top-1/total: 213 / 240 ≈ 0.8875
(|σ|∈[60, 70), Δ=2): Top-1/total: 102 / 147 ≈ 0.6938775510204082
(|σ|∈[60, 70), Δ=3): Top-1/total: 30 / 95 ≈ 0.3157894736842105
(|σ|∈[70, 80), Δ=1): Top-1/total: 159 / 176 ≈ 0.9034090909090909
(|σ|∈[70, 80), Δ=2): Top-1/total: 73 / 110 ≈ 0.6636363636363637
(|σ|∈[70, 80), Δ=3): Top-1/total: 19 / 63 ≈ 0.30158730158730157

w/ Listwise reranker + LED+1 + DFA-NGRAM-5000

Lev(*): Top-1/rec/pos/total: 213 / 1644 / 1755 / 1755, errors: 0, P@1: 0.12136752136752137, P@All: 0.9367521367521368
Lev(1): Top-1/rec/pos/total: 90 / 1031 / 1031 / 1031, errors: 0, P@1: 0.08729388942774007, P@All: 1.0
Lev(2): Top-1/rec/pos/total: 116 / 556 / 556 / 556, errors: 0, P@1: 0.20863309352517986, P@All: 1.0
Lev(3): Top-1/rec/pos/total: 7 / 57 / 168 / 168, errors: 0, P@1: 0.041666666666666664, P@All: 0.3392857142857143
Draw timings (ms): {1=1119.7384428223845, 2=1451.109489051095, 3=228.54075425790754}
Full timings (ms): {1=7325.353406326034, 2=5571.371654501217, 3=657.1411192214111}
Avg samples drawn: {1=9201.919099756691, 2=45567.62469586375, 3=7447.3205596107055}


Precision@1
===========
|σ|∈[0, 10): Top-1/total: 27 / 64 ≈ 0.421875
|σ|∈[10, 20): Top-1/total: 74 / 309 ≈ 0.23948220064724918
|σ|∈[20, 30): Top-1/total: 44 / 391 ≈ 0.11253196930946291
|σ|∈[30, 40): Top-1/total: 39 / 353 ≈ 0.11048158640226628
|σ|∈[40, 50): Top-1/total: 13 / 224 ≈ 0.05803571428571429
|σ|∈[50, 60): Top-1/total: 7 / 171 ≈ 0.04093567251461988
|σ|∈[60, 70): Top-1/total: 5 / 125 ≈ 0.04
|σ|∈[70, 80): Top-1/total: 4 / 118 ≈ 0.03389830508474576
Δ(1)= Top-1/total: 90 / 1031 ≈ 0.08729388942774007
Δ(2)= Top-1/total: 116 / 556 ≈ 0.20863309352517986
Δ(3)= Top-1/total: 7 / 168 ≈ 0.041666666666666664
(|σ|∈[0, 10), Δ=1): Top-1/total: 16 / 32 ≈ 0.5
(|σ|∈[0, 10), Δ=2): Top-1/total: 11 / 28 ≈ 0.39285714285714285
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 34 / 186 ≈ 0.1827956989247312
(|σ|∈[10, 20), Δ=2): Top-1/total: 40 / 95 ≈ 0.42105263157894735
(|σ|∈[10, 20), Δ=3): Top-1/total: 0 / 28 ≈ 0.0
(|σ|∈[20, 30), Δ=1): Top-1/total: 21 / 251 ≈ 0.08366533864541832
(|σ|∈[20, 30), Δ=2): Top-1/total: 22 / 108 ≈ 0.2037037037037037
(|σ|∈[20, 30), Δ=3): Top-1/total: 1 / 32 ≈ 0.03125
(|σ|∈[30, 40), Δ=1): Top-1/total: 14 / 213 ≈ 0.06572769953051644
(|σ|∈[30, 40), Δ=2): Top-1/total: 21 / 102 ≈ 0.20588235294117646
(|σ|∈[30, 40), Δ=3): Top-1/total: 4 / 38 ≈ 0.10526315789473684
(|σ|∈[40, 50), Δ=1): Top-1/total: 2 / 125 ≈ 0.016
(|σ|∈[40, 50), Δ=2): Top-1/total: 11 / 75 ≈ 0.14666666666666667
(|σ|∈[40, 50), Δ=3): Top-1/total: 0 / 24 ≈ 0.0
(|σ|∈[50, 60), Δ=1): Top-1/total: 2 / 96 ≈ 0.020833333333333332
(|σ|∈[50, 60), Δ=2): Top-1/total: 4 / 63 ≈ 0.06349206349206349
(|σ|∈[50, 60), Δ=3): Top-1/total: 1 / 12 ≈ 0.08333333333333333
(|σ|∈[60, 70), Δ=1): Top-1/total: 0 / 60 ≈ 0.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 4 / 50 ≈ 0.08
(|σ|∈[60, 70), Δ=3): Top-1/total: 1 / 15 ≈ 0.06666666666666667
(|σ|∈[70, 80), Δ=1): Top-1/total: 1 / 68 ≈ 0.014705882352941176
(|σ|∈[70, 80), Δ=2): Top-1/total: 3 / 35 ≈ 0.08571428571428572
(|σ|∈[70, 80), Δ=3): Top-1/total: 0 / 15 ≈ 0.0

Precision@10
===========
|σ|∈[0, 10): Top-1/total: 43 / 64 ≈ 0.671875
|σ|∈[10, 20): Top-1/total: 188 / 309 ≈ 0.6084142394822006
|σ|∈[20, 30): Top-1/total: 177 / 391 ≈ 0.45268542199488493
|σ|∈[30, 40): Top-1/total: 129 / 353 ≈ 0.3654390934844193
|σ|∈[40, 50): Top-1/total: 69 / 224 ≈ 0.3080357142857143
|σ|∈[50, 60): Top-1/total: 39 / 171 ≈ 0.22807017543859648
|σ|∈[60, 70): Top-1/total: 16 / 125 ≈ 0.128
|σ|∈[70, 80): Top-1/total: 15 / 118 ≈ 0.1271186440677966
Δ(1)= Top-1/total: 384 / 1031 ≈ 0.37245392822502427
Δ(2)= Top-1/total: 267 / 556 ≈ 0.4802158273381295
Δ(3)= Top-1/total: 25 / 168 ≈ 0.1488095238095238
(|σ|∈[0, 10), Δ=1): Top-1/total: 23 / 32 ≈ 0.71875
(|σ|∈[0, 10), Δ=2): Top-1/total: 20 / 28 ≈ 0.7142857142857143
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 113 / 186 ≈ 0.6075268817204301
(|σ|∈[10, 20), Δ=2): Top-1/total: 74 / 95 ≈ 0.7789473684210526
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 28 ≈ 0.03571428571428571
(|σ|∈[20, 30), Δ=1): Top-1/total: 112 / 251 ≈ 0.44621513944223107
(|σ|∈[20, 30), Δ=2): Top-1/total: 57 / 108 ≈ 0.5277777777777778
(|σ|∈[20, 30), Δ=3): Top-1/total: 8 / 32 ≈ 0.25
(|σ|∈[30, 40), Δ=1): Top-1/total: 73 / 213 ≈ 0.3427230046948357
(|σ|∈[30, 40), Δ=2): Top-1/total: 49 / 102 ≈ 0.4803921568627451
(|σ|∈[30, 40), Δ=3): Top-1/total: 7 / 38 ≈ 0.18421052631578946
(|σ|∈[40, 50), Δ=1): Top-1/total: 36 / 125 ≈ 0.288
(|σ|∈[40, 50), Δ=2): Top-1/total: 28 / 75 ≈ 0.37333333333333335
(|σ|∈[40, 50), Δ=3): Top-1/total: 5 / 24 ≈ 0.20833333333333334
(|σ|∈[50, 60), Δ=1): Top-1/total: 19 / 96 ≈ 0.19791666666666666
(|σ|∈[50, 60), Δ=2): Top-1/total: 18 / 63 ≈ 0.2857142857142857
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 12 ≈ 0.16666666666666666
(|σ|∈[60, 70), Δ=1): Top-1/total: 2 / 60 ≈ 0.03333333333333333
(|σ|∈[60, 70), Δ=2): Top-1/total: 13 / 50 ≈ 0.26
(|σ|∈[60, 70), Δ=3): Top-1/total: 1 / 15 ≈ 0.06666666666666667
(|σ|∈[70, 80), Δ=1): Top-1/total: 6 / 68 ≈ 0.08823529411764706
(|σ|∈[70, 80), Δ=2): Top-1/total: 8 / 35 ≈ 0.22857142857142856
(|σ|∈[70, 80), Δ=3): Top-1/total: 1 / 15 ≈ 0.06666666666666667

Precision@100
===========
|σ|∈[0, 10): Top-1/total: 55 / 64 ≈ 0.859375
|σ|∈[10, 20): Top-1/total: 249 / 309 ≈ 0.8058252427184466
|σ|∈[20, 30): Top-1/total: 300 / 391 ≈ 0.7672634271099744
|σ|∈[30, 40): Top-1/total: 263 / 353 ≈ 0.7450424929178471
|σ|∈[40, 50): Top-1/total: 139 / 224 ≈ 0.6205357142857143
|σ|∈[50, 60): Top-1/total: 96 / 171 ≈ 0.5614035087719298
|σ|∈[60, 70): Top-1/total: 58 / 125 ≈ 0.464
|σ|∈[70, 80): Top-1/total: 56 / 118 ≈ 0.4745762711864407
Δ(1)= Top-1/total: 787 / 1031 ≈ 0.7633365664403492
Δ(2)= Top-1/total: 391 / 556 ≈ 0.7032374100719424
Δ(3)= Top-1/total: 38 / 168 ≈ 0.2261904761904762
(|σ|∈[0, 10), Δ=1): Top-1/total: 31 / 32 ≈ 0.96875
(|σ|∈[0, 10), Δ=2): Top-1/total: 24 / 28 ≈ 0.8571428571428571
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 163 / 186 ≈ 0.8763440860215054
(|σ|∈[10, 20), Δ=2): Top-1/total: 84 / 95 ≈ 0.8842105263157894
(|σ|∈[10, 20), Δ=3): Top-1/total: 2 / 28 ≈ 0.07142857142857142
(|σ|∈[20, 30), Δ=1): Top-1/total: 210 / 251 ≈ 0.8366533864541833
(|σ|∈[20, 30), Δ=2): Top-1/total: 81 / 108 ≈ 0.75
(|σ|∈[20, 30), Δ=3): Top-1/total: 9 / 32 ≈ 0.28125
(|σ|∈[30, 40), Δ=1): Top-1/total: 176 / 213 ≈ 0.8262910798122066
(|σ|∈[30, 40), Δ=2): Top-1/total: 76 / 102 ≈ 0.7450980392156863
(|σ|∈[30, 40), Δ=3): Top-1/total: 11 / 38 ≈ 0.2894736842105263
(|σ|∈[40, 50), Δ=1): Top-1/total: 87 / 125 ≈ 0.696
(|σ|∈[40, 50), Δ=2): Top-1/total: 45 / 75 ≈ 0.6
(|σ|∈[40, 50), Δ=3): Top-1/total: 7 / 24 ≈ 0.2916666666666667
(|σ|∈[50, 60), Δ=1): Top-1/total: 58 / 96 ≈ 0.6041666666666666
(|σ|∈[50, 60), Δ=2): Top-1/total: 35 / 63 ≈ 0.5555555555555556
(|σ|∈[50, 60), Δ=3): Top-1/total: 3 / 12 ≈ 0.25
(|σ|∈[60, 70), Δ=1): Top-1/total: 28 / 60 ≈ 0.4666666666666667
(|σ|∈[60, 70), Δ=2): Top-1/total: 26 / 50 ≈ 0.52
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 15 ≈ 0.26666666666666666
(|σ|∈[70, 80), Δ=1): Top-1/total: 34 / 68 ≈ 0.5
(|σ|∈[70, 80), Δ=2): Top-1/total: 20 / 35 ≈ 0.5714285714285714
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 15 ≈ 0.13333333333333333

Precision@1000
=============
|σ|∈[0, 10): Top-1/total: 60 / 64 ≈ 0.9375
|σ|∈[10, 20): Top-1/total: 266 / 309 ≈ 0.86084142394822
|σ|∈[20, 30): Top-1/total: 343 / 391 ≈ 0.8772378516624041
|σ|∈[30, 40): Top-1/total: 311 / 353 ≈ 0.8810198300283286
|σ|∈[40, 50): Top-1/total: 179 / 224 ≈ 0.7991071428571429
|σ|∈[50, 60): Top-1/total: 138 / 171 ≈ 0.8070175438596491
|σ|∈[60, 70): Top-1/total: 91 / 125 ≈ 0.728
|σ|∈[70, 80): Top-1/total: 91 / 118 ≈ 0.7711864406779662
Δ(1)= Top-1/total: 979 / 1031 ≈ 0.9495635305528612
Δ(2)= Top-1/total: 455 / 556 ≈ 0.8183453237410072
Δ(3)= Top-1/total: 45 / 168 ≈ 0.26785714285714285
(|σ|∈[0, 10), Δ=1): Top-1/total: 32 / 32 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 28 / 28 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 176 / 186 ≈ 0.946236559139785
(|σ|∈[10, 20), Δ=2): Top-1/total: 87 / 95 ≈ 0.9157894736842105
(|σ|∈[10, 20), Δ=3): Top-1/total: 3 / 28 ≈ 0.10714285714285714
(|σ|∈[20, 30), Δ=1): Top-1/total: 242 / 251 ≈ 0.9641434262948207
(|σ|∈[20, 30), Δ=2): Top-1/total: 92 / 108 ≈ 0.8518518518518519
(|σ|∈[20, 30), Δ=3): Top-1/total: 9 / 32 ≈ 0.28125
(|σ|∈[30, 40), Δ=1): Top-1/total: 209 / 213 ≈ 0.9812206572769953
(|σ|∈[30, 40), Δ=2): Top-1/total: 88 / 102 ≈ 0.8627450980392157
(|σ|∈[30, 40), Δ=3): Top-1/total: 14 / 38 ≈ 0.3684210526315789
(|σ|∈[40, 50), Δ=1): Top-1/total: 118 / 125 ≈ 0.944
(|σ|∈[40, 50), Δ=2): Top-1/total: 53 / 75 ≈ 0.7066666666666667
(|σ|∈[40, 50), Δ=3): Top-1/total: 8 / 24 ≈ 0.3333333333333333
(|σ|∈[50, 60), Δ=1): Top-1/total: 86 / 96 ≈ 0.8958333333333334
(|σ|∈[50, 60), Δ=2): Top-1/total: 49 / 63 ≈ 0.7777777777777778
(|σ|∈[50, 60), Δ=3): Top-1/total: 3 / 12 ≈ 0.25
(|σ|∈[60, 70), Δ=1): Top-1/total: 53 / 60 ≈ 0.8833333333333333
(|σ|∈[60, 70), Δ=2): Top-1/total: 32 / 50 ≈ 0.64
(|σ|∈[60, 70), Δ=3): Top-1/total: 6 / 15 ≈ 0.4
(|σ|∈[70, 80), Δ=1): Top-1/total: 63 / 68 ≈ 0.9264705882352942
(|σ|∈[70, 80), Δ=2): Top-1/total: 26 / 35 ≈ 0.7428571428571429
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 15 ≈ 0.13333333333333333

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 61 / 64 ≈ 0.953125
|σ|∈[10, 20): Top-1/total: 285 / 309 ≈ 0.9223300970873787
|σ|∈[20, 30): Top-1/total: 369 / 391 ≈ 0.9437340153452686
|σ|∈[30, 40): Top-1/total: 331 / 353 ≈ 0.9376770538243626
|σ|∈[40, 50): Top-1/total: 209 / 224 ≈ 0.9330357142857143
|σ|∈[50, 60): Top-1/total: 162 / 171 ≈ 0.9473684210526315
|σ|∈[60, 70): Top-1/total: 119 / 125 ≈ 0.952
|σ|∈[70, 80): Top-1/total: 108 / 118 ≈ 0.9152542372881356
Δ(1)= Top-1/total: 1031 / 1031 ≈ 1.0
Δ(2)= Top-1/total: 556 / 556 ≈ 1.0
Δ(3)= Top-1/total: 57 / 168 ≈ 0.3392857142857143
(|σ|∈[0, 10), Δ=1): Top-1/total: 32 / 32 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 28 / 28 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 1 / 4 ≈ 0.25
(|σ|∈[10, 20), Δ=1): Top-1/total: 186 / 186 ≈ 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 95 / 95 ≈ 1.0
(|σ|∈[10, 20), Δ=3): Top-1/total: 4 / 28 ≈ 0.14285714285714285
(|σ|∈[20, 30), Δ=1): Top-1/total: 251 / 251 ≈ 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 108 / 108 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 10 / 32 ≈ 0.3125
(|σ|∈[30, 40), Δ=1): Top-1/total: 213 / 213 ≈ 1.0
(|σ|∈[30, 40), Δ=2): Top-1/total: 102 / 102 ≈ 1.0
(|σ|∈[30, 40), Δ=3): Top-1/total: 16 / 38 ≈ 0.42105263157894735
(|σ|∈[40, 50), Δ=1): Top-1/total: 125 / 125 ≈ 1.0
(|σ|∈[40, 50), Δ=2): Top-1/total: 75 / 75 ≈ 1.0
(|σ|∈[40, 50), Δ=3): Top-1/total: 9 / 24 ≈ 0.375
(|σ|∈[50, 60), Δ=1): Top-1/total: 96 / 96 ≈ 1.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 63 / 63 ≈ 1.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 3 / 12 ≈ 0.25
(|σ|∈[60, 70), Δ=1): Top-1/total: 60 / 60 ≈ 1.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 50 / 50 ≈ 1.0
(|σ|∈[60, 70), Δ=3): Top-1/total: 9 / 15 ≈ 0.6
(|σ|∈[70, 80), Δ=1): Top-1/total: 68 / 68 ≈ 1.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 35 / 35 ≈ 1.0
(|σ|∈[70, 80), Δ=3): Top-1/total: 5 / 15 ≈ 0.3333333333333333

w/ Listwise reranker + LED+1 + DFA-NGRAM-1000

Lev(*): Top-1/rec/pos/total: 240 / 1642 / 1755 / 1755, errors: 0, P@1: 0.13675213675213677, P@All: 0.9356125356125357
Lev(1): Top-1/rec/pos/total: 125 / 1031 / 1031 / 1031, errors: 0, P@1: 0.12124151309408342, P@All: 1.0
Lev(2): Top-1/rec/pos/total: 107 / 555 / 556 / 556, errors: 0, P@1: 0.19244604316546762, P@All: 0.9982014388489209
Lev(3): Top-1/rec/pos/total: 8 / 56 / 168 / 168, errors: 0, P@1: 0.047619047619047616, P@All: 0.3333333333333333
Draw timings (ms): {1=779.0426309378806, 2=1034.7722289890378, 3=170.57612667478685}
Full timings (ms): {1=2527.8672350791717, 2=2027.068209500609, 3=268.6985383678441}
Avg samples drawn: {1=9213.127283800244, 2=30313.473812423872, 3=6487.1613885505485}


Precision@1
===========
|σ|∈[0, 10): Top-1/total: 28 / 64 ≈ 0.4375
|σ|∈[10, 20): Top-1/total: 76 / 309 ≈ 0.2459546925566343
|σ|∈[20, 30): Top-1/total: 57 / 391 ≈ 0.14578005115089515
|σ|∈[30, 40): Top-1/total: 42 / 353 ≈ 0.11898016997167139
|σ|∈[40, 50): Top-1/total: 17 / 224 ≈ 0.07589285714285714
|σ|∈[50, 60): Top-1/total: 11 / 171 ≈ 0.06432748538011696
|σ|∈[60, 70): Top-1/total: 3 / 125 ≈ 0.024
|σ|∈[70, 80): Top-1/total: 6 / 118 ≈ 0.05084745762711865
Δ(1)= Top-1/total: 125 / 1031 ≈ 0.12124151309408342
Δ(2)= Top-1/total: 107 / 556 ≈ 0.19244604316546762
Δ(3)= Top-1/total: 8 / 168 ≈ 0.047619047619047616
(|σ|∈[0, 10), Δ=1): Top-1/total: 17 / 32 ≈ 0.53125
(|σ|∈[0, 10), Δ=2): Top-1/total: 11 / 28 ≈ 0.39285714285714285
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 41 / 186 ≈ 0.22043010752688172
(|σ|∈[10, 20), Δ=2): Top-1/total: 35 / 95 ≈ 0.3684210526315789
(|σ|∈[10, 20), Δ=3): Top-1/total: 0 / 28 ≈ 0.0
(|σ|∈[20, 30), Δ=1): Top-1/total: 37 / 251 ≈ 0.14741035856573706
(|σ|∈[20, 30), Δ=2): Top-1/total: 19 / 108 ≈ 0.17592592592592593
(|σ|∈[20, 30), Δ=3): Top-1/total: 1 / 32 ≈ 0.03125
(|σ|∈[30, 40), Δ=1): Top-1/total: 19 / 213 ≈ 0.0892018779342723
(|σ|∈[30, 40), Δ=2): Top-1/total: 19 / 102 ≈ 0.18627450980392157
(|σ|∈[30, 40), Δ=3): Top-1/total: 4 / 38 ≈ 0.10526315789473684
(|σ|∈[40, 50), Δ=1): Top-1/total: 6 / 125 ≈ 0.048
(|σ|∈[40, 50), Δ=2): Top-1/total: 11 / 75 ≈ 0.14666666666666667
(|σ|∈[40, 50), Δ=3): Top-1/total: 0 / 24 ≈ 0.0
(|σ|∈[50, 60), Δ=1): Top-1/total: 3 / 96 ≈ 0.03125
(|σ|∈[50, 60), Δ=2): Top-1/total: 6 / 63 ≈ 0.09523809523809523
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 12 ≈ 0.16666666666666666
(|σ|∈[60, 70), Δ=1): Top-1/total: 0 / 60 ≈ 0.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 2 / 50 ≈ 0.04
(|σ|∈[60, 70), Δ=3): Top-1/total: 1 / 15 ≈ 0.06666666666666667
(|σ|∈[70, 80), Δ=1): Top-1/total: 2 / 68 ≈ 0.029411764705882353
(|σ|∈[70, 80), Δ=2): Top-1/total: 4 / 35 ≈ 0.11428571428571428
(|σ|∈[70, 80), Δ=3): Top-1/total: 0 / 15 ≈ 0.0

Precision@10
===========
|σ|∈[0, 10): Top-1/total: 44 / 64 ≈ 0.6875
|σ|∈[10, 20): Top-1/total: 178 / 309 ≈ 0.5760517799352751
|σ|∈[20, 30): Top-1/total: 194 / 391 ≈ 0.4961636828644501
|σ|∈[30, 40): Top-1/total: 153 / 353 ≈ 0.43342776203966005
|σ|∈[40, 50): Top-1/total: 85 / 224 ≈ 0.3794642857142857
|σ|∈[50, 60): Top-1/total: 50 / 171 ≈ 0.29239766081871343
|σ|∈[60, 70): Top-1/total: 23 / 125 ≈ 0.184
|σ|∈[70, 80): Top-1/total: 26 / 118 ≈ 0.22033898305084745
Δ(1)= Top-1/total: 464 / 1031 ≈ 0.45004849660523766
Δ(2)= Top-1/total: 265 / 556 ≈ 0.4766187050359712
Δ(3)= Top-1/total: 24 / 168 ≈ 0.14285714285714285
(|σ|∈[0, 10), Δ=1): Top-1/total: 24 / 32 ≈ 0.75
(|σ|∈[0, 10), Δ=2): Top-1/total: 20 / 28 ≈ 0.7142857142857143
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 115 / 186 ≈ 0.6182795698924731
(|σ|∈[10, 20), Δ=2): Top-1/total: 62 / 95 ≈ 0.6526315789473685
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 28 ≈ 0.03571428571428571
(|σ|∈[20, 30), Δ=1): Top-1/total: 131 / 251 ≈ 0.5219123505976095
(|σ|∈[20, 30), Δ=2): Top-1/total: 56 / 108 ≈ 0.5185185185185185
(|σ|∈[20, 30), Δ=3): Top-1/total: 7 / 32 ≈ 0.21875
(|σ|∈[30, 40), Δ=1): Top-1/total: 98 / 213 ≈ 0.460093896713615
(|σ|∈[30, 40), Δ=2): Top-1/total: 49 / 102 ≈ 0.4803921568627451
(|σ|∈[30, 40), Δ=3): Top-1/total: 6 / 38 ≈ 0.15789473684210525
(|σ|∈[40, 50), Δ=1): Top-1/total: 48 / 125 ≈ 0.384
(|σ|∈[40, 50), Δ=2): Top-1/total: 33 / 75 ≈ 0.44
(|σ|∈[40, 50), Δ=3): Top-1/total: 4 / 24 ≈ 0.16666666666666666
(|σ|∈[50, 60), Δ=1): Top-1/total: 25 / 96 ≈ 0.2604166666666667
(|σ|∈[50, 60), Δ=2): Top-1/total: 23 / 63 ≈ 0.36507936507936506
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 12 ≈ 0.16666666666666666
(|σ|∈[60, 70), Δ=1): Top-1/total: 9 / 60 ≈ 0.15
(|σ|∈[60, 70), Δ=2): Top-1/total: 11 / 50 ≈ 0.22
(|σ|∈[60, 70), Δ=3): Top-1/total: 3 / 15 ≈ 0.2
(|σ|∈[70, 80), Δ=1): Top-1/total: 14 / 68 ≈ 0.20588235294117646
(|σ|∈[70, 80), Δ=2): Top-1/total: 11 / 35 ≈ 0.3142857142857143
(|σ|∈[70, 80), Δ=3): Top-1/total: 1 / 15 ≈ 0.06666666666666667

Precision@100
===========
|σ|∈[0, 10): Top-1/total: 53 / 64 ≈ 0.828125
|σ|∈[10, 20): Top-1/total: 217 / 309 ≈ 0.7022653721682848
|σ|∈[20, 30): Top-1/total: 289 / 391 ≈ 0.7391304347826086
|σ|∈[30, 40): Top-1/total: 251 / 353 ≈ 0.7110481586402266
|σ|∈[40, 50): Top-1/total: 143 / 224 ≈ 0.6383928571428571
|σ|∈[50, 60): Top-1/total: 109 / 171 ≈ 0.6374269005847953
|σ|∈[60, 70): Top-1/total: 57 / 125 ≈ 0.456
|σ|∈[70, 80): Top-1/total: 69 / 118 ≈ 0.5847457627118644
Δ(1)= Top-1/total: 808 / 1031 ≈ 0.7837051406401552
Δ(2)= Top-1/total: 347 / 556 ≈ 0.6241007194244604
Δ(3)= Top-1/total: 33 / 168 ≈ 0.19642857142857142
(|σ|∈[0, 10), Δ=1): Top-1/total: 31 / 32 ≈ 0.96875
(|σ|∈[0, 10), Δ=2): Top-1/total: 22 / 28 ≈ 0.7857142857142857
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 149 / 186 ≈ 0.8010752688172043
(|σ|∈[10, 20), Δ=2): Top-1/total: 67 / 95 ≈ 0.7052631578947368
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 28 ≈ 0.03571428571428571
(|σ|∈[20, 30), Δ=1): Top-1/total: 209 / 251 ≈ 0.8326693227091634
(|σ|∈[20, 30), Δ=2): Top-1/total: 72 / 108 ≈ 0.6666666666666666
(|σ|∈[20, 30), Δ=3): Top-1/total: 8 / 32 ≈ 0.25
(|σ|∈[30, 40), Δ=1): Top-1/total: 176 / 213 ≈ 0.8262910798122066
(|σ|∈[30, 40), Δ=2): Top-1/total: 66 / 102 ≈ 0.6470588235294118
(|σ|∈[30, 40), Δ=3): Top-1/total: 9 / 38 ≈ 0.23684210526315788
(|σ|∈[40, 50), Δ=1): Top-1/total: 93 / 125 ≈ 0.744
(|σ|∈[40, 50), Δ=2): Top-1/total: 43 / 75 ≈ 0.5733333333333334
(|σ|∈[40, 50), Δ=3): Top-1/total: 7 / 24 ≈ 0.2916666666666667
(|σ|∈[50, 60), Δ=1): Top-1/total: 68 / 96 ≈ 0.7083333333333334
(|σ|∈[50, 60), Δ=2): Top-1/total: 39 / 63 ≈ 0.6190476190476191
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 12 ≈ 0.16666666666666666
(|σ|∈[60, 70), Δ=1): Top-1/total: 34 / 60 ≈ 0.5666666666666667
(|σ|∈[60, 70), Δ=2): Top-1/total: 19 / 50 ≈ 0.38
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 15 ≈ 0.26666666666666666
(|σ|∈[70, 80), Δ=1): Top-1/total: 48 / 68 ≈ 0.7058823529411765
(|σ|∈[70, 80), Δ=2): Top-1/total: 19 / 35 ≈ 0.5428571428571428
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 15 ≈ 0.13333333333333333

Precision@1000
=============
|σ|∈[0, 10): Top-1/total: 58 / 64 ≈ 0.90625
|σ|∈[10, 20): Top-1/total: 233 / 309 ≈ 0.7540453074433657
|σ|∈[20, 30): Top-1/total: 316 / 391 ≈ 0.8081841432225064
|σ|∈[30, 40): Top-1/total: 270 / 353 ≈ 0.7648725212464589
|σ|∈[40, 50): Top-1/total: 163 / 224 ≈ 0.7276785714285714
|σ|∈[50, 60): Top-1/total: 131 / 171 ≈ 0.7660818713450293
|σ|∈[60, 70): Top-1/total: 76 / 125 ≈ 0.608
|σ|∈[70, 80): Top-1/total: 86 / 118 ≈ 0.7288135593220338
Δ(1)= Top-1/total: 913 / 1031 ≈ 0.8855480116391853
Δ(2)= Top-1/total: 384 / 556 ≈ 0.6906474820143885
Δ(3)= Top-1/total: 36 / 168 ≈ 0.21428571428571427
(|σ|∈[0, 10), Δ=1): Top-1/total: 32 / 32 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 26 / 28 ≈ 0.9285714285714286
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 163 / 186 ≈ 0.8763440860215054
(|σ|∈[10, 20), Δ=2): Top-1/total: 69 / 95 ≈ 0.7263157894736842
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 28 ≈ 0.03571428571428571
(|σ|∈[20, 30), Δ=1): Top-1/total: 226 / 251 ≈ 0.900398406374502
(|σ|∈[20, 30), Δ=2): Top-1/total: 82 / 108 ≈ 0.7592592592592593
(|σ|∈[20, 30), Δ=3): Top-1/total: 8 / 32 ≈ 0.25
(|σ|∈[30, 40), Δ=1): Top-1/total: 189 / 213 ≈ 0.8873239436619719
(|σ|∈[30, 40), Δ=2): Top-1/total: 70 / 102 ≈ 0.6862745098039216
(|σ|∈[30, 40), Δ=3): Top-1/total: 11 / 38 ≈ 0.2894736842105263
(|σ|∈[40, 50), Δ=1): Top-1/total: 108 / 125 ≈ 0.864
(|σ|∈[40, 50), Δ=2): Top-1/total: 48 / 75 ≈ 0.64
(|σ|∈[40, 50), Δ=3): Top-1/total: 7 / 24 ≈ 0.2916666666666667
(|σ|∈[50, 60), Δ=1): Top-1/total: 85 / 96 ≈ 0.8854166666666666
(|σ|∈[50, 60), Δ=2): Top-1/total: 44 / 63 ≈ 0.6984126984126984
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 12 ≈ 0.16666666666666666
(|σ|∈[60, 70), Δ=1): Top-1/total: 50 / 60 ≈ 0.8333333333333334
(|σ|∈[60, 70), Δ=2): Top-1/total: 22 / 50 ≈ 0.44
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 15 ≈ 0.26666666666666666
(|σ|∈[70, 80), Δ=1): Top-1/total: 60 / 68 ≈ 0.8823529411764706
(|σ|∈[70, 80), Δ=2): Top-1/total: 23 / 35 ≈ 0.6571428571428571
(|σ|∈[70, 80), Δ=3): Top-1/total: 3 / 15 ≈ 0.2

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 57 / 62 ≈ 0.9193548387096774
|σ|∈[10, 20): Top-1/total: 223 / 298 ≈ 0.7483221476510067
|σ|∈[20, 30): Top-1/total: 306 / 381 ≈ 0.8031496062992126
|σ|∈[30, 40): Top-1/total: 260 / 341 ≈ 0.7624633431085044
|σ|∈[40, 50): Top-1/total: 158 / 217 ≈ 0.728110599078341
|σ|∈[50, 60): Top-1/total: 126 / 164 ≈ 0.7682926829268293
|σ|∈[60, 70): Top-1/total: 75 / 123 ≈ 0.6097560975609756
|σ|∈[70, 80): Top-1/total: 83 / 114 ≈ 0.7280701754385965
Δ(1)= Top-1/total: 881 / 997 ≈ 0.8836509528585758
Δ(2)= Top-1/total: 374 / 542 ≈ 0.6900369003690037
Δ(3)= Top-1/total: 33 / 161 ≈ 0.20496894409937888
(|σ|∈[0, 10), Δ=1): Top-1/total: 32 / 32 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 25 / 27 ≈ 0.9259259259259259
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 3 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 156 / 179 ≈ 0.8715083798882681
(|σ|∈[10, 20), Δ=2): Top-1/total: 66 / 91 ≈ 0.7252747252747253
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 28 ≈ 0.03571428571428571
(|σ|∈[20, 30), Δ=1): Top-1/total: 218 / 242 ≈ 0.9008264462809917
(|σ|∈[20, 30), Δ=2): Top-1/total: 81 / 108 ≈ 0.75
(|σ|∈[20, 30), Δ=3): Top-1/total: 7 / 31 ≈ 0.22580645161290322
(|σ|∈[30, 40), Δ=1): Top-1/total: 181 / 204 ≈ 0.8872549019607843
(|σ|∈[30, 40), Δ=2): Top-1/total: 69 / 101 ≈ 0.6831683168316832
(|σ|∈[30, 40), Δ=3): Top-1/total: 10 / 36 ≈ 0.2777777777777778
(|σ|∈[40, 50), Δ=1): Top-1/total: 106 / 123 ≈ 0.8617886178861789
(|σ|∈[40, 50), Δ=2): Top-1/total: 46 / 71 ≈ 0.647887323943662
(|σ|∈[40, 50), Δ=3): Top-1/total: 6 / 23 ≈ 0.2608695652173913
(|σ|∈[50, 60), Δ=1): Top-1/total: 81 / 92 ≈ 0.8804347826086957
(|σ|∈[50, 60), Δ=2): Top-1/total: 43 / 61 ≈ 0.7049180327868853
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 11 ≈ 0.18181818181818182
(|σ|∈[60, 70), Δ=1): Top-1/total: 50 / 60 ≈ 0.8333333333333334
(|σ|∈[60, 70), Δ=2): Top-1/total: 21 / 48 ≈ 0.4375
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 15 ≈ 0.26666666666666666
(|σ|∈[70, 80), Δ=1): Top-1/total: 57 / 65 ≈ 0.8769230769230769
(|σ|∈[70, 80), Δ=2): Top-1/total: 23 / 35 ≈ 0.6571428571428571
(|σ|∈[70, 80), Δ=3): Top-1/total: 3 / 14 ≈ 0.21428571428571427

w/ Listwise reranker + LED+1 + GRE-NGRAM-1000 reranking

Lev(*): Top-1/rec/pos/total: 162 / 927 / 1637 / 1753, errors: 116, P@1: 0.09241300627495722, P@All: 0.5288077581289219
Lev(1): Top-1/rec/pos/total: 112 / 709 / 1031 / 1031, errors: 0, P@1: 0.10863239573229874, P@All: 0.6876818622696411
Lev(2): Top-1/rec/pos/total: 43 / 190 / 554 / 554, errors: 0, P@1: 0.0776173285198556, P@All: 0.34296028880866425
Lev(3): Top-1/rec/pos/total: 7 / 28 / 52 / 168, errors: 116, P@1: 0.041666666666666664, P@All: 0.16666666666666666
Draw timings (ms): {1=2153.6666666666665, 2=699.168284789644, 3=200.02481121898597}
Full timings (ms): {1=3599.235167206041, 2=1296.7486515641856, 3=346.0053937432578}
Avg samples drawn: {1=7283.741100323625, 2=11884.895361380799, 3=7259.135922330097}

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 23 / 64 ≈ 0.359375
|σ|∈[10, 20): Top-1/total: 55 / 309 ≈ 0.1779935275080906
|σ|∈[20, 30): Top-1/total: 39 / 391 ≈ 0.09974424552429667
|σ|∈[30, 40): Top-1/total: 27 / 353 ≈ 0.0764872521246459
|σ|∈[40, 50): Top-1/total: 10 / 224 ≈ 0.044642857142857144
|σ|∈[50, 60): Top-1/total: 6 / 171 ≈ 0.03508771929824561
|σ|∈[60, 70): Top-1/total: 0 / 125 ≈ 0.0
|σ|∈[70, 80): Top-1/total: 2 / 118 ≈ 0.01694915254237288
Δ(1)= Top-1/total: 112 / 1031 ≈ 0.10863239573229874
Δ(2)= Top-1/total: 43 / 556 ≈ 0.07733812949640288
Δ(3)= Top-1/total: 7 / 168 ≈ 0.041666666666666664
(|σ|∈[0, 10), Δ=1): Top-1/total: 14 / 32 ≈ 0.4375
(|σ|∈[0, 10), Δ=2): Top-1/total: 9 / 28 ≈ 0.32142857142857145
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 38 / 186 ≈ 0.20430107526881722
(|σ|∈[10, 20), Δ=2): Top-1/total: 17 / 95 ≈ 0.17894736842105263
(|σ|∈[10, 20), Δ=3): Top-1/total: 0 / 28 ≈ 0.0
(|σ|∈[20, 30), Δ=1): Top-1/total: 29 / 251 ≈ 0.11553784860557768
(|σ|∈[20, 30), Δ=2): Top-1/total: 6 / 108 ≈ 0.05555555555555555
(|σ|∈[20, 30), Δ=3): Top-1/total: 4 / 32 ≈ 0.125
(|σ|∈[30, 40), Δ=1): Top-1/total: 20 / 213 ≈ 0.09389671361502347
(|σ|∈[30, 40), Δ=2): Top-1/total: 6 / 102 ≈ 0.058823529411764705
(|σ|∈[30, 40), Δ=3): Top-1/total: 1 / 38 ≈ 0.02631578947368421
(|σ|∈[40, 50), Δ=1): Top-1/total: 6 / 125 ≈ 0.048
(|σ|∈[40, 50), Δ=2): Top-1/total: 2 / 75 ≈ 0.02666666666666667
(|σ|∈[40, 50), Δ=3): Top-1/total: 2 / 24 ≈ 0.08333333333333333
(|σ|∈[50, 60), Δ=1): Top-1/total: 3 / 96 ≈ 0.03125
(|σ|∈[50, 60), Δ=2): Top-1/total: 3 / 63 ≈ 0.047619047619047616
(|σ|∈[50, 60), Δ=3): Top-1/total: 0 / 12 ≈ 0.0
(|σ|∈[60, 70), Δ=1): Top-1/total: 0 / 60 ≈ 0.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 0 / 50 ≈ 0.0
(|σ|∈[60, 70), Δ=3): Top-1/total: 0 / 15 ≈ 0.0
(|σ|∈[70, 80), Δ=1): Top-1/total: 2 / 68 ≈ 0.029411764705882353
(|σ|∈[70, 80), Δ=2): Top-1/total: 0 / 35 ≈ 0.0
(|σ|∈[70, 80), Δ=3): Top-1/total: 0 / 15 ≈ 0.0

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 60 / 64 ≈ 0.9375
|σ|∈[10, 20): Top-1/total: 284 / 309 ≈ 0.919093851132686
|σ|∈[20, 30): Top-1/total: 368 / 391 ≈ 0.9411764705882353
|σ|∈[30, 40): Top-1/total: 329 / 353 ≈ 0.9320113314447592
|σ|∈[40, 50): Top-1/total: 208 / 224 ≈ 0.9285714285714286
|σ|∈[50, 60): Top-1/total: 162 / 171 ≈ 0.9473684210526315
|σ|∈[60, 70): Top-1/total: 116 / 125 ≈ 0.928
|σ|∈[70, 80): Top-1/total: 108 / 118 ≈ 0.9152542372881356
Δ(1)= Top-1/total: 1031 / 1031 ≈ 1.0
Δ(2)= Top-1/total: 555 / 556 ≈ 0.9982014388489209
Δ(3)= Top-1/total: 49 / 168 ≈ 0.2916666666666667
(|σ|∈[0, 10), Δ=1): Top-1/total: 32 / 32 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 28 / 28 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 4 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 186 / 186 ≈ 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 95 / 95 ≈ 1.0
(|σ|∈[10, 20), Δ=3): Top-1/total: 3 / 28 ≈ 0.10714285714285714
(|σ|∈[20, 30), Δ=1): Top-1/total: 251 / 251 ≈ 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 108 / 108 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 9 / 32 ≈ 0.28125
(|σ|∈[30, 40), Δ=1): Top-1/total: 213 / 213 ≈ 1.0
(|σ|∈[30, 40), Δ=2): Top-1/total: 101 / 102 ≈ 0.9901960784313726
(|σ|∈[30, 40), Δ=3): Top-1/total: 15 / 38 ≈ 0.39473684210526316
(|σ|∈[40, 50), Δ=1): Top-1/total: 125 / 125 ≈ 1.0
(|σ|∈[40, 50), Δ=2): Top-1/total: 75 / 75 ≈ 1.0
(|σ|∈[40, 50), Δ=3): Top-1/total: 8 / 24 ≈ 0.3333333333333333
(|σ|∈[50, 60), Δ=1): Top-1/total: 96 / 96 ≈ 1.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 63 / 63 ≈ 1.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 3 / 12 ≈ 0.25
(|σ|∈[60, 70), Δ=1): Top-1/total: 60 / 60 ≈ 1.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 50 / 50 ≈ 1.0
(|σ|∈[60, 70), Δ=3): Top-1/total: 6 / 15 ≈ 0.4
(|σ|∈[70, 80), Δ=1): Top-1/total: 68 / 68 ≈ 1.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 35 / 35 ≈ 1.0
(|σ|∈[70, 80), Δ=3): Top-1/total: 5 / 15 ≈ 0.3333333333333333

w/ Regex LBH + Markov chain

Lev(*): Top-1/rec/pos/total: 26 / 1573 / 1700 / 1700, errors: 0, P@1: 0.015294117647058824, P@All: 0.9252941176470588
Lev(1): Top-1/rec/pos/total: 11 / 1007 / 1008 / 1008, errors: 0, P@1: 0.010912698412698412, P@All: 0.9990079365079365
Lev(2): Top-1/rec/pos/total: 13 / 519 / 532 / 532, errors: 0, P@1: 0.02443609022556391, P@All: 0.9755639097744361
Lev(3): Top-1/rec/pos/total: 2 / 47 / 160 / 160, errors: 0, P@1: 0.0125, P@All: 0.29375
Draw timings (ms): {1=1250.0158931977114, 2=1465.9675778766689, 3=313.32676414494597}
Full timings (ms): {1=1250.1627463445645, 2=1466.075651621106, 3=313.3490146217419}
Avg samples drawn: {1=8730.873490146218, 2=8656.591226954863, 3=1553.5276541640178}

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 1 / 61 ≈ 0.01639344262295082
|σ|∈[10, 20): Top-1/total: 7 / 292 ≈ 0.023972602739726026
|σ|∈[20, 30): Top-1/total: 8 / 379 ≈ 0.021108179419525065
|σ|∈[30, 40): Top-1/total: 2 / 345 ≈ 0.005797101449275362
|σ|∈[40, 50): Top-1/total: 5 / 217 ≈ 0.02304147465437788
|σ|∈[50, 60): Top-1/total: 0 / 170 ≈ 0.0
|σ|∈[60, 70): Top-1/total: 3 / 122 ≈ 0.02459016393442623
|σ|∈[70, 80): Top-1/total: 0 / 114 ≈ 0.0
Δ(1)= Top-1/total: 11 / 1008 ≈ 0.010912698412698412
Δ(2)= Top-1/total: 13 / 532 ≈ 0.02443609022556391
Δ(3)= Top-1/total: 2 / 160 ≈ 0.0125
(|σ|∈[0, 10), Δ=1): Top-1/total: 0 / 30 ≈ 0.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 1 / 28 ≈ 0.03571428571428571
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 3 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 3 / 178 ≈ 0.016853932584269662
(|σ|∈[10, 20), Δ=2): Top-1/total: 4 / 87 ≈ 0.04597701149425287
(|σ|∈[10, 20), Δ=3): Top-1/total: 0 / 27 ≈ 0.0
(|σ|∈[20, 30), Δ=1): Top-1/total: 4 / 248 ≈ 0.016129032258064516
(|σ|∈[20, 30), Δ=2): Top-1/total: 3 / 102 ≈ 0.029411764705882353
(|σ|∈[20, 30), Δ=3): Top-1/total: 1 / 29 ≈ 0.034482758620689655
(|σ|∈[30, 40), Δ=1): Top-1/total: 1 / 209 ≈ 0.004784688995215311
(|σ|∈[30, 40), Δ=2): Top-1/total: 1 / 99 ≈ 0.010101010101010102
(|σ|∈[30, 40), Δ=3): Top-1/total: 0 / 37 ≈ 0.0
(|σ|∈[40, 50), Δ=1): Top-1/total: 1 / 122 ≈ 0.00819672131147541
(|σ|∈[40, 50), Δ=2): Top-1/total: 3 / 71 ≈ 0.04225352112676056
(|σ|∈[40, 50), Δ=3): Top-1/total: 1 / 24 ≈ 0.041666666666666664
(|σ|∈[50, 60), Δ=1): Top-1/total: 0 / 96 ≈ 0.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 0 / 63 ≈ 0.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 0 / 11 ≈ 0.0
(|σ|∈[60, 70), Δ=1): Top-1/total: 2 / 60 ≈ 0.03333333333333333
(|σ|∈[60, 70), Δ=2): Top-1/total: 1 / 47 ≈ 0.02127659574468085
(|σ|∈[60, 70), Δ=3): Top-1/total: 0 / 15 ≈ 0.0
(|σ|∈[70, 80), Δ=1): Top-1/total: 0 / 65 ≈ 0.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 0 / 35 ≈ 0.0
(|σ|∈[70, 80), Δ=3): Top-1/total: 0 / 14 ≈ 0.0

Precision@10
===========
|σ|∈[0, 10): Top-1/total: 22 / 61 ≈ 0.36065573770491804
|σ|∈[10, 20): Top-1/total: 60 / 292 ≈ 0.2054794520547945
|σ|∈[20, 30): Top-1/total: 73 / 379 ≈ 0.19261213720316622
|σ|∈[30, 40): Top-1/total: 59 / 345 ≈ 0.17101449275362318
|σ|∈[40, 50): Top-1/total: 46 / 217 ≈ 0.2119815668202765
|σ|∈[50, 60): Top-1/total: 32 / 170 ≈ 0.18823529411764706
|σ|∈[60, 70): Top-1/total: 22 / 122 ≈ 0.18032786885245902
|σ|∈[70, 80): Top-1/total: 15 / 114 ≈ 0.13157894736842105
Δ(1)= Top-1/total: 238 / 1008 ≈ 0.2361111111111111
Δ(2)= Top-1/total: 81 / 532 ≈ 0.15225563909774437
Δ(3)= Top-1/total: 10 / 160 ≈ 0.0625
(|σ|∈[0, 10), Δ=1): Top-1/total: 12 / 30 ≈ 0.4
(|σ|∈[0, 10), Δ=2): Top-1/total: 10 / 28 ≈ 0.35714285714285715
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 3 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 39 / 178 ≈ 0.21910112359550563
(|σ|∈[10, 20), Δ=2): Top-1/total: 20 / 87 ≈ 0.22988505747126436
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 27 ≈ 0.037037037037037035
(|σ|∈[20, 30), Δ=1): Top-1/total: 55 / 248 ≈ 0.2217741935483871
(|σ|∈[20, 30), Δ=2): Top-1/total: 16 / 102 ≈ 0.1568627450980392
(|σ|∈[20, 30), Δ=3): Top-1/total: 2 / 29 ≈ 0.06896551724137931
(|σ|∈[30, 40), Δ=1): Top-1/total: 52 / 209 ≈ 0.24880382775119617
(|σ|∈[30, 40), Δ=2): Top-1/total: 5 / 99 ≈ 0.050505050505050504
(|σ|∈[30, 40), Δ=3): Top-1/total: 2 / 37 ≈ 0.05405405405405406
(|σ|∈[40, 50), Δ=1): Top-1/total: 32 / 122 ≈ 0.26229508196721313
(|σ|∈[40, 50), Δ=2): Top-1/total: 13 / 71 ≈ 0.18309859154929578
(|σ|∈[40, 50), Δ=3): Top-1/total: 1 / 24 ≈ 0.041666666666666664
(|σ|∈[50, 60), Δ=1): Top-1/total: 22 / 96 ≈ 0.22916666666666666
(|σ|∈[50, 60), Δ=2): Top-1/total: 10 / 63 ≈ 0.15873015873015872
(|σ|∈[50, 60), Δ=3): Top-1/total: 0 / 11 ≈ 0.0
(|σ|∈[60, 70), Δ=1): Top-1/total: 15 / 60 ≈ 0.25
(|σ|∈[60, 70), Δ=2): Top-1/total: 4 / 47 ≈ 0.0851063829787234
(|σ|∈[60, 70), Δ=3): Top-1/total: 3 / 15 ≈ 0.2
(|σ|∈[70, 80), Δ=1): Top-1/total: 11 / 65 ≈ 0.16923076923076924
(|σ|∈[70, 80), Δ=2): Top-1/total: 3 / 35 ≈ 0.08571428571428572
(|σ|∈[70, 80), Δ=3): Top-1/total: 1 / 14 ≈ 0.07142857142857142

Precision@100
===========
|σ|∈[0, 10): Top-1/total: 38 / 61 ≈ 0.6229508196721312
|σ|∈[10, 20): Top-1/total: 124 / 292 ≈ 0.4246575342465753
|σ|∈[20, 30): Top-1/total: 187 / 379 ≈ 0.49340369393139843
|σ|∈[30, 40): Top-1/total: 149 / 345 ≈ 0.4318840579710145
|σ|∈[40, 50): Top-1/total: 103 / 217 ≈ 0.47465437788018433
|σ|∈[50, 60): Top-1/total: 81 / 170 ≈ 0.4764705882352941
|σ|∈[60, 70): Top-1/total: 54 / 122 ≈ 0.4426229508196721
|σ|∈[70, 80): Top-1/total: 48 / 114 ≈ 0.42105263157894735
Δ(1)= Top-1/total: 554 / 1008 ≈ 0.5496031746031746
Δ(2)= Top-1/total: 209 / 532 ≈ 0.39285714285714285
Δ(3)= Top-1/total: 21 / 160 ≈ 0.13125
(|σ|∈[0, 10), Δ=1): Top-1/total: 17 / 30 ≈ 0.5666666666666667
(|σ|∈[0, 10), Δ=2): Top-1/total: 21 / 28 ≈ 0.75
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 3 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 87 / 178 ≈ 0.4887640449438202
(|σ|∈[10, 20), Δ=2): Top-1/total: 36 / 87 ≈ 0.41379310344827586
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 27 ≈ 0.037037037037037035
(|σ|∈[20, 30), Δ=1): Top-1/total: 137 / 248 ≈ 0.5524193548387096
(|σ|∈[20, 30), Δ=2): Top-1/total: 43 / 102 ≈ 0.4215686274509804
(|σ|∈[20, 30), Δ=3): Top-1/total: 7 / 29 ≈ 0.2413793103448276
(|σ|∈[30, 40), Δ=1): Top-1/total: 114 / 209 ≈ 0.5454545454545454
(|σ|∈[30, 40), Δ=2): Top-1/total: 31 / 99 ≈ 0.31313131313131315
(|σ|∈[30, 40), Δ=3): Top-1/total: 4 / 37 ≈ 0.10810810810810811
(|σ|∈[40, 50), Δ=1): Top-1/total: 73 / 122 ≈ 0.5983606557377049
(|σ|∈[40, 50), Δ=2): Top-1/total: 28 / 71 ≈ 0.39436619718309857
(|σ|∈[40, 50), Δ=3): Top-1/total: 2 / 24 ≈ 0.08333333333333333
(|σ|∈[50, 60), Δ=1): Top-1/total: 56 / 96 ≈ 0.5833333333333334
(|σ|∈[50, 60), Δ=2): Top-1/total: 24 / 63 ≈ 0.38095238095238093
(|σ|∈[50, 60), Δ=3): Top-1/total: 1 / 11 ≈ 0.09090909090909091
(|σ|∈[60, 70), Δ=1): Top-1/total: 35 / 60 ≈ 0.5833333333333334
(|σ|∈[60, 70), Δ=2): Top-1/total: 15 / 47 ≈ 0.3191489361702128
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 15 ≈ 0.26666666666666666
(|σ|∈[70, 80), Δ=1): Top-1/total: 35 / 65 ≈ 0.5384615384615384
(|σ|∈[70, 80), Δ=2): Top-1/total: 11 / 35 ≈ 0.3142857142857143
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 14 ≈ 0.14285714285714285

Precision@1000
=============
|σ|∈[0, 10): Top-1/total: 56 / 61 ≈ 0.9180327868852459
|σ|∈[10, 20): Top-1/total: 221 / 292 ≈ 0.7568493150684932
|σ|∈[20, 30): Top-1/total: 309 / 379 ≈ 0.8153034300791556
|σ|∈[30, 40): Top-1/total: 262 / 345 ≈ 0.7594202898550725
|σ|∈[40, 50): Top-1/total: 158 / 217 ≈ 0.728110599078341
|σ|∈[50, 60): Top-1/total: 131 / 170 ≈ 0.7705882352941177
|σ|∈[60, 70): Top-1/total: 76 / 122 ≈ 0.6229508196721312
|σ|∈[70, 80): Top-1/total: 82 / 114 ≈ 0.7192982456140351
Δ(1)= Top-1/total: 894 / 1008 ≈ 0.8869047619047619
Δ(2)= Top-1/total: 369 / 532 ≈ 0.693609022556391
Δ(3)= Top-1/total: 32 / 160 ≈ 0.2
(|σ|∈[0, 10), Δ=1): Top-1/total: 30 / 30 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 26 / 28 ≈ 0.9285714285714286
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 3 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 157 / 178 ≈ 0.8820224719101124
(|σ|∈[10, 20), Δ=2): Top-1/total: 63 / 87 ≈ 0.7241379310344828
(|σ|∈[10, 20), Δ=3): Top-1/total: 1 / 27 ≈ 0.037037037037037035
(|σ|∈[20, 30), Δ=1): Top-1/total: 224 / 248 ≈ 0.9032258064516129
(|σ|∈[20, 30), Δ=2): Top-1/total: 78 / 102 ≈ 0.7647058823529411
(|σ|∈[20, 30), Δ=3): Top-1/total: 7 / 29 ≈ 0.2413793103448276
(|σ|∈[30, 40), Δ=1): Top-1/total: 184 / 209 ≈ 0.8803827751196173
(|σ|∈[30, 40), Δ=2): Top-1/total: 68 / 99 ≈ 0.6868686868686869
(|σ|∈[30, 40), Δ=3): Top-1/total: 10 / 37 ≈ 0.2702702702702703
(|σ|∈[40, 50), Δ=1): Top-1/total: 106 / 122 ≈ 0.8688524590163934
(|σ|∈[40, 50), Δ=2): Top-1/total: 46 / 71 ≈ 0.647887323943662
(|σ|∈[40, 50), Δ=3): Top-1/total: 6 / 24 ≈ 0.25
(|σ|∈[50, 60), Δ=1): Top-1/total: 85 / 96 ≈ 0.8854166666666666
(|σ|∈[50, 60), Δ=2): Top-1/total: 44 / 63 ≈ 0.6984126984126984
(|σ|∈[50, 60), Δ=3): Top-1/total: 2 / 11 ≈ 0.18181818181818182
(|σ|∈[60, 70), Δ=1): Top-1/total: 50 / 60 ≈ 0.8333333333333334
(|σ|∈[60, 70), Δ=2): Top-1/total: 22 / 47 ≈ 0.46808510638297873
(|σ|∈[60, 70), Δ=3): Top-1/total: 4 / 15 ≈ 0.26666666666666666
(|σ|∈[70, 80), Δ=1): Top-1/total: 58 / 65 ≈ 0.8923076923076924
(|σ|∈[70, 80), Δ=2): Top-1/total: 22 / 35 ≈ 0.6285714285714286
(|σ|∈[70, 80), Δ=3): Top-1/total: 2 / 14 ≈ 0.14285714285714285

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 58 / 61 ≈ 0.9508196721311475
|σ|∈[10, 20): Top-1/total: 266 / 292 ≈ 0.910958904109589
|σ|∈[20, 30): Top-1/total: 359 / 379 ≈ 0.9472295514511874
|σ|∈[30, 40): Top-1/total: 318 / 345 ≈ 0.9217391304347826
|σ|∈[40, 50): Top-1/total: 200 / 217 ≈ 0.9216589861751152
|σ|∈[50, 60): Top-1/total: 161 / 170 ≈ 0.9470588235294117
|σ|∈[60, 70): Top-1/total: 111 / 122 ≈ 0.9098360655737705
|σ|∈[70, 80): Top-1/total: 100 / 114 ≈ 0.8771929824561403
Δ(1)= Top-1/total: 1007 / 1008 ≈ 0.9990079365079365
Δ(2)= Top-1/total: 519 / 532 ≈ 0.9755639097744361
Δ(3)= Top-1/total: 47 / 160 ≈ 0.29375
(|σ|∈[0, 10), Δ=1): Top-1/total: 30 / 30 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 28 / 28 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 3 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 178 / 178 ≈ 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 85 / 87 ≈ 0.9770114942528736
(|σ|∈[10, 20), Δ=3): Top-1/total: 3 / 27 ≈ 0.1111111111111111
(|σ|∈[20, 30), Δ=1): Top-1/total: 248 / 248 ≈ 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 101 / 102 ≈ 0.9901960784313726
(|σ|∈[20, 30), Δ=3): Top-1/total: 10 / 29 ≈ 0.3448275862068966
(|σ|∈[30, 40), Δ=1): Top-1/total: 208 / 209 ≈ 0.9952153110047847
(|σ|∈[30, 40), Δ=2): Top-1/total: 97 / 99 ≈ 0.9797979797979798
(|σ|∈[30, 40), Δ=3): Top-1/total: 13 / 37 ≈ 0.35135135135135137
(|σ|∈[40, 50), Δ=1): Top-1/total: 122 / 122 ≈ 1.0
(|σ|∈[40, 50), Δ=2): Top-1/total: 70 / 71 ≈ 0.9859154929577465
(|σ|∈[40, 50), Δ=3): Top-1/total: 8 / 24 ≈ 0.3333333333333333
(|σ|∈[50, 60), Δ=1): Top-1/total: 96 / 96 ≈ 1.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 62 / 63 ≈ 0.9841269841269841
(|σ|∈[50, 60), Δ=3): Top-1/total: 3 / 11 ≈ 0.2727272727272727
(|σ|∈[60, 70), Δ=1): Top-1/total: 60 / 60 ≈ 1.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 44 / 47 ≈ 0.9361702127659575
(|σ|∈[60, 70), Δ=3): Top-1/total: 7 / 15 ≈ 0.4666666666666667
(|σ|∈[70, 80), Δ=1): Top-1/total: 65 / 65 ≈ 1.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 32 / 35 ≈ 0.9142857142857143
(|σ|∈[70, 80), Δ=3): Top-1/total: 3 / 14 ≈ 0.21428571428571427

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