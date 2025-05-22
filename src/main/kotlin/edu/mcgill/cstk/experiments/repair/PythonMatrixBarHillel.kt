package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.experiments.probing.MakeMore
import edu.mcgill.cstk.utils.lastGitMessage
import java.io.File
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds

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

    val humanRepairANSI = levAlign.paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")

    // Declare the number of edits we are going to make up front
    val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(brokeToks, pad = 2 * MAX_RADIUS)
    val langEditDist = FSA.LED(s2pg, brokeToks, monoEditBounds = monoEditBounds)
    val levGuess = langEditDist + 1//levAlign.patchSize()

    val levDist = levAlign.patchSize() // True distance, only used for logging purposes
    println("Predicted edit dist: $levGuess (true dist: $levDist, LED: $langEditDist)")

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    var levBallSize = 1
    allRate.total++; levRates.getOrPut(levDist) { LBHMetrics() }.total++

    fun failed(msg: Σᐩ?, st: Σᐩ) {
      println("Encountered error $msg ${allTime.elapsedNow()}):\n$humanRepairANSI\n$st")
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
    }

    try {
//    val multiEditBounds = vanillaS2PCFGWE.findMinimalMultiEditBounds(toRepair, monoEditBounds, levDist)
      val fsa = makeLevFSA(brokeToks, levGuess, monoEditBounds).also { levBallSize = it.Q.size }

//      val tt = measureTimedValue { jvmIntersectDFA(brokeToks, s2pg, levGuess, fsa, parikhMap) }
//      println("Constructed DFA in ${tt.duration}")
//      val dfa = tt.value!!
//      val intGramSize = 0
//      val langSize = 0

      val tt = measureTimedValue { jvmIntersectPTree(brokeToks, s2pg, levGuess, fsa, parikhMap) }
      val pTree = tt.value!!
      val icfg = pTree.toCFG.freeze()
      val icfgRecognized = fixedToks in icfg.language
      val intGramSize = icfg.size
      val icfgpt = icfg.toPTree()
      val langSize = icfgpt.totalTreesStr
      println("Constructed PTree in ${tt.duration} with $intGramSize productions and $langSize trees")
//
      val dfa = icfgpt.toDFA(minimize = true)!!

      val dfaRecognized = try { dfa.run(termDict.encode(fixedToks)) } catch (_: Exception) { false }
//      println("∩-CFG ${if (icfgRecognized) "accepted" else "rejected"} human repair!")
      println("∩-DFA ${if (dfaRecognized) "accepted" else "rejected"} human repair!")
//      if (!dfaRecognized || !icfgRecognized) { throw Exception("Unrecognizable repair!") }

      val clock = TimeSource.Monotonic.markNow()
      var totalSamples = 0
      var matchFound = false
      val timeout = (TIMEOUT_MS / 1000).seconds
      var elapsed = clock.elapsedNow().inWholeMilliseconds


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
        println("Drew $totalSamples samples in ${clock.elapsedNow()}/$timeout with $intGramSize prods, " +
            //        "${dfa.numStates} states, ${dfa.numberOfTransitions} transitions, " +
            "length-$levDist human repair not found")
        negative.appendText(
          "${brokeToks.size}, $levDist, $totalSamples, ${levBallSize}, " +
              "$intGramSize, $langSize, " +
              //      "${dfa.numStates}, ${dfa.numberOfTransitions}, " +
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
        println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / levRates[it.key]!!.recall }}")
        allTimeByLevDist[levDist] = (allTimeByLevDist[levDist] ?: 0.0) + allElapsed
        println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / levRates[it.key]!!.recall }}")
        samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + totalSamples
        println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / levRates[it.key]!!.recall }}")
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
    catch (e: Exception) { failed(e.message, e.stackTraceToString()) }
    catch (e: Error) { failed(e.message, e.stackTraceToString()) }
  }
}

/** Parallel version of [FSA.intersectPTree] */
fun jvmIntersectPTree(brokenStr: List<Σᐩ>, cfg: CFG, radius: Int,
                   levFSA: FSA = makeLevFSA(brokenStr, radius),
                   pm: ParikhMap): PTree? {
  val timer = TimeSource.Monotonic.markNow()
  val bindex = cfg.bindex
  val bimap = cfg.bimap
  val width = cfg.nonterminals.size
  val vindex = cfg.vindex
  val parikhMap = pm

  val nStates = levFSA.numStates
  val startIdx = bindex[START_SYMBOL]

  // 1) Create dp array of parse trees
  val dp: Array<Array<Array<PTree?>>> = Array(nStates) { Array(nStates) { Array(width) { null } } }

  // 2) Initialize terminal productions A -> a
  val aitx = levFSA.allIndexedTxs1(cfg.unitProductions)
  aitx.parallelStream().forEach { (p, σ, q) ->
    val Aidxs = bimap.TDEPS[σ]!!.map { bindex[it] }
    for (Aidx in Aidxs) {
      val newLeaf = PTree(root = "[$p~${bindex[Aidx]}~$q]", branches = PSingleton(σ))
      dp[p][q][Aidx] = newLeaf + dp[p][q][Aidx]
    }
  }

  fun computeNTCompat(cfg: CFG, levStr: List<Σᐩ>): Array<Array<Array<Boolean>>> {
    val tbl = cfg.parseTableBln(levStr)
    val arr = Array(tbl.numRows) { Array(tbl.numCols) { Array(cfg.nonterminals.size) { false } } }
    for (r in 0 until tbl.numRows)
      for (c in r until tbl.numCols)
        for (k in cfg.nonterminals.indices)
          arr[r][c][k] = tbl[r, c][k]

    return arr
  }

  fun FSA.compat(a: STC, b: STC, nt: Int, compat: Array<Array<Array<Boolean>>>) =
    if (a.π3 != b.π3) true else compat[a.π2][b.π2][nt]

  val ct = (levFSA.validPairs * cfg.nonterminals.indices.toSet()).toList()
  val ct2 = Array(levFSA.numStates) { Array(cfg.nonterminals.size) { Array(levFSA.numStates) { false } } }
  val compat: Array<Array<Array<Boolean>>> = computeNTCompat(cfg, levFSA.levString)
  ct.parallelStream()
    .filter { it: Π3<STC, STC, Int> ->
      // Checks whether the distinct subtrajectory between two horizontal states is parseable by a given NT
      levFSA.compat(it.π1, it.π2, it.π3, compat)
          // Checks whether the length bounds for the nonterminal (i.e., the range of the number of terminals it can
          // parse) is compatible with the range of path lengths across all paths connecting two states in an FSA.
          // This is a coarse approximation, but is cheaper to compute, so it filters out most invalid triples.
          && parikhMap.ntLengthBounds[it.π3].overlaps(SPLPArith(it.π1, it.π2))
          // Checks the Parikh map for compatibility between the CFG nonterminals and state pairs in the FSA.
          // This is a finer grained filter, but more expensive to compute, so we use the coarse filter first
          && levFSA.obeys(it.π1, it.π2, it.π3, parikhMap)
    }
//    .toList().also {
//      val candidates = (fsa.numStates * nonterminals.size * fsa.numStates)
//      val fraction = it.size.toDouble() / candidates
//      println("Fraction of valid LBH triples: ${it.size}/$candidates ≈ $fraction")
//    }
    .forEach { ct2[it.π1.π1][it.π3][it.π2.π1] = true }

  // 3) CYK + Floyd Warshall parsing
  for (dist in 0 until nStates) {
    (0 until (nStates - dist)).toList().parallelStream().forEach { p ->
//    for (p in 0 until (nStates - dist)) {
      val q = p + dist
//      if (p to q !in levFSA.allPairs) continue
      if (levFSA.allPairs[p][q] == null) return@forEach
      val appq = levFSA.allPairs[p][q]!!
//      vindex.withIndex().toList().parallelStream().forEach { (Aidx, indexArray) ->
//      if (!ct2[p][Aidx][q]) return@forEach
      for ((Aidx, indexArray) in vindex.withIndex()) {
        if (!ct2[p][Aidx][q]) continue
        val rhsPairs = dp[p][q][Aidx]?.branches?.toMutableList() ?: mutableListOf()
        outerLoop@for (j in 0..<indexArray.size step 2) {
          val Bidx = indexArray[j]
          val Cidx = indexArray[j + 1]
          for (r in appq) {
            val left = dp[p][r][Bidx]
            val right = dp[r][q][Cidx]
            if (left != null && right != null) {
              // Found a parse for A
              rhsPairs += left to right
//              if (rhsPairs.size > 10) break@outerLoop
            }
          }
        }

        if (rhsPairs.isNotEmpty()) dp[p][q][Aidx] = PTree("[$p~${bindex[Aidx]}~$q]", rhsPairs)
      }
    }
  }

  println("Completed parse matrix in: ${timer.elapsedNow()}")

  // 4) Gather final parse trees from dp[0][f][startIdx], for all final states f
  val allParses = levFSA.finalIdxs.mapNotNull { q -> dp[0][q][startIdx] }

  return PTree(START_SYMBOL, allParses.flatMap { forest -> forest.branches })
}

/*
MatrixLBH + Markov Chain + LED+1

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 52 / 375 ≈ 0.13866666666666666
|σ|∈[10, 20): Top-1/total: 60 / 770 ≈ 0.07792207792207792
|σ|∈[20, 30): Top-1/total: 53 / 767 ≈ 0.06910039113428944
|σ|∈[30, 40): Top-1/total: 34 / 770 ≈ 0.04415584415584416
|σ|∈[40, 50): Top-1/total: 20 / 717 ≈ 0.02789400278940028
|σ|∈[50, 60): Top-1/total: 25 / 682 ≈ 0.036656891495601175
|σ|∈[60, 70): Top-1/total: 22 / 585 ≈ 0.037606837606837605
|σ|∈[70, 80): Top-1/total: 17 / 467 ≈ 0.03640256959314775
Δ(1)= Top-1/total: 16 / 2158 ≈ 0.0074142724745134385
Δ(2)= Top-1/total: 186 / 1911 ≈ 0.09733124018838304
Δ(3)= Top-1/total: 81 / 1064 ≈ 0.07612781954887218
(|σ|∈[0, 10), Δ=1): Top-1/total: 9 / 184 ≈ 0.04891304347826087
(|σ|∈[0, 10), Δ=2): Top-1/total: 37 / 133 ≈ 0.2781954887218045
(|σ|∈[0, 10), Δ=3): Top-1/total: 6 / 58 ≈ 0.10344827586206896
(|σ|∈[10, 20), Δ=1): Top-1/total: 5 / 299 ≈ 0.016722408026755852
(|σ|∈[10, 20), Δ=2): Top-1/total: 42 / 297 ≈ 0.1414141414141414
(|σ|∈[10, 20), Δ=3): Top-1/total: 13 / 174 ≈ 0.07471264367816093
(|σ|∈[20, 30), Δ=1): Top-1/total: 1 / 293 ≈ 0.0034129692832764505
(|σ|∈[20, 30), Δ=2): Top-1/total: 36 / 292 ≈ 0.1232876712328767
(|σ|∈[20, 30), Δ=3): Top-1/total: 16 / 182 ≈ 0.08791208791208792
(|σ|∈[30, 40), Δ=1): Top-1/total: 0 / 287 ≈ 0.0
(|σ|∈[30, 40), Δ=2): Top-1/total: 16 / 291 ≈ 0.054982817869415807
(|σ|∈[30, 40), Δ=3): Top-1/total: 18 / 192 ≈ 0.09375
(|σ|∈[40, 50), Δ=1): Top-1/total: 1 / 288 ≈ 0.003472222222222222
(|σ|∈[40, 50), Δ=2): Top-1/total: 12 / 291 ≈ 0.041237113402061855
(|σ|∈[40, 50), Δ=3): Top-1/total: 7 / 138 ≈ 0.050724637681159424
(|σ|∈[50, 60), Δ=1): Top-1/total: 0 / 285 ≈ 0.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 18 / 265 ≈ 0.06792452830188679
(|σ|∈[50, 60), Δ=3): Top-1/total: 7 / 132 ≈ 0.05303030303030303
(|σ|∈[60, 70), Δ=1): Top-1/total: 0 / 278 ≈ 0.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 15 / 197 ≈ 0.07614213197969544
(|σ|∈[60, 70), Δ=3): Top-1/total: 7 / 110 ≈ 0.06363636363636363
(|σ|∈[70, 80), Δ=1): Top-1/total: 0 / 244 ≈ 0.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 10 / 145 ≈ 0.06896551724137931
(|σ|∈[70, 80), Δ=3): Top-1/total: 7 / 78 ≈ 0.08974358974358974

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 339 / 375 ≈ 0.904
|σ|∈[10, 20): Top-1/total: 648 / 770 ≈ 0.8415584415584415
|σ|∈[20, 30): Top-1/total: 641 / 767 ≈ 0.8357235984354628
|σ|∈[30, 40): Top-1/total: 660 / 770 ≈ 0.8571428571428571
|σ|∈[40, 50): Top-1/total: 633 / 717 ≈ 0.8828451882845189
|σ|∈[50, 60): Top-1/total: 606 / 682 ≈ 0.8885630498533724
|σ|∈[60, 70): Top-1/total: 520 / 585 ≈ 0.8888888888888888
|σ|∈[70, 80): Top-1/total: 422 / 467 ≈ 0.9036402569593148
Δ(1)= Top-1/total: 2158 / 2158 ≈ 1.0
Δ(2)= Top-1/total: 1906 / 1911 ≈ 0.9973835688121402
Δ(3)= Top-1/total: 405 / 1064 ≈ 0.3806390977443609
(|σ|∈[0, 10), Δ=1): Top-1/total: 184 / 184 ≈ 1.0
(|σ|∈[0, 10), Δ=2): Top-1/total: 133 / 133 ≈ 1.0
(|σ|∈[0, 10), Δ=3): Top-1/total: 22 / 58 ≈ 0.3793103448275862
(|σ|∈[10, 20), Δ=1): Top-1/total: 299 / 299 ≈ 1.0
(|σ|∈[10, 20), Δ=2): Top-1/total: 296 / 297 ≈ 0.9966329966329966
(|σ|∈[10, 20), Δ=3): Top-1/total: 53 / 174 ≈ 0.3045977011494253
(|σ|∈[20, 30), Δ=1): Top-1/total: 293 / 293 ≈ 1.0
(|σ|∈[20, 30), Δ=2): Top-1/total: 292 / 292 ≈ 1.0
(|σ|∈[20, 30), Δ=3): Top-1/total: 56 / 182 ≈ 0.3076923076923077
(|σ|∈[30, 40), Δ=1): Top-1/total: 287 / 287 ≈ 1.0
(|σ|∈[30, 40), Δ=2): Top-1/total: 290 / 291 ≈ 0.9965635738831615
(|σ|∈[30, 40), Δ=3): Top-1/total: 83 / 192 ≈ 0.4322916666666667
(|σ|∈[40, 50), Δ=1): Top-1/total: 288 / 288 ≈ 1.0
(|σ|∈[40, 50), Δ=2): Top-1/total: 290 / 291 ≈ 0.9965635738831615
(|σ|∈[40, 50), Δ=3): Top-1/total: 55 / 138 ≈ 0.39855072463768115
(|σ|∈[50, 60), Δ=1): Top-1/total: 285 / 285 ≈ 1.0
(|σ|∈[50, 60), Δ=2): Top-1/total: 265 / 265 ≈ 1.0
(|σ|∈[50, 60), Δ=3): Top-1/total: 56 / 132 ≈ 0.42424242424242425
(|σ|∈[60, 70), Δ=1): Top-1/total: 278 / 278 ≈ 1.0
(|σ|∈[60, 70), Δ=2): Top-1/total: 196 / 197 ≈ 0.9949238578680203
(|σ|∈[60, 70), Δ=3): Top-1/total: 46 / 110 ≈ 0.41818181818181815
(|σ|∈[70, 80), Δ=1): Top-1/total: 244 / 244 ≈ 1.0
(|σ|∈[70, 80), Δ=2): Top-1/total: 144 / 145 ≈ 0.993103448275862
(|σ|∈[70, 80), Δ=3): Top-1/total: 34 / 78 ≈ 0.4358974358974359

BUILD SUCCESSFUL in 4h 33m 14s
 */