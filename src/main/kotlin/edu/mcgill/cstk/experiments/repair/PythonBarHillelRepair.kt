package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.types.to
import ai.hypergraph.kaliningraph.visualization.show
import edu.mcgill.cstk.experiments.repair.sizeAndDistBalancedRepairsUnminimized
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
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
  evaluateBarHillelRepairOnStackOverflow()
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
  readResourceFile("models/pcfg3_BIFI.csv")
  .lines().map { it.split(" ::: ") }.associate { Pair(it[0].split(" ").let { it[0] to it[1] to it[2] }, it[1].toInt()) }

fun readPCFG5(s2pg: CFG): Map<Int, Int> =
  readResourceFile("models/pcfg5_BIFI.csv")
    .lines().map { it.split(" ::: ") }
    .associate { Pair(it[0].split(" ")
      .map { if (it.endsWith('*') && it.length > 1) (31 * s2pg.ntMap[it.dropLast(1)]!!) else s2pg.ntMap[it] ?: Int.MAX_VALUE }
      /** See [Tree.quintuples] */
      .let { hash(it[0], it[1], it[2], it[3], it[4]) }, it[1].toInt()) }

fun evaluateBarHillelRepairOnStackOverflow() {
  val dataset = sizeAndDistBalancedRepairsUnminimized//corruptedBIFIGoodCode//sizeAndDistBalancedRepairsUnminimized.toList()
   // timeoutCases // corruptedBIFIGoodCode // balancedSmallRepairsUnminimized.toList() // naturallySmallRepairs //pairwiseUniformAll
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val s2pg = vanillaS2PCFG
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

  dataset.forEach { (invalidTokens, validTokens) ->
    val allTime = TimeSource.Monotonic.markNow()
    val toRepair = invalidTokens.tokenizeByWhitespace()
    val humanRepair = validTokens.tokenizeByWhitespace()
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
      val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(toRepair, pad = levDist)
//    val multiEditBounds = vanillaS2PCFGWE.findMinimalMultiEditBounds(toRepair, monoEditBounds, levDist)
      val fsa = makeLevFSA(toRepair, levDist, monoEditBounds).also { levBallSize = it.Q.size }

      if (!fsa.recognizes(humanRepair))
        throw Exception("Human repair is unrecognizable! (Total time=${allTime.elapsedNow()})")
      else println("LEV-FSA recognizes human repair (Total time=${allTime.elapsedNow()})")

      s2pg.jvmIntersectLevFSAP(fsa = fsa, parikhMap = parikhMap)
        .also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
    } catch (e: Exception) { println("$humanRepairANSI\nIntersection exception: ${e.stackTraceToString()}"); null }
    catch (e: Error) { println("$humanRepairANSI\nIntersection error: ${e.stackTraceToString()}"); null }

    if (intGram != null) println("Constructed LEV($levDist, ${toRepair.size}, $levBallSize) " +
      "∩ CFG grammar with ${intGram.size} productions in ${allTime.elapsedNow()}")

    println("Implicated nonterminals: " + (intGram?.nonterminals?.map { if(it == "START") it else it.split("~")[1] }?.toSet()?.size ?: 0) + " / " + s2pg.nonterminals.size)

    try {
      if (intGram == null) throw Exception("Exception while building grammar!")
      else if (30_000 < intGram.size) throw Exception("Int grammar was still too large!")
      else if (humanRepair !in intGram.language) {
        println("Human repair recognized by original CFG: " + (humanRepair in vanillaS2PCFG.language))
        throw Exception("Human repair is unrecognizable by LEV ∩ CFG! (Total time=${allTime.elapsedNow()})")
      } else println("Human repair is recognized by LEV ∩ CFG! (Total time=${allTime.elapsedNow()})")
    } catch (e: Exception) {
      println("Encountered error ${e.message} ${allTime.elapsedNow()}):\n$humanRepairANSI\n${e.stackTraceToString()}")
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

//    val results = ConcurrentRankedProbabilisticSet<Σᐩ>(MAX_UNIQUE)
////      if (intGram.size < CFG_THRESH) {
////        println("Small grammar, sampling without replacement...")
//        pTree
////          .sampleDirectlyWOR(stoppingCriterion = { clock.elapsedNow() < timeout })
//          .sampleDirectlyWORAndScore(stoppingCriterion = { clock.elapsedNow() < timeout }, pcfgMap = pcfgMap, pcfgNorm = pcfgNorm)
////      } else {
////        println("Large grammar, sampling with replacement using PCFG...")
////        pTree.sampleWithPCFG(pcfgMap, stoppingCriterion = { clock.elapsedNow() < timeout })
//  //        .map { println(levenshteinAlign(source, it).paintANSIColors()); it }
////      }
//          .map {
//            totalSamples.incrementAndGet()
//            if (it.first == target) { matchFound = true; elapsed = clock.elapsedNow().inWholeMilliseconds }
////            results.add(it.first, P_BIFI_PY150.score(it.first.tokenizeByWhitespace()))
//            // PCFG likelihood reranker
//            results.add(it.first, -it.second + P_BIFI_PY150.score(it.first.tokenizeByWhitespace()))
//          }
//          .toList()
//
//    println("Found $totalSamples samples in ${clock.elapsedNow()}")
//
//    val rankedResults = results.mostLikely.entries.map { it.value }
//
//    println("Ranked ${results.size} samples in ${clock.elapsedNow()}")

    val dfa = pTree.toDFA(minimize = false)!!

//    println(dfa.toDot().replaceAll(vanillaS2PCFG.unicodeMap))

    val dfaRecognized = try { dfa.run(pTree.termDict.encode(humanRepair)) } catch (_: Exception) { false }
    println("∩-DFA ${if (dfaRecognized) "accepted" else "rejected"} human repair! (Total time=${allTime.elapsedNow()})")

    val rankedResults = dfa.decodeDFAWithBeamSearch(
      mc = P_BIFI_PY150,
      timeout = timeout,
      dec = pTree.termDict,
      callback = {
        totalSamples.incrementAndGet()
        if (it == target) {
          matchFound = true
          println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
          elapsed = clock.elapsedNow().inWholeMilliseconds
        }
      }
    )

//    rankedResults.take(100).forEach {
//      println("Sample: ${levenshteinAlign(humanRepair, it.tokenizeByWhitespace()).paintANSIColors()}")
//      println(it in vanillaS2PCFG.language)
//    }

    val indexOfTarget = rankedResults.indexOf(target).also {
      if (it == 0) P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      if (matchFound) P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
    }

    rankedResults.firstOrNull()?.tokenizeByWhitespace()
      ?.let { println("Top1 scoring repair: ${levenshteinAlign(toRepair, it).paintANSIColors()}") }

    if (indexOfTarget < 0) {
      println("Drew $totalSamples samples in ${clock.elapsedNow()}/$timeout with ${intGram.size} prods, " +
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
  "NAME ( STRING . NAME ( NAME & NAME ) or STRING ) NEWLINE"
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

val sizeAndDistBalancedRepairsUnminimized: Sequence<Π4A<Σᐩ>> by lazy {
//  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
//  val file = File(File("").absolutePath + path).readText()
  val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
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
  readBIFIContents()
    .map { it.mapToUnquotedPythonTokens().addNewLineIfMissing() }
    .filter {
      it.tokenizeByWhitespace().size in MIN_TOKENS..MAX_TOKENS &&
        it in vanillaS2PCFG.language
    }
    .flatMap { goodCode ->
      goodCode.naturalPythonCorruptions().distinct().filter {
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
    "Top-1/total: $top1 / $total ≈ ${top1.toDouble() / total}"
}

/*
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

// w/ Language edit distance radius

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 61 / 257 ≈ 0.23735408560311283
|σ|∈[10, 20): Top-1/total: 60 / 293 ≈ 0.20477815699658702
|σ|∈[20, 30): Top-1/total: 57 / 294 ≈ 0.19387755102040816
|σ|∈[30, 40): Top-1/total: 37 / 78 ≈ 0.47435897435897434
Δ(1)= Top-1/total: 168 / 763 ≈ 0.22018348623853212
Δ(2)= Top-1/total: 37 / 136 ≈ 0.27205882352941174
Δ(3)= Top-1/total: 10 / 23 ≈ 0.43478260869565216
(|σ|∈[0, 10), Δ=1): Top-1/total: 52 / 221 ≈ 0.23529411764705882
(|σ|∈[0, 10), Δ=2): Top-1/total: 9 / 35 ≈ 0.2571428571428571
(|σ|∈[0, 10), Δ=3): Top-1/total: 0 / 1 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 46 / 245 ≈ 0.18775510204081633
(|σ|∈[10, 20), Δ=2): Top-1/total: 9 / 38 ≈ 0.23684210526315788
(|σ|∈[10, 20), Δ=3): Top-1/total: 5 / 10 ≈ 0.5
(|σ|∈[20, 30), Δ=1): Top-1/total: 41 / 239 ≈ 0.17154811715481172
(|σ|∈[20, 30), Δ=2): Top-1/total: 12 / 46 ≈ 0.2608695652173913
(|σ|∈[20, 30), Δ=3): Top-1/total: 4 / 9 ≈ 0.4444444444444444
(|σ|∈[30, 40), Δ=1): Top-1/total: 29 / 58 ≈ 0.5
(|σ|∈[30, 40), Δ=2): Top-1/total: 7 / 17 ≈ 0.4117647058823529
(|σ|∈[30, 40), Δ=3): Top-1/total: 1 / 3 ≈ 0.3333333333333333

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 112 / 257 ≈ 0.4357976653696498
|σ|∈[10, 20): Top-1/total: 129 / 293 ≈ 0.4402730375426621
|σ|∈[20, 30): Top-1/total: 128 / 294 ≈ 0.43537414965986393
|σ|∈[30, 40): Top-1/total: 55 / 78 ≈ 0.7051282051282052
Δ(1)= Top-1/total: 335 / 763 ≈ 0.43905635648754915
Δ(2)= Top-1/total: 66 / 136 ≈ 0.4852941176470588
Δ(3)= Top-1/total: 23 / 23 ≈ 1.0
(|σ|∈[0, 10), Δ=1): Top-1/total: 98 / 221 ≈ 0.4434389140271493
(|σ|∈[0, 10), Δ=2): Top-1/total: 13 / 35 ≈ 0.37142857142857144
(|σ|∈[0, 10), Δ=3): Top-1/total: 1 / 1 ≈ 1.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 99 / 245 ≈ 0.40408163265306124
(|σ|∈[10, 20), Δ=2): Top-1/total: 20 / 38 ≈ 0.5263157894736842
(|σ|∈[10, 20), Δ=3): Top-1/total: 10 / 10 ≈ 1.0
(|σ|∈[20, 30), Δ=1): Top-1/total: 98 / 239 ≈ 0.4100418410041841
(|σ|∈[20, 30), Δ=2): Top-1/total: 21 / 46 ≈ 0.45652173913043476
(|σ|∈[20, 30), Δ=3): Top-1/total: 9 / 9 ≈ 1.0
(|σ|∈[30, 40), Δ=1): Top-1/total: 40 / 58 ≈ 0.6896551724137931
(|σ|∈[30, 40), Δ=2): Top-1/total: 12 / 17 ≈ 0.7058823529411765
(|σ|∈[30, 40), Δ=3): Top-1/total: 3 / 3 ≈ 1.0
 */