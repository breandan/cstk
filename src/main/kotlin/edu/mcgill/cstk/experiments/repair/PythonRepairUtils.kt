package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.approximations.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.experiments.probing.MakeMore
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.*
import java.util.concurrent.locks.ReentrantLock
import java.util.stream.Stream
import kotlin.concurrent.withLock
import kotlin.math.absoluteValue
import kotlin.streams.asStream
import kotlin.text.contains
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds


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
fun readResourceBytes(path: String) = object {}.javaClass.classLoader.getResource(path)!!.readBytes()

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

val pythonPDFA: WFA by lazy { readResourceBytes("models/wfa_ckpt_90000.safetensors").toWFA() }

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

private val fileLock = ReentrantLock()

fun writeToFileWithThreadLock(filePath: String, data: String) {
  fileLock.withLock {
    // Synchronized block for thread-safe writing
    File(filePath).appendText(data, Charsets.UTF_8)
  }
}


fun List<String>.filterErrors(clock: TimeSource.Monotonic.ValueTimeMark): List<String> {
  var filtered = 0
  val s = asSequence().asStream().parallel()
//    .map { it to "".let {
      .map { fix -> fix to getOutput(fix).let {
//        if ("SyntaxError: invalid syntax" in it && fix !in cfg.language) println("$fix")
        if (it.isEmpty()) ""
        else it.trim().replace("\n", "\\n")
          .let { op -> op.getPyErrorType() + ": " + op.getPyErrorMessage() }
      }
    }
    .limit(10_000)
    .toList().let { errorsAndRepairs ->
        val errHst = mutableMapOf<String, Int>()
        val pad = (errHst.values.maxOrNull()?.toString()?.length ?: 1) + 1
        errorsAndRepairs.filter { it.second.isNotEmpty() }
          .forEach { it.second.also { errHst[it] = 1 + errHst.getOrElse(it) { 0 } } }
        val summary = errHst.toMap().entries.sortedBy { -it.component2() }.take(10)
          .joinToString("\n") { "${it.value.toString().padEnd(pad)}| ${it.key}" }
        println("Rejection histogram:\n$summary")
        errorsAndRepairs.filter { it.second.isNotEmpty() }.also { ls ->
          ls.mapIndexed { i, it -> it to i }
            .joinToString("\n") { "${it.first}\n${"E(${it.third}/${ls.size})".padEnd(15)}| ${it.second}" }
            .also { File("error_messages.log").apply { createNewFile() }.appendText(it) }
        }
        errorsAndRepairs.map { it.first }.filter { ("Error" !in it).also { if (!it) filtered++ } }
    }

  return s.also { println("Filtered out $filtered invalid samples! (in ${clock.elapsedNow()})") }
}

fun String.scoreWithPDFA(): Double = -pythonPDFA.scoreString(this)

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
    val levGuess = langEditDist + 1// levAlign.patchSize() //min(predDist, langEditDist)

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
    ).let { unrankedResults ->
      rerankGPU(brokeStr, unrankedResults.take(10_000).joinToString("\n"))
        .map { it.addNewLineIfMissing() }.onEachIndexed { i, it ->
          totalSamples++
          if (it == fixedStr) {
            matchFound = true
            val origRank = unrankedResults.indexOf(it)
            println("Found human repair ((rank: $i, orig: $origRank) ${clock.elapsedNow()}): $humanRepairANSI")
            elapsed = clock.elapsedNow().inWholeMilliseconds
          }
        }
    }

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
          "Δ=levDist human repair not found")
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
    "Top-1/rec/pos/total: $top1 / $recall / ${total-error} / $total, " +
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

val trainingSet: Set<Pair<String, String>> by lazy {
  val filename = "datasets/python/stack_overflow/so_ts_markov.txt"
  val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()

  contents.split("\n\n").map { it.lines().take(2).map { it.uncharify().addNewLineIfMissing() }.let { it[0] to it[1] } }.toSet()
}

// Returns a quintuple of lexical (broke, fixed) and original code (broke, fixed) pairs
val sizeAndDistBalancedRepairsUnminimized: Sequence<Π4A<Σᐩ>> by lazy {
  println("Training set size: ${trainingSet.size}")
//  val path = "/src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64_tst.txt"
//  val file = File(File("").absolutePath + path).readText()
  val filename = "datasets/python/stack_overflow/so_err_rep.txt"
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
    .filter { (broke, fixed) -> (broke to fixed) !in trainingSet }
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

      val sampler: Stream<String> =
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
