package edu.mcgill.cstk.experiments.repair

import ConcurrentRankedProbabilisticSet
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import com.google.common.util.concurrent.AtomicLongMap
import edu.mcgill.cstk.utils.*
import java.io.File
import kotlin.math.*
import kotlin.random.Random
import kotlin.streams.asStream
import kotlin.time.*

/*
./gradlew contextualRepair
 */
fun main() {
  contextualRepair()
//  contextualRepairV0()
}

//Found length-3 fix in 50.966458ms after 146 total and 1 valid samples (2 samples/ms)
//Average time to find human fix: ~347ms (44 trials, 4 expired after 10000ms)
//Average samples before matched: ~29838
//Average repair throughput / ms: ~48
//Average valid repairs detected: ~50


data class Seq2ParseRepair(val matched: Boolean, val time: Int)

fun readStackOverflow() =
  preprocessStackOverflow().map { (broke, humFix, minFix) ->
    Triple(
      broke.lexToStrTypesAsPython().joinToString(" "),
      humFix.lexToStrTypesAsPython().joinToString(" "),
      minFix.lexToStrTypesAsPython().joinToString(" "),
    )
  }

fun readStackOverflowAndGetSeq2ParseRepair() =
  preprocessStackOverflow().map { (broke, humFix, minFix) ->
//    val s2pf = measureTimedValue { seq2parseFix(broke, minFix) }
    val s2pf = measureTimedValue { "" }
    val seq2ParseFix = s2pf.value.lexToIntTypesAsPython()

    val plainTksInt = minFix.lexToIntTypesAsPython()
    val seq2ParseMatched = seq2ParseFix == plainTksInt
    val s2prInfo = Seq2ParseRepair(seq2ParseMatched, s2pf.duration.inWholeMilliseconds.toInt())

    Triple(
      broke.lexToStrTypesAsPython().joinToString(" "),
//      humFix.lexToStrTypesAsPython().joinToString(" "),
      minFix.lexToStrTypesAsPython().joinToString(" "),
      s2prInfo
    )
  }

fun contextualRepair() {
  var avgHumFixMs = 0
  var succTrials = 0
  var totSmplSize = 0
  var valSmplSize = 0
  var expirTrials = 0
  var avrgThruput = 0
  var avgFVFindMs = 0
  var avgSeq2PAcc = 0
  val startTimeMs = System.currentTimeMillis()
  val timingFile = File("repair_timings_$startTimeMs.csv")
    .also { it.writeText("Snippet length, Patch size, Time to find human repair (ms), First valid repair, Total hypotheses checked, Distinct valid repairs, Rank of human repair, Throughput, Saturation, Seq2Parse matched, Seq2Parse time\n") }
  val timeoutFile = File("repair_timeouts_$startTimeMs.csv")
    .also { it.writeText("Snippet length, Patch size, Bonus Actions, Bonus Total, Possible, Distinct valid repairs, Total hypotheses checked, Relevant edit actions, Cumultative Rank, Saturation, Seq2Parse matched, Seq2Parse time\n") }

  fun <T> List<T>.dropBOSEOS() = drop(1).dropLast(1)

//  readGoodBIFIAndCorrupt().forEach { (broke, minFix) ->
  readStackOverflowAndGetSeq2ParseRepair()
    .forEach { (broke, minFix, s2pRepairInfo) ->
    val brokeTks = listOf("BOS") + broke.tokenizeByWhitespace() + "EOS"
    val minFixTks = listOf("BOS") + minFix.tokenizeByWhitespace() + "EOS"
//    val brokeTksInt = listOf(Int.MIN_VALUE) + broke.lexToIntTypesAsPython() + Int.MAX_VALUE
    val minFixTksItt = listOf(Int.MIN_VALUE) + minFix.lexToIntTypesAsPython() + Int.MAX_VALUE
    val brokeTksInt = brokeTks.map { it.toPythonIntType() }
    val minFixTksInt = minFixTks.map { it.toPythonIntType() }
    val plainTksInt = minFixTksInt.dropBOSEOS()

    val patchSize = extractPatch(brokeTks, minFixTks).changedIndices().size

    if (!plainTksInt.isValidPython()) {
      println()
      println("Invalid Python: ${plainTksInt.joinToString(" ")}")
      println("Invalid Python: ${minFixTksItt.dropBOSEOS().joinToString(" ")}")
      println("Invalid Python: ${minFixTks.joinToString(" ")}")
      println()
      return@forEach
    }

    println("Repairing: ${brokeTks.joinToString(" ")}")

    val startTime = TimeSource.Monotonic.markNow()

    var levenshteinBlanket = brokeTksInt
    var blanketSeq: PTree? = null
    val initREAs: List<CEAProb> = contextCSV.relevantEditActions(brokeTksInt)
      .let { val last = it.last(); it + CEAProb(last.cea, last.idx, it.sumOf { it.frequency }) }
    // Bonuses for previously sampled edits that produced a valid repair
    val bonusProbs = AtomicLongMap.create<ContextEdit>()
    val uniqRepairs = AtomicLongMap.create<List<Int>>()
    val allEdits = AtomicLongMap.create<Int>()
    val goodECs = ConcurrentRankedProbabilisticSet<Context>()

//    println("Total relevant edit actions: ${initREAs.size}\n${initREAs.take(5).joinToString("\n")}\n...")
    val samplerTimeout = s2pRepairInfo.time.coerceAtLeast(30_000)
    var (total, uniqueValid) = 0 to 0
    var firstValidFoundAfter = 0L
    var saturation = (initREAs.size - 1).toDouble()
      .let { p -> (1..3).sumOf { p.pow(it) } }

    // Average time to find human fix: ~665ms (870 trials, 121 expired after 10000ms)
    // Average time to find valid fix: ~329ms
    // Average samples before matched: ~67017
    // Average repair throughput / ms: ~59
    // Average # unique valid repairs: ~37
    generateSequence { brokeTksInt }
      .asStream().parallel() // Measure latency (and precision!) with and without parallelism
      .map {
//        if (blanketSeq != null && Random.nextBoolean() && brokeTks.size > 20 && levenshteinBlanket.count { it == -1 } in 1.. 5)
//          try { blanketSeq!!.sample().removeEpsilon()
//            .also { println(it) }
//            .let { "BOS $it EOS".tokenizeByWhitespace().map { it.toPythonIntType() } } to null
//          } catch (e: Exception) { e.printStackTrace(); listOf<Int>() to listOf() }
//        else
          try { it.sampleEditTrajectory(contextCSV, initREAs, goodECs, if (firstValidFoundAfter != 0L) bonusProbs else null) }
        catch (e: Exception) { println(brokeTks); e.printStackTrace(); listOf<Int>() to listOf() }
      }
      .filter { (_, edits) -> !allEdits.containsKey(edits.hashCode()) }
      .takeWhile { (finalSeq, edits) ->
        total++; allEdits.incrementAndGet(edits.hashCode())

        if (finalSeq.dropBOSEOS().isValidPython()) {
//          println("Valid fix: ${prettyDiffNoFrills(brokeTks.joinToString(" "),
//                finalSeq.joinToString(" ") { it.toPyRuleName() })}")
          if (uniqueValid == 0 && firstValidFoundAfter == 0L)
            firstValidFoundAfter = startTime.elapsedNow().inWholeMilliseconds

//          minimizeFixInt(brokeTksInt, finalSeq) { isValidPython() }.forEach { minfix ->
//            if (uniqRepairs.incrementAndGet(minfix) == 1L) {
//              val nextLevBlanket =
//                updateLevenshteinBlanket(levenshteinBlanket, minfix)
//              if (nextLevBlanket != levenshteinBlanket) {
//                levenshteinBlanket = nextLevBlanket
//                val strLevBlanket = levenshteinBlanket.drop(1)
//                  .dropLast(1).toStrLevBlanket { it.toPyRuleNameUnquoted() }
//                println("${edits == null}: ${strLevBlanket.joinToString(" ")}")
//                blanketSeq = seq2ParseCFGNNTs.startPTree(strLevBlanket)
//              }
//            }
//          }

          // Adaptive sampler: increases probability of resampling edits
          // that result in valid repairs
          if (uniqRepairs.incrementAndGet(finalSeq) == 1L) {
            edits?.forEach {
              bonusProbs.incrementAndGet(it.cea)
              goodECs.add(it.cea.context, ln(1.0 / it.frequency.toDouble()))
            }

            uniqueValid++
          }
        }

        finalSeq != minFixTksInt
          && startTime.elapsedNow().inWholeMilliseconds < samplerTimeout
      }.forEach { }

    val rankedRepairs = uniqRepairs.asMap().keys
      .map { ints -> ints.map { it.toPyRuleName() }.let { ints to P_BIFI.score(it) } }
      .sortedBy { it.second }
//    val rankedRepairs = uniqRepairs.asMap().entries.map { it.key to it.value }

//    println(rankedRepairs.map { it.second }.take(10).joinToString("\n"))

    val repairRank = rankedRepairs.indexOfFirst { it.first == minFixTksInt }
    val repairCount = total to uniqueValid + 1
    val elapsedTime = startTime.elapsedNow().inWholeMilliseconds.toInt()
    val throughput = total / (elapsedTime + 1)
    val s2pMatchId = if (s2pRepairInfo.matched) 1 else 0
    avgSeq2PAcc += s2pMatchId
    saturation = total / saturation

    if (elapsedTime < samplerTimeout) {
      avgHumFixMs += elapsedTime
      totSmplSize += repairCount.first
      valSmplSize += repairCount.second
      avrgThruput += throughput
      avgFVFindMs += firstValidFoundAfter.toInt()

      succTrials++

      // Snippet length, Patch size, Time to find human repair (ms), First valid repair, Total repairs sampled, Distinct valid repairs, Rank of human repair, Throughput, Saturation, Seq2Parse matched, Seq2Parse time
      val timingInfo = listOf(brokeTksInt.size, patchSize,
        elapsedTime, firstValidFoundAfter.toInt(), repairCount.first,
        repairCount.second, repairRank, throughput, saturation,
        s2pMatchId, s2pRepairInfo.time)
      timingFile.appendText(timingInfo.joinToString(", ") + "\n")

      println("""Found length-${patchSize} fix in ${elapsedTime}ms after ${repairCount.first} total and ${repairCount.second} valid samples 
(${throughput} samples/ms, |REAs| = ${initREAs.size}, saturation: $saturation, bonus probs: (${bonusProbs.size()}, ${bonusProbs.sum()}), first valid sample: ${firstValidFoundAfter}ms)
(Rank of human fix: $repairRank/${repairCount.second}, Seq2Parse matched: ${s2pRepairInfo.matched})

        Average time to find human fix: ~${avgHumFixMs / succTrials}ms ($succTrials successful trials, $expirTrials expired after ${samplerTimeout}ms)
        Average time to find valid fix: ~${avgFVFindMs / succTrials}ms
        Average samples before matched: ~${totSmplSize / succTrials}
        Average repair throughput / ms: ~${avrgThruput / succTrials}
        Average # unique valid repairs: ~${valSmplSize / succTrials}
        Average Seq2Parse Precision@1:  ~${avgSeq2PAcc.toDouble() / (succTrials + expirTrials)}
        Average Tidyparse Precision@*:  ~${succTrials.toDouble() / (succTrials + expirTrials)}
      """.trimIndent())
    } else {
      val (bonusEdits, bonusTotal) = bonusProbs.size() to bonusProbs.sum()
      val trueContextEdits = extractContextEdits(brokeTks, minFixTks)
      val bonusREAs = initREAs.dropLast(1)
        .map { CEAProb(it.cea, it.idx, (it.frequency + it.frequency * bonusProbs[it.cea]).toInt()) }

      val ceaNorm = bonusREAs.sumOf { it.frequency }
      val rank = bonusREAs.sortedBy { it.frequency }
      var cumRank = 0
      println("True context edits:\n${trueContextEdits.joinToString("\n") { cea ->
        var rankIdx = -1
        val ceaProb = rank.firstOrNull { rankIdx++; it.cea == cea  }
        cumRank += rankIdx
        "CEA: $cea, CEARANK: ${if(ceaProb == null) -1 else rankIdx}, FREQ: ${ceaProb?.frequency?: 0}/$ceaNorm"
      }}")

//      println("Unknown context edits: ${trueContextEdits.filter { it !in contextCSV.allProbs }.joinToString("\n")}")
      val possibleToSample = trueContextEdits.all { it in contextCSV.allProbs }.let { if (it) 1 else 0 }
      println("""
Sampling timeout expired after $repairCount (total, valid) samples, |REAs|: ${initREAs.size}, saturation: $saturation
(${throughput} samples/ms, bonus probs: ($bonusEdits, $bonusTotal), first valid sample: ${firstValidFoundAfter}ms, possible: $possibleToSample), ground truth repair was $patchSize edits:
      
${prettyDiffNoFrillsTrimAndAlignWithOriginal(brokeTks.joinToString(" "), minFixTks.joinToString(" "))}
${prettyDiffNoFrillsTrimAndAlignWithOriginal(brokeTksInt.joinToString(" "), minFixTksInt.joinToString(" "))}
      """).also { expirTrials += 1 }
//    "Snippet length, Patch size, Bonus Actions, Bonus Total, Possible,
//    Distinct valid repairs, Total hypotheses checked, Relevant edit actions, Cumulative Rank, Saturation, Seq2Parse matched, Seq2Parse time
      val timeoutInfo = listOf(brokeTksInt.size,
        patchSize, bonusEdits, bonusTotal, possibleToSample, repairCount.second,
        repairCount.first, initREAs.size, cumRank, saturation, s2pMatchId, s2pRepairInfo.time)
      timeoutFile.appendText(timeoutInfo.joinToString(", ") + "\n")
    }

    println()
  }
}

val contextCSV by lazy { File("context_edits.csv").readTrigramStats() }

enum class EditType { INS, DEL, SUB }
data class ContextEdit(val type: EditType, val context: Context, val newMid: Int) {
  override fun toString(): String = context.run {
    "$type, (( " + when (type) {
      EditType.INS -> "${left.toPyRuleName()} [${newMid.toPyRuleName()}] ${right.toPyRuleName()}"
      EditType.DEL -> "${left.toPyRuleName()} ~${mid.toPyRuleName()}~ ${right.toPyRuleName()}"
      EditType.SUB -> "${left.toPyRuleName()} [${mid.toPyRuleName()} -> ${newMid.toPyRuleName()}] ${right.toPyRuleName()}"
    } + " // " + when (type) {
      EditType.INS -> "$left [${newMid}] $right"
      EditType.DEL -> "$left ~${mid}~ $right"
      EditType.SUB -> "$left [${mid} -> ${newMid}] $right"
    } + " ))"
  }
}
data class CEAProb(val cea: ContextEdit, val idx: Int, val frequency: Int) {
  override fun equals(other: Any?): Boolean = when (other) {
    is CEAProb -> cea == other.cea && idx == other.idx
    else -> false
  }
  override fun hashCode(): Int = 31 * cea.hashCode() + idx
  override fun toString(): String = "[[ $cea, $idx, $frequency ]]"
}
data class Context(val left: Int, val mid: Int, val right: Int) {
  constructor(left: String, mid: String, right: String) :
    this(left.toPythonIntType(), mid.toPythonIntType(), right.toPythonIntType())

  override fun equals(other: Any?) = when (other) {
    is Context -> left == other.left && mid == other.mid && right == other.right
    else -> false
  }

  override fun hashCode(): Int {
    var result = left.hashCode()
    result = 31 * result + mid.hashCode()
    result = 31 * result + right.hashCode()
    return result
  }
}

data class CEADist(val allProbs: Map<ContextEdit, Int>) {
  val P_delSub = allProbs.filter { it.key.type != EditType.INS }
  val P_insert = allProbs.filter { it.key.type == EditType.INS }
  val P_delSubOnCtx = P_delSub.keys.groupBy { it.context }
  val P_insertOnCtx = P_insert.keys.groupBy { it.context }
}

// Divesity: lower is more diverse, higher is less diverse, 1.0 is natural frequencies
fun File.readTrigramStats(diversity: Double = 1.0): CEADist =
  readLines().drop(1).map { it.split(", ") }.associate {
    (ContextEdit(
      type = EditType.valueOf(it[0].trim()),
      context = Context(it[1], it[2], it[3]),
      newMid = it[4].toPythonIntType()
    )
//      .also { t -> println(it.joinToString(", ") + " :: $t") }
      ) to it[5].trim().toDouble().pow(diversity).toInt().coerceAtLeast(1)
  }.let { CEADist(it) }

fun List<Int>.sampleEditTrajectory(
  ceaDist: CEADist,
  initREAs: List<CEAProb>, // Last element is the normalization constant
  goodECs: ConcurrentRankedProbabilisticSet<Context>? = null,
  // Bonuses for previously sampled edits that produced a valid repair
  bonusProbs: AtomicLongMap<ContextEdit>? = null,
  lengthCDF: List<Double> = listOf(0.5, 0.8, 1.0)
): Pair<List<Int>, List<CEAProb>> {
  // First sample the length of the edit trajectory from the length distribution
  val rand = Math.random()
  val length = lengthCDF.indexOfFirst { rand < it } + 1

  if (initREAs.size < 2) return this to listOf()
  val usedCEAProbs = mutableListOf<CEAProb>()
  // Now sample an edit trajectory of that length from the edit distribution
  val normConst = initREAs.last().frequency
  val initREAs = initREAs.dropLast(1)
  var listPrime = (
     if (goodECs != null && goodECs.size != 0 && Random.nextDouble() < 0.5)
       initREAs.filter { it.cea.context in goodECs }
         .normalizeAndSample(bonusProbs = bonusProbs)
     else initREAs
//       .random()
       .normalizeAndSample(normConst, bonusProbs)
       .also { usedCEAProbs.add(it) }
    ).let { applyEditAction(it.cea, it.idx + 1) }

  for (i in 1..length) {
    val relevantEditActions =
      ceaDist.relevantEditActions(listPrime).let {
        if (goodECs != null && goodECs.size != 0 && Random.nextDouble() < 0.5)
          it.filter { it.cea.context in goodECs }
            .ifEmpty { ceaDist.relevantEditActions(listPrime) }
        else it
      }

    if (relevantEditActions.isEmpty()) break
    val sampledEdit = relevantEditActions
//      .random()
      .normalizeAndSample(bonusProbs = bonusProbs)
      .also { usedCEAProbs.add(it) }
    listPrime = listPrime.applyEditAction(sampledEdit.cea, sampledEdit.idx + 1)
  }
  return listPrime to usedCEAProbs.sortedBy { it.idx }
}

fun List<CEAProb>.normalizeAndSample(normConst: Int = -1, bonusProbs: AtomicLongMap<ContextEdit>?): CEAProb =
  if (bonusProbs == null) {
    val sample: Int = Random.nextInt(if (normConst == -1) sumOf { it.frequency } else normConst)
    var sum = 0
    var last: CEAProb? = null
    for (i in this) {
      sum += i.frequency
      last = i
      if (sum > sample) break
    }
    last!!
  } else {
    val cdf: List<Int> = map { it.frequency + it.frequency * bonusProbs[it.cea].toInt().coerceAtMost(5) }
      .let { freqs ->
        val cdf = mutableListOf<Int>()
        var sum = 0
        for (i in freqs.indices) {
          sum += freqs[i]
          cdf.add(sum)
        }
        cdf
      }
    val sample: Int = Random.nextInt(cdf.last())
    this[cdf.binarySearch(sample).let { if (it < 0) -it - 1 else it }.coerceIn(indices)]
  }

//Found length-3 fix in 1.289318375s after 103328 total and 1 valid samples (80 samples/ms)
//Average time to find human fix: ~629ms (74 trials, 12 expired after 10000ms)
//Average samples before matched: ~62480
//Average repair throughput / ms: ~58
//Average valid repairs detected: ~52

fun CEADist.relevantEditActions(snippet: List<Int>): List<CEAProb> {
  val relevantEditActions = mutableListOf<CEAProb>()
  for (i in 0 until snippet.size - 2) {
    val ctx = Context(snippet[i], snippet[i + 1], snippet[i + 2])
    P_insertOnCtx[Context(ctx.left, -1, ctx.mid)]?.forEach {
      relevantEditActions.add(CEAProb(it, i, P_insert[it]!!))
    }
    if (i == snippet.size - 3)
      P_insertOnCtx[Context(ctx.mid, -1, ctx.right)]?.forEach {
        relevantEditActions.add(CEAProb(it, i, P_insert[it]!!))
      }
    P_delSubOnCtx[ctx]?.forEach {
      relevantEditActions.add(CEAProb(it, i, P_delSub[it]!!))
    }
  }
  return relevantEditActions
}

fun List<Int>.applyEditAction(cea: ContextEdit, idx: Int): List<Int> =
  when (cea.type) {                                                       // 6409ms, 20%
    EditType.INS -> subList(0, idx) + cea.newMid + subList(idx + 1, size) // 17937ms, 55%
    EditType.DEL -> subList(0, idx) + subList(idx + 1, size)              // 2607ms, 8%
    EditType.SUB -> subList(0, idx) + cea.newMid + subList(idx + 1, size) // 5552ms, 17%
  }//.also { println("Start:$this\n${cea.type}/${cea.context}/${cea.newMid}/${idx}\nAfter:$it") }

fun extractContextEdits(broke: List<String>, minfix: List<String>): List<ContextEdit> {
  val patch = extractPatch(broke, minfix)
  return patch.run {
    changedIndices().map { i ->
      val (old, new) = get(i).old to get(i).new

      if (old == "")
        ContextEdit(EditType.INS, Context(sln(i).toPythonIntType(), -1, sro(i).toPythonIntType()), new.toPythonIntType())
      else if (new == "")
//        "DEL, ${sln(i)}, $old, ${sro(i)}, "
        ContextEdit(EditType.DEL, Context(sln(i).toPythonIntType(), old.toPythonIntType(), sro(i).toPythonIntType()), -1)
      else
//        "SUB, ${sln(i)}, $old, ${sro(i)}, $new"
        ContextEdit(EditType.SUB, Context(sln(i).toPythonIntType(), old.toPythonIntType(), sro(i).toPythonIntType()), new.toPythonIntType())
    }
  }
}

fun contextualRepairV0() {
  var avgHumFixMs = 0
  var totalTrials = 0
  var totSmplSize = 0
  var valSmplSize = 0
  var expiredSize = 0
  var avrgThruput = 0
  var avgFVFindMs = 0
  val timingFile = File("repair_timings_${System.currentTimeMillis()}.csv")
    .also { it.writeText("Snippet length, Patch size, Time to find human repair (ms), First valid repair, Total repairs sampled, Distinct valid repairs, Throughput\n") }

//  readGoodBIFIAndCorrupt().forEach { (broke, minFix) ->
  readStackOverflow().forEach { (broke, _, minFix) ->
    val brokeTks = listOf("BOS") + broke.tokenizeByWhitespace() + "EOS"
    val minFixTks = listOf("BOS") + minFix.tokenizeByWhitespace() + "EOS"
//    val brokeTksInt = listOf(Int.MIN_VALUE) + broke.lexToIntTypesAsPython() + Int.MAX_VALUE
    val minFixTksItt = listOf(Int.MIN_VALUE) + minFix.lexToIntTypesAsPython() + Int.MAX_VALUE
    val brokeTksInt = brokeTks.map { it.toPythonIntType() }
    val minFixTksInt = minFixTks.map { it.toPythonIntType() }

    val patchSize = extractPatch(brokeTks, minFixTks).changedIndices().size

    val clr = minFixTksInt.drop(1).dropLast(1)
    if (!clr.isValidPython()) {
      println()
      println("Invalid Python: ${clr.joinToString(" ")}")
      println("Invalid Python: ${minFixTksItt.drop(1).dropLast(1).joinToString(" ")}")
      println("Invalid Python: ${minFixTks.joinToString(" ")}")
      println()
      return@forEach
    }

    println("Repairing: ${brokeTks.joinToString(" ")}")

    val startTime = TimeSource.Monotonic.markNow()

    val initREAs: List<CEAProb> = contextCSV.relevantEditActions(brokeTksInt)
    // Bonuses for previously sampled edits that produced a valid repair
    val bonusProbs = AtomicLongMap.create<ContextEdit>()
    val uniqRepairs = AtomicLongMap.create<List<Int>>()

//    println("Total relevant edit actions: ${initREAs.size}\n${initREAs.take(5).joinToString("\n")}\n...")
    val samplerTimeout = 10000L
    var (total, uniqueValid) = 0 to 0
    var firstValidFoundAfter = 0L

    // Average time to find human fix: ~665ms (870 trials, 121 expired after 10000ms)
    // Average time to find valid fix: ~329ms
    // Average samples before matched: ~67017
    // Average repair throughput / ms: ~59
    // Average # unique valid repairs: ~37
    generateSequence { brokeTksInt }
      .asStream().parallel() // Measure latency with and without parallelism
      .map {
        try { it.sampleEditTrajectoryV0(contextCSV, initREAs,
          if (firstValidFoundAfter != 0L) bonusProbs else null) }
        catch (e: Exception) {
          println(brokeTks); e.printStackTrace(); listOf<Int>() to listOf()
        }
      }.takeWhile { (finalSeq, _) ->
        finalSeq != minFixTksInt
          && startTime.elapsedNow().inWholeMilliseconds < samplerTimeout
      }.forEach { (finalSeq, edits) ->
        total++

        if (finalSeq.drop(1).dropLast(1).isValidPython()) {
//          println("Valid fix: ${prettyDiffNoFrills(brokeTks.joinToString(" "),
//                finalSeq.joinToString(" ") { it.toPyRuleName() })}")
          if (uniqueValid == 0 && firstValidFoundAfter == 0L)
            firstValidFoundAfter = startTime.elapsedNow().inWholeMilliseconds

          // Timings with adaptive sampling enabled:

          // Adaptive sampler: increases probability of resampling edits
          // that result in valid repairs
          if (uniqRepairs.incrementAndGet(finalSeq) == 1L) {
            edits.forEach { bonusProbs.incrementAndGet(it.cea) }

            uniqueValid++
          }
        }
      }

    val repairCount = total to uniqueValid.coerceAtLeast(1)

    val elapsedTime = startTime.elapsedNow().inWholeMilliseconds.toInt()
    val throughput = repairCount.first / (elapsedTime + 1)
    totalTrials++

    if (elapsedTime < samplerTimeout) {
      avgHumFixMs += elapsedTime
      totSmplSize += repairCount.first
      valSmplSize += repairCount.second
      avrgThruput += throughput
      avgFVFindMs += firstValidFoundAfter.toInt()

      // "Snippet length, Patch size, Time to find human repair (ms), First valid repair, Total repairs sampled, Distinct valid repairs, Throughput"
      val timing = listOf(brokeTksInt.size, patchSize, elapsedTime, firstValidFoundAfter.toInt(), repairCount.first, repairCount.second, throughput)
      timingFile.appendText(timing.joinToString(", ") + "\n")

      println("""Found length-${patchSize} fix in ${elapsedTime}ms after ${repairCount.first} total and ${repairCount.second} valid samples (${throughput} samples/ms, first valid sample: ${firstValidFoundAfter}ms)
        
        Average time to find human fix: ~${avgHumFixMs / totalTrials}ms (${totalTrials - expiredSize} successful trials, $expiredSize expired after ${samplerTimeout}ms)
        Average time to find valid fix: ~${avgFVFindMs / totalTrials}ms
        Average samples before matched: ~${totSmplSize / totalTrials}
        Average repair throughput / ms: ~${avrgThruput / totalTrials}
        Average # unique valid repairs: ~${valSmplSize / totalTrials}
      """.trimIndent())
    } else println("""
      Sampling timeout expired after $repairCount (total, valid) samples (${throughput} samples/ms, first valid sample: ${firstValidFoundAfter}ms), ground truth repair was $patchSize edits:
      ${brokeTks.joinToString(" ")}
      ${prettyDiffNoFrills(brokeTks.joinToString(" "), minFixTks.joinToString(" "))}
      ${brokeTksInt.joinToString(" ")}
      ${prettyDiffNoFrills(brokeTksInt.joinToString(" "), minFixTksInt.joinToString(" "))}
    """.trimIndent()).also { expiredSize += 1 }

    println()
  }
}

fun List<Int>.sampleEditTrajectoryV0(
  ceaDist: CEADist,
  initREAs: List<CEAProb>,
  // Bonuses for previously sampled edits that produced a valid repair
  bonusProbs: AtomicLongMap<ContextEdit>? = null,
  lengthCDF: List<Double> = listOf(0.5, 0.8, 1.0)
): Pair<List<Int>, List<CEAProb>> {
  // First sample the length of the edit trajectory from the length distribution
  val rand = Math.random()
  val length = lengthCDF.indexOfFirst { rand < it } + 1

  if (initREAs.isEmpty()) return this to listOf()
  val ceaProbs = mutableListOf<CEAProb>()
  // Now sample an edit trajectory of that length from the edit distribution
  var listPrime =
    initREAs.normalizeAndSampleV0(bonusProbs)
      .also { ceaProbs.add(it) }
      .let { applyEditAction(it.cea, it.idx + 1) }

  for (i in 1..length) {
    val relevantEditActions = ceaDist.relevantEditActions(listPrime)
    if (relevantEditActions.isEmpty()) {
      println("$i-th iteration, no relevant edit actions for: ${listPrime.joinToString(" "){ it.toPyRuleName() }}")
      return listPrime to ceaProbs
    }
    val sampledEdit = relevantEditActions.normalizeAndSampleV0(bonusProbs)
      .also { ceaProbs.add(it) }
    listPrime = listPrime.applyEditAction(sampledEdit.cea, sampledEdit.idx + 1)
  }
  return listPrime to ceaProbs
}

// Faster than the above
fun List<CEAProb>.normalizeAndSampleV0(bonusProbs: AtomicLongMap<ContextEdit>?): CEAProb {
  val cdf: List<Int> = (if (bonusProbs == null) map { it.frequency }
  else map { it.frequency + bonusProbs[it.cea].toInt() * 100 })
    .let { freqs ->
      val cdf = mutableListOf<Int>()
      var sum = 0
      for (i in freqs.indices) {
        sum += freqs[i]
        cdf.add(sum)
      }
      cdf
    }
  val sample: Int = Random.nextInt(cdf.last())
  return this[cdf.binarySearch(sample).let { if (it < 0) -it - 1 else it }.coerceIn(indices)]
}