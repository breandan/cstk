package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import edu.mcgill.cstk.utils.*
import edu.mcgill.cstk.utils.Edit
import java.io.File
import java.util.concurrent.ForkJoinPool
import kotlin.math.absoluteValue
import kotlin.random.Random
import kotlin.streams.asStream
import kotlin.time.TimeSource

/*
./gradlew collectSummaryStats
 */
fun main() {
//  stackOverflowSnips().computeLengthDistributionStats()
//  stackOverflowSnips().computeRawTokenFrequencies()
//  seq2ParseSnips().computeBigramFrequencies()
//  computeErrorSizeFreq()
//  computePatchStats()
//  computePatchTrigramStats()
//  readBIFI()
//  computeEditLocationFrequency()
//  computeRelativeIntraEditDistance()
//  totalCharacterEditDistance()
  mostCommonSubstitutions()
//  contextualRepair()
}

fun mostCommonSubstitutions() =
  File("context_edits.csv").readTrigramStats()
    .allProbs.map { it }.filter { it.key.type == EditType.SUB }
    .filter { it.key.context.mid.toPyRuleName().contains('\'') }
    .filter { it.key.newMid.toPyRuleName().contains('\'') }
    .map { Triple(it.value, it.key.context.mid, it.key.newMid) }
    .groupBy { it.second to it.third }.mapValues { it.value.sumOf { it.first } }
    .entries.sortedByDescending { it.value }
    .joinToString("\n") { (pair, freq) ->
      "$freq, ${pair.first.toPyRuleName()}, ${pair.second.toPyRuleName()}" }
    .also { println(("freq, before, after\n$it").reformatCSVIntoPrettyColumns()) }

fun readBIFI() =
  readBIFIContents().take(100_000)
    .map { "\n$it\n".lexToStrTypesAsPython().let { listOf("BOS") + it + "EOS" } }
    .forEach { println(it.joinToString(" ")) }

/*
Percentage of snippets with lexical token length <= n

| n  | StackOverflow | Seq2Parse |
|:--:|:-------------:|:---------:|
| 10 |      9%       |    29%    |
| 20 |      16%      |    47%    |
| 30 |      25%      |    62%    |
| 40 |      34%      |    73%    |
| 50 |      41%      |    79%    |
 */

// Returns the frequency of tokenized snippet lengths for buckets of size 10
fun Sequence<List<String>>.computeLengthDistributionStats(
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
) =
    map { it.size }.take(10000).groupBy { it / 10 }.mapValues { it.value.size }
    .toList().sortedBy { it.first }.map { it.first * 10 to it.second }
    .runningFold(0 to 0) { (_, prevCount), (n, count) -> n to (prevCount + count) }
    .let { it.map { (n, count) -> n to count.toDouble() / it.last().second } }
    .joinToString("\n") { "<=${it.first}, ${it.second}" }.also { println(it) }

fun Sequence<List<String>>.computeRawTokenFrequencies() =
  take(10000).flatten().groupingBy { it }.eachCount()
    // Now normalize by total number of tokens
    .let { val total = it.values.sum(); it.mapValues { (_, count) -> count.toDouble() / total } }
    // Use precision of 4 decimal places
    .mapValues { (_, count) -> "%.4f".format(count) }
    .toList().sortedByDescending { it.second }
    .joinToString("\n") { "${it.first}, ${it.second}" }.also { println("Token, Frequency\n$it") }

/*
Top Bigrams:

Bigram, Frequency
"_NAME_ (", 0.0551
") _NEWLINE_", 0.0424
"( _NAME_", 0.0330
"_NEWLINE_ _DEDENT_", 0.0321
"_NEWLINE_ _INDENT_", 0.0258
"_NEWLINE_ _NAME_", 0.0255
": _NEWLINE_", 0.0238
"_NAME_ )", 0.0237
"_NAME_ =", 0.0232
"_NEWLINE_ _NEWLINE_", 0.0224
"Colon Newline", 0.0197
"_INDENT_ _NAME_", 0.0190
"Newline Indent", 0.0167
"( _STRING_", 0.0149
"= _NAME_", 0.0146
"_NEWLINE_ _ENDMARKER_", 0.0146
"_STRING_ )", 0.0136
"_DEDENT_ _DEDENT_", 0.0128
"_NUMBER_ ,", 0.0126
", _NUMBER_", 0.0116
"_NAME_ _NEWLINE_", 0.0113
"_DEDENT_ _NAME_", 0.0112
", _NAME_", 0.0102
"_NUMBER_ _NEWLINE_", 0.0101
"_NUMBER_ )", 0.0097
"_ENDMARKER_ Stmts_Or_Newlines", 0.0095
". _NAME_", 0.0092
"Indent Stmts_Or_Newlines", 0.0090
"_DEDENT_ _ENDMARKER_", 0.0085
"_NAME_ .", 0.0084
"Def_Keyword Simple_Name", 0.0081
 */

fun Sequence<List<String>>.computeBigramFrequencies() =
    take(10000).flatten().zipWithNext().groupingBy { it }.eachCount()
    // Now normalize by total number of tokens
    .let { val total = it.values.sum(); it.mapValues { (_, count) -> count.toDouble() / total } }
    // Use precision of 4 decimal places
    .mapValues { (_, count) -> "%.4f".format(count) }
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "\"${it.first.first} ${it.first.second}\", ${it.second}" }.also { println("Bigram, Frequency\n$it") }

fun stackOverflowSnips() =
  readContents("parse_errors.json").map { it.lexToStrTypesAsPython() }

fun seq2ParseSnips() =
  brokenPythonSnippets.map { it.substringBefore("<||>").tokenizeByWhitespace() }

/*
Number of edits, Frequency
1, 0.512
2, 0.834
3, 1.0
*/
fun computeErrorSizeFreq() =
  preprocessStackOverflow().map { (broke, humfix, minfix) ->
    val brokeLex = broke.lexToStrTypesAsPython()
    val minfixLex = minfix.lexToStrTypesAsPython()
    val minpatch = extractPatch(brokeLex, minfixLex)
//    println(prettyDiffs(listOf(brokeLex.joinToString(" "), minfixLex.joinToString(" ")), listOf("broken", "minimized fix")))
    minpatch.changedIndices().size
  }.take(1000).groupBy { it }.mapValues { it.value.size }
    .toList().sortedBy { it.first }.map { it.first to it.second }
    .runningFold(0 to 0) { (_, prevCount), (n, count) -> n to (prevCount + count) }
    .let { it.map { (n, count) -> n to count.toDouble() / it.last().second } }
    .joinToString("\n") { "${it.first}, ${it.second}" }
      .also { println("Number of edits, Frequency\n$it") }

// Approximate location of edits normalized by snippet length
//10%, 11.6539%
//20%, 5.7252%
//30%, 6.2087%
//40%, 5.9542%
//50%, 5.5980%
//60%, 7.9389%
//70%, 7.0738%
//80%, 6.9466%
//90%, 12.4173%
//100%, 30.4835%
fun computeEditLocationFrequency() =
  preprocessStackOverflow().runningFold(List(10) { 0 }.toMutableList()) { hist, (b, h, m) ->
    val brokeLex = b.lexToStrTypesAsPython()
    val minfixLex = m.lexToStrTypesAsPython()
    val minpatch = extractPatch(brokeLex, minfixLex)
//    println(prettyDiffs(listOf(brokeLex.joinToString(" "), minfixLex.joinToString(" ")), listOf("broken", "minimized fix")))
    minpatch.changedIndices()
      .map { ((100.0 * it / minfixLex.size) / 10).toInt().coerceAtMost(9) }
      .forEach { hist[it]++ }

    hist
  }.forEachIndexed { i, rawCounts ->
    if (i % 10 == 0) {
      val sum = rawCounts.sum()
      rawCounts.forEachIndexed { i, it ->
        println("${(i + 1) * 10}%, ${"%.4f".format(100.0 * it / sum)}%")
      }; println()
    }
  }

//1, 40.66%
//2, 15.00%
//3, 5.80%
//4, 4.86%
//5, 4.26%
//6, 2.98%
//7, 2.05%
//8, 2.73%
//9, 1.62%
//10, 2.30%
//11, 1.88%
//12, 2.81%
//13, 1.11%
//14, 0.60%
//15, 1.28%
//16, 1.45%
//17, 1.02%
//18, 0.68%
//19, 0.51%
// Relative distance between edits in multi-edit patches
fun computeRelativeIntraEditDistance() =
  preprocessStackOverflow().map { (b, h, m) ->
    val brokeLex = b.lexToStrTypesAsPython()
    val minfixLex = m.lexToStrTypesAsPython()
    extractPatch(brokeLex, minfixLex)
  }.filter { 1 < it.size }.runningFold(mutableMapOf<Int, Int>()) { hist, patch ->
    patch.changedIndices().zipWithNext().forEach { (i, j) ->
      val d = (j - i).absoluteValue
      hist[d] = hist.getOrDefault(d, 0) + 1
    }
    hist
  }.forEachIndexed { i, rawCounts ->
    if (i % 10 == 0) {
      val sum = rawCounts.values.sum()
      rawCounts.toList().sortedBy { it.first }.forEach { (dist, count) ->
        println("${dist}, ${"%.2f".format(100.0 * count / sum)}%")
      }; println()
    }
  }

//1, 0.72%
//2, 53.48%
//3, 29.48%
//4, 11.35%
//5, 3.32%
//6, 0.83%
//7, 0.29%
//9, 0.20%
//10, 0.04%
fun totalCharacterEditDistance() =
  preprocessStackOverflow().map { (b, h, m) ->
    val brokeLex = b.tokenizeAsPython()
    val minfixLex = m.tokenizeAsPython()
    extractPatch(brokeLex, minfixLex).totalCharacterwiseEditDistance()
  }.runningFold(mutableMapOf<Int, Int>()) { h, p ->
    h[p] = h.getOrDefault(p, 0) + 1; h
  }.forEachIndexed { i, rawCounts ->
    if (i % 10 == 0) {
      val sum = rawCounts.values.sum()
      rawCounts.toList().sortedBy { it.first }.forEach { (dist, count) ->
        println("${dist + 1}, ${"%.2f".format(100.0 * count / sum)}%")
      }; println()
    }
  }

/*
Insertion, Frequency, Deletion     , Frequency, Substitution            , Frequency
')'      , 98       , NAME         , 219      , (UNKNOWN_CHAR -> STRING), 53
','      , 87       , UNKNOWN_CHAR , 160      , (NAME -> ',')           , 22
98       , 59       , '.'          , 50       , (':' -> ',')            , 12
99       , 58       , NUMBER       , 41       , (NEWLINE -> ':')        , 12
'}'      , 46       , '>'          , 30       , ('=' -> ':')            , 11
']'      , 36       , NEWLINE      , 28       , (STRING -> ',')         , 8
':'      , 35       , ')'          , 26       , ('.' -> ',')            , 8
'('      , 33       , '**'         , 25       , (NAME -> STRING)        , 8
NEWLINE  , 28       , ':'          , 23       , ('pass' -> NAME)        , 8
NAME     , 19       , '>>'         , 20       , ('=' -> '==')           , 7
STRING   , 17       , STRING       , 19       , ('class' -> NAME)       , 6
'{'      , 13       , 98           , 13       , ('[' -> '{')            , 6
'='      , 12       , '...'        , 12       , ('break' -> NAME)       , 6
'['      , 10       , '('          , 10       , (UNKNOWN_CHAR -> ',')   , 6
'class'  , 8        , '{'          , 10       , (',' -> ':')            , 6
'def'    , 7        , 99           , 10       , ('(' -> '[')            , 6
'import' , 6        , '['          , 9        , (']' -> '}')            , 5
'.'      , 6        , ']'          , 7        , (UNKNOWN_CHAR -> '.')   , 4
'from'   , 5        , '*'          , 6        , (NAME -> ']')           , 4
'return' , 3        , '/'          , 5        , ('...' -> ',')          , 4
'in'     , 2        , '//'         , 5        , ('in' -> NAME)          , 4
'...'    , 2        , ','          , 4        , ('<' -> '&')            , 3
'except' , 2        , '}'          , 4        , ('>' -> ';')            , 3
*/

fun computePatchStats() =
  preprocessStackOverflowInParallel(take = 100_000).map { (broke, _, minfix) ->
    val brokeLexed = broke.lexToStrTypesAsPython()
    val minfixLexed = minfix.lexToStrTypesAsPython()
    val patch = extractPatch(brokeLexed, minfixLexed)
    patch.changedIndices().map {
      if (patch[it].old == "") "INS, ${patch[it].new}"
      else if (patch[it].new == "") "DEL, ${patch[it].old}"
      else "SUB, ${patch[it].old} -> ${patch[it].new}"
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "${it.first}, ${it.second}" }
    .also { println("Type, Edit, Frequency\n$it") }

/*
Type, Edit, Frequency
INS, ')' [')'] END, 467
INS, NAME ['('] NAME, 456
INS, NAME [')'] END, 198
INS, NAME ['('] STRING, 174
DEL, NAME [NAME] NEWLINE, 154
DEL, START ['>>'] '>', 142
INS, NAME [')'] NEWLINE, 131
DEL, NAME [NAME] END, 124
DEL, START ['>'] NAME, 117
DEL, ')' [UNKNOWN_CHAR] END, 112
INS, ')' [NEWLINE] NAME, 110
INS, ']' [')'] END, 103
INS, NEWLINE [98] NAME, 92
INS, ')' [')'] NEWLINE, 92
INS, ':' [NAME] END, 89
INS, NAME ['='] NAME, 82
DEL, NAME [NAME] '.', 81
INS, ')' [':'] NEWLINE, 80
DEL, NAME [NAME] ':', 80
INS, 99 [99] END, 79
DEL, NAME [NAME] NAME, 76
INS, STRING [')'] END, 76
DEL, ')' [')'] END, 70
DEL, STRING [NAME] UNKNOWN_CHAR, 65
INS, STRING [')'] NEWLINE, 62
DEL, START [UNKNOWN_CHAR] NAME, 60
DEL, NAME [NAME] '(', 58
SUB, '(' [UNKNOWN_CHAR -> STRING] NAME, 57
DEL, STRING [NAME] ')', 56
INS, NUMBER [','] NUMBER, 55
SUB, 98 ['pass' -> NAME] NEWLINE, 55
DEL, STRING [NAME] ']', 54
SUB, ',' [UNKNOWN_CHAR -> STRING] NAME, 53
INS, NEWLINE [99] NAME, 53
INS, ']' [')'] NEWLINE, 52
INS, '}' ['}'] END, 48
DEL, NAME [UNKNOWN_CHAR] ']', 47
INS, START ['import'] NAME, 45
INS, ')' [')'] NAME, 42
INS, NEWLINE [98] 'def', 41
DEL, ']' [UNKNOWN_CHAR] END, 40
DEL, ')' ['.'] END, 39
INS, NAME [','] NAME, 39
DEL, STRING [NAME] STRING, 39
INS, ')' [']'] END, 39
DEL, NAME [UNKNOWN_CHAR] ')', 38
SUB, STRING [NAME -> ','] UNKNOWN_CHAR, 37
DEL, NAME [NAME] '=', 37
INS, ']' ['}'] END, 36
INS, ']' [']'] END, 35
INS, NAME ['.'] NAME, 34
DEL, NAME [UNKNOWN_CHAR] END, 33
SUB, '=' [UNKNOWN_CHAR -> STRING] NAME, 33
DEL, NAME [':'] NEWLINE, 33
INS, '(' [')'] END, 33
DEL, NAME ['('] NAME, 33
DEL, 99 [99] END, 32
INS, ',' [']'] END, 32
SUB, '[' [UNKNOWN_CHAR -> STRING] NAME, 32
DEL, NAME [NAME] ',', 32
INS, STRING [STRING] NEWLINE, 31
DEL, NAME [NAME] ')', 31
INS, STRING [','] NAME, 31
INS, NAME [':'] NEWLINE, 31
INS, STRING [','] STRING, 31
DEL, STRING [UNKNOWN_CHAR] ')', 31
DEL, ']' ['.'] END, 29
INS, NAME ['in'] NAME, 29
INS, ']' [NEWLINE] NAME, 28
INS, NEWLINE [98] 'if', 28
INS, START ['def'] NAME, 27
INS, START ['{'] STRING, 27
DEL, STRING [UNKNOWN_CHAR] ']', 27
INS, NUMBER [','] STRING, 27
INS, ',' ['}'] END, 27
DEL, START ['**'] NAME, 27
DEL, ')' [':'] END, 26
INS, ':' ['...'] END, 26
DEL, NAME [NAME] '[', 25
INS, STRING [']'] ')', 25
DEL, STRING [NAME] '.', 25
INS, ')' [')'] 'for', 25
INS, STRING ['}'] END, 24
INS, START ['from'] NAME, 24
DEL, ')' ['**'] END, 24
DEL, NAME ['>'] NEWLINE, 24
INS, START ['class'] NAME, 24
INS, 'for' [NAME] 'in', 24
INS, NAME [','] STRING, 23
SUB, STRING ['=' -> ':'] STRING, 23
INS, '}' [')'] END, 23
DEL, NAME ['.'] END, 22
INS, ':' ['('] END, 22
DEL, STRING [NAME] ',', 22
DEL, START [NEWLINE] 98, 22
INS, NAME ['('] '[', 22
INS, ']' [']'] ')', 21
DEL, ',' [NEWLINE] STRING, 20
INS, NEWLINE [98] 'for', 20
SUB, 98 ['break' -> NAME] NEWLINE, 20
*/

fun Patch.scan(i: Int, direction: Boolean, age: Edit.() -> Σᐩ): Σᐩ? =
  (if (direction) (i + 1 until size) else (i - 1 downTo 0))
    .firstOrNull { this[it].age() != "" }?.let { this[it].age() }

// Scan [l]eft/[r]ight for first non-empty [n]ew/[o]ld token
fun Patch.sln(i: Int): String = scan(i, false) { new }!!
fun Patch.srn(i: Int): String = scan(i, true) { new }!!
fun Patch.slo(i: Int): String = scan(i, false) { old }!!
fun Patch.sro(i: Int): String = scan(i, true) { old }!!

//Found length-3 fix in 50.966458ms after 146 total and 1 valid samples (2 samples/ms)
//Average time to find human fix: ~347ms (44 trials, 4 expired after 10000ms)
//Average samples before matched: ~29838
//Average repair throughput / ms: ~48
//Average valid repairs detected: ~50

fun contextualRepair() {
  var averageTime = 0
  var totalTrials = 0
  var totSmplSize = 0
  var valSmplSize = 0
  var expiredSize = 0
  var avrgThruput = 0
  val contextCSV = File("context_edits.csv").readTrigramStats()
  preprocessStackOverflow().take(1000).forEach { (broke, humFix, minFix) ->
    val brokeTks = listOf("START") + broke.lexToStrTypesAsPython() + "END"
    val minFixTks = listOf("START") + minFix.lexToStrTypesAsPython() + "END"
//    val brokeTksInt = listOf(Int.MIN_VALUE) + broke.lexToIntTypesAsPython() + Int.MAX_VALUE
    val minFixTksItt = listOf(Int.MIN_VALUE) + minFix.lexToIntTypesAsPython() + Int.MAX_VALUE
    val brokeTksInt = brokeTks.map { it.lexAsPythonIntType() }
    val minFixTksInt = minFixTks.map { it.lexAsPythonIntType() }

    val patchSize = extractPatch(brokeTks, minFixTks).changedIndices().size
    val startTime = TimeSource.Monotonic.markNow()

    val clr = minFixTksInt.drop(1).dropLast(1)
    if (!clr.isValidPython()) {
      println()
      println("Invalid Python: ${clr.joinToString(" ")}")
      println("Invalid Python: ${minFixTksItt.drop(1).dropLast(1).joinToString(" ")}")
      println("Invalid Python: ${minFixTks.joinToString(" ")}")
      println()
      return@forEach
    }

    val initREAs: List<CEAProb> = contextCSV.relevantEditActions(brokeTksInt)
//    println("Total relevant edit actions: ${initREAs.size}\n${initREAs.take(5).joinToString("\n")}\n...")
    val samplerTimeout = 10000L
    var (total, valid) = 0 to 0

    // 1 core results:
    //Found length-1 fix in 87.208375ms after 1720 total and 11 valid samples (19 samples/ms)
    //Average time to find human fix: ~709ms (71 trials, 16 expired after 10000ms)
    //Average samples before matched: ~14864
    //Average repair throughput / ms: ~20
    //Average valid repairs detected: ~54
    // 8 core results:
    //Found length-1 fix in 59.953584ms after 5288 total and 13 valid samples (88 samples/ms)
    //Average time to find human fix: ~814ms (71 trials, 7 expired after 10000ms)
    //Average samples before matched: ~80673
    //Average repair throughput / ms: ~55
    //Average valid repairs detected: ~82
    ForkJoinPool(8).submit {
      generateSequence { brokeTksInt }
        .asStream().parallel() // Measure latency with and without parallelism
        .map {
          try {
            it.sampleEditTrajectory(contextCSV, initREAs)
          } catch (e: Exception) {
            println(brokeTks); e.printStackTrace(); listOf()
          }
        }.takeWhile {
          it != minFixTksInt
            && startTime.elapsedNow().inWholeMilliseconds < samplerTimeout
        }.forEach {
          total++; if (it.drop(1).dropLast(1).isValidPython()) valid++
        }
    }.get()

    val sampleSize = total to valid.coerceAtLeast(1)

    val elapsedTime = startTime.elapsedNow().inWholeMilliseconds.toInt()
    val throughput = sampleSize.first / (elapsedTime + 1)
    if (elapsedTime < samplerTimeout) {
      averageTime += elapsedTime
      totSmplSize += sampleSize.first
      valSmplSize += sampleSize.second
      avrgThruput += throughput
      totalTrials++

      println("""
        Found length-${patchSize} fix in ${startTime.elapsedNow()} after ${sampleSize.first} total and ${sampleSize.second} valid samples (${throughput} samples/ms)
        Average time to find human fix: ~${averageTime / totalTrials}ms ($totalTrials trials, $expiredSize expired after ${samplerTimeout}ms)
        Average samples before matched: ~${totSmplSize / totalTrials}
        Average repair throughput / ms: ~${avrgThruput / totalTrials}
        Average valid repairs detected: ~${valSmplSize / totalTrials}
      """.trimIndent())
    } else println("""
      Sampling timeout expired after $sampleSize (total, valid) samples, ground truth repair was $patchSize edits:
      ${brokeTks.joinToString(" ")}
      ${prettyDiffNoFrills(brokeTks.joinToString(" "), minFixTks.joinToString(" "))}
      ${brokeTksInt.joinToString(" ")}
      ${prettyDiffNoFrills(brokeTksInt.joinToString(" "), minFixTksInt.joinToString(" "))}
    """.trimIndent()).also { expiredSize += 1 }

    println("\n\n")
  }
}

enum class EditType { INS, DEL, SUB }
data class ContextEdit(val type: EditType, val context: Context, val newMid: Int) {
  override fun toString(): String = context.run {
    "$type, ((" + when (type) {
      EditType.INS -> "${left.toPyRuleName()} [${newMid.toPyRuleName()}] ${right.toPyRuleName()}"
      EditType.DEL -> "${left.toPyRuleName()} ~${mid.toPyRuleName()}~ ${right.toPyRuleName()}"
      EditType.SUB -> "${left.toPyRuleName()} [${mid.toPyRuleName()} -> ${newMid.toPyRuleName()}] ${right.toPyRuleName()}"
    } + " // " + when (type) {
      EditType.INS -> "$left [${newMid}] $right"
      EditType.DEL -> "$left ~${mid}~ $right"
      EditType.SUB -> "$left [${mid} -> ${newMid}] $right"
    } + "))"
  }
}
data class CEAProb(val cea: ContextEdit, val idx: Int, val frequency: Int) {
  override fun toString(): String = "[[$cea, $idx, $frequency]]"
}
data class Context(val left: Int, val mid: Int, val right: Int) {
  constructor(left: String, mid: String, right: String) :
    this(left.lexAsPythonIntType(), mid.lexAsPythonIntType(), right.lexAsPythonIntType())
}
data class CEADist(val allProbs: Map<ContextEdit, Int>) {
  val P_delSub = allProbs.filter { it.key.type != EditType.INS }
  val P_insert = allProbs.filter { it.key.type == EditType.INS }
  val P_delSubOnCtx = P_delSub.keys.groupBy { it.context }
  val P_insertOnCtx = P_insert.keys.groupBy { it.context }
}

fun File.readTrigramStats(): CEADist =
  readLines().drop(1).map { it.split(", ") }.associate {
    (ContextEdit(
      type = EditType.valueOf(it[0].trim()),
      context = Context(it[1], it[2], it[3]),
      newMid = it[4].lexAsPythonIntType()
    )
      .also { t -> println(it.joinToString(", ") + " :: $t") }
    ) to it[5].trim().toInt()
  }.let { CEADist(it) }

fun List<Int>.sampleEditTrajectory(
  ceaDist: CEADist,
  initREAs: List<CEAProb>,
  lengthCDF: List<Double> = listOf(0.5, 0.8, 1.0)
): List<Int> {
  // First sample the length of the edit trajectory from the length distribution
  val rand = Math.random()
  val length = lengthCDF.indexOfFirst { rand < it } + 1

  if (initREAs.isEmpty()) return this
  // Now sample an edit trajectory of that length from the edit distribution
  val firstEdit =
    initREAs.normalizeAndSample().let { applyEditAction(it.cea, it.idx + 1) }

  return (1..length).foldIndexed(firstEdit) { i, acc, _ ->
    ceaDist.relevantEditActions(acc)
      .also {
        if (it.isEmpty()) {
          println("$i-th iteration, no relevant edit actions for: ${acc.joinToString(" "){ it.toPyRuleName() }}")
          return@foldIndexed acc
        }
//        else println("Relevant edit actions: ${it}")
      }
      .normalizeAndSample()
//      .also { println("Sampled edit action: $it") }
      .let { acc.applyEditAction(it.cea, it.idx + 1) }
//      .also {
//        println("""
//          Sampled length-$length edit, diff:
//          Before: ${joinToString(" ")}
//          After:  ${prettyDiffNoFrills(joinToString(" "), it.joinToString(" "))}
//        """.trimIndent())
//      }
  }
}

fun List<CEAProb>.normalizeAndSample(total: Int = sumOf { it.frequency }): CEAProb {
  if (total <= 0) throw IllegalArgumentException("Total frequency must be positive!\n$this")
  val sample = Random.nextInt(total)
  var sum = 0
  for (i in indices) {
    sum += this[i].frequency
    if (sum > sample) return this[i]
  }
  return last()
}

fun CEADist.relevantEditActions(snippet: List<Int>): List<CEAProb> =
  snippet.windowed(3)
    .map { Context(it[0], it[1], it[2]) }
    .mapIndexed { idx, ctx ->
      ((P_insertOnCtx[Context(ctx.left, -1, ctx.mid)] ?: listOf()) +
        (P_insertOnCtx[Context(ctx.mid, -1, ctx.right)] ?: listOf()))
        .map { CEAProb(it, idx, P_insert[it]!!) } +
      (P_delSubOnCtx[ctx] ?: listOf())
        .map { CEAProb(it, idx, P_delSub[it]!!) }
    }.flatten()

fun List<Int>.applyEditAction(cea: ContextEdit, idx: Int): List<Int> =
  when (cea.type) {
    EditType.INS -> take(idx) + cea.newMid + drop(idx)
    EditType.DEL -> take(idx) + drop(idx + 1)
    EditType.SUB -> take(idx) + cea.newMid + drop(idx + 1)
  }//.also { println("Start:$this\n${cea.type}/${cea.context}/${cea.newMid}/${idx}\nAfter:$it") }

fun computePatchTrigramStats() =
  preprocessStackOverflowInParallel(take = 100_000).map { (broke, _, minfix) ->
    val brokeLexed = listOf("START") + broke.lexToStrTypesAsPython() + listOf("END")
    val minfixLexed = listOf("START") + minfix.lexToStrTypesAsPython() + listOf("END")
    val patch: Patch = extractPatch(brokeLexed, minfixLexed)
    patch.run {
      changedIndices().map { i ->
        val (old, new) = get(i).old to get(i).new

        if (old == "") "INS, ${sln(i)}, , ${sro(i)}, $new"
        else if (new == "") "DEL, ${sln(i)}, $old, ${sro(i)}, "
        else "SUB, ${sln(i)}, $old, ${sro(i)}, $new"
      }
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }
    .joinToString("\n", "Type, Left, Old Mid, Right, New Mid, Frequency\n") { "${it.first}, ${it.second}" }
    .also {
      File("context_edits.csv").apply {
        writeText(it.reformatCSVIntoPrettyColumns())
        println(readLines().take(100).joinToString("\n"))
        println("Edit trigrams written to: $absolutePath")
      }
    }

fun String.reformatCSVIntoPrettyColumns(): String {
  val lines = split('\n')
  if (lines.isEmpty()) return this

  // Split each line into columns
  val linesByColumns = lines.map { it.split(", ").toMutableList() }

  // Find the max length of each column
  val maxLengths = IntArray(linesByColumns[0].size) { 0 }
  for (columns in linesByColumns)
    for ((index, column) in columns.withIndex())
      maxLengths[index] = maxOf(maxLengths[index], column.trim().length)

  // Pad each element in the columns
  for (columns in linesByColumns)
    for ((index, column) in columns.withIndex())
      columns[index] = column.trim().padEnd(maxLengths[index], ' ')

  // Reassemble the lines and then the entire string
  return linesByColumns.joinToString("\n") { it.joinToString(" , ") }
}