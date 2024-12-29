package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.visualization.alsoCopy
import ai.hypergraph.markovian.mcmc.toMarkovChain
import com.beust.klaxon.Klaxon
import com.google.common.util.concurrent.AtomicLongMap
import edu.mcgill.cstk.experiments.probing.MakeMore
import edu.mcgill.cstk.utils.*
import java.io.File
import java.net.URL
import java.net.URLEncoder
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.function.Function
import java.util.stream.*
import kotlin.collections.filter
import kotlin.math.*
import kotlin.streams.*
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds

/*
./gradlew collectSummaryStats
 */
fun main() {
//  stackOverflowSnips().computeLengthDistributionStats()
//  stackOverflowSnips().computeRawTokenFrequencies()
//  seq2ParseSnips().computeBigramFrequencies()
//  computeErrorSizeFreq()
//  computePatchStats()
//  readPy150()
//  paperExample()
//  computeSnippetLengthDistribution()
//  computeLevDistDistribution()
//  fetchLevenshteinAlignment()
//  collectPCFGQuintuples()
//  collectNaturallySmallRepairs()
//  prepareUnsupervisedMakemoreDataset()
//  checkSemanticAdmissibility()
//  collectShortRepairSpecimens()
//  collectSyntheticRepairs()
//  collectPairwisePythonRepairs()
//    println(naturallySmallRepairs.map { it.second }.joinToString("\n").parseAndCountActiveSymbols().alsoCopy())
//  estimateLevenshteinDistanceDistribution()
//  computePatchTrigramStats()
//  corruptedTrigramStats()
//  readBIFI().toList()
//  computeEditLocationFrequency()
//  computeRelativeIntraEditDistance()
//  totalCharacterEditDistance()
//  mostCommonSubstitutions()
//  testContextEditIssue()
//  measureThroughput()
//  MakeMore.checkSamples()
//  prepareBreakerDataset()
  MakeMore.checkPairwiseSamples()
//  MakeMore.previewSamples()
//  prepareFixerDataset()
//  correctNames()
//  errorPredDiscrepancy()
}

fun errorPredDiscrepancy() {
  var (sum, acc, total) = 0 to 0 to 0
  sizeAndDistBalancedRepairsUnminimized.forEach {
    val trueDist = levenshtein(it.π1, it.π2)
    println("TrueDist = $trueDist")
    val predDist = MakeMore.checkEditRadiusPred(it.π1)
    println("PredDist = $predDist")
    sum += (trueDist - predDist).absoluteValue
    acc += if (trueDist == predDist) 1 else 0
    println("Running average edit distance discrepancy: sum=${sum.toDouble() / ++total}, acc=${acc.toDouble() / total}")
  }
}

fun prepareFixerDataset() =
  File("names.txt").readLines().stream().parallel()
    .map { MakeMore.decode(it) }
    .flatMap { goodCode ->
      goodCode.naturalPythonCorruptions().distinct()
        .filter { levenshtein(goodCode, it) in 1..3 && it !in vanillaS2PCFG.language }
        .take(10).map { it to goodCode }.asStream()
    }.forEach { (a, b) ->
      val i = a.tokenizeByWhitespace().map { MakeMore.PyTokMap.tm[it] }.joinToString("")
      val o = b.tokenizeByWhitespace().map { MakeMore.PyTokMap.tm[it] }.joinToString("")
      val d = levenshtein(a, b)
      println("|$i $d $o}")
    }

fun measureThroughput() {
  var str = ""
  val startTime = TimeSource.Monotonic.markNow()
  var i = 0

  while (startTime.elapsedNow() < 10.seconds) {
    try {
      str += MakeMore.nextTokensReadable(str).random()
      println(str)
    } catch (_: Exception) { str = "" }
    i++
  }

  println("Done: ${i/10f} tok/sec")
}

fun prepareUnsupervisedMakemoreDataset() {
  streamBIFIContents().forEach {
    val str = it.mapToUnquotedPythonTokens() + " NEWLINE"
    if (str in vanillaS2PCFG.language)
      println(str.encodeToMakemore())
  }
}

fun Σᐩ.encodeToMakemore() = tokenizeByWhitespace().map { MakeMore.PyTokMap.tm[it]!! }.joinToString("")

fun prepareBreakerDataset() {
  val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
  val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()
  contents.lines().asSequence().windowed(4, 4).map { it[0] to it[1] to it[2] to it[3] }
    .asStream().parallel()
    .map { (broke, fixed) -> broke.addNewLineIfMissing() to fixed.addNewLineIfMissing() }
    .map { (broke, fixed) -> broke to fixed to levenshtein(broke, fixed) }
    .forEach { (broke, fixed, d) -> try { println("${(MakeMore.PyTokMap.size + 34).toChar()}" + fixed.encodeToMakemore() + " $d " + broke.encodeToMakemore() + "${(MakeMore.PyTokMap.size + 35).toChar()}") } catch (_: Exception) {} }
}

fun checkSemanticAdmissibility() {
  sizeAndDistBalancedRepairsUnminimized.toList().forEach {
    println(prettyDiffHorizontal(it.π3, it.π4))
    println("" + it.π3.isInterpretablePython() + " :: " + it.π4.isInterpretablePython())
  }
}

// ./gradlew collectSummaryStats 2>&1 | tee shortRepairSpecimens.log
fun collectShortRepairSpecimens() {
  preprocessStackOverflowQuickly()
    .filter { (a, b) ->
      a.lines().size < 10 &&
      (a.lines().maxOf { it.length } + b.lines().maxOf { it.length }) < 92 &&
      levenshtein(a.mapToUnquotedPythonTokens(), b.mapToUnquotedPythonTokens()) in 2..5 &&
      "\u001B" in prettyDiffHorizontal(a, b, "broken", "fixed")
    }.forEach { (a, b) ->
      println(a.mapToUnquotedPythonTokens())
      println(prettyDiffHorizontal(a, b, "broken", "fixed"))
      latexDiffMultilineStrings(a, b).let { (a, b) -> println("$a\n\n$b\n\n") }
    }
}

fun fetchLevenshteinAlignment() {
  preprocessStackOverflowQuickly()
    .map { (a, b) -> a.mapToUnquotedPythonTokens() to b.mapToUnquotedPythonTokens() }
    .filter { (a, b) -> b.length < 120 }
    .filter { (a, b) -> levenshtein(a, b) == 4 }
    .map { (a, b) -> levenshteinAlign(b, a).let { it.paintANSIColors() + "\n" + it.printLaTeX() } }
    .filter { ANSI_GREEN_BACKGROUND in it && ANSI_RED_BACKGROUND in it && ANSI_ORANGE_BACKGROUND in it }
    .forEach { println("$it\n") }
}

/**
 * cd src/main/resources/datasets/python &&
 * wget https://nlp.stanford.edu/projects/myasu/BIFI/data_minimal.zip &&
 * unzip data_minimal.zip && rm data_minimal.zip
 */

fun streamBIFIContents(
  good: Boolean = true,
  kind: String = if (good) "good" else "bad",
  filename: String = "src/main/resources/datasets/python/bifi/data/orig_${kind}_code/orig.${kind}.json",
  file: File = File(filename)
): Stream<String> =
  file.readLines().stream().parallel()
    .filter { it.trimStart().startsWith("\"code_string\": \"") }
    .map {
      val json = "{$it}"
      val parsedObject = Klaxon().parseJsonObject(json.reader())
      parsedObject.string("code_string")
    }

fun collectPCFGTriples() {
  val i = AtomicInteger(0)
  streamBIFIContents()
    .map { it.mapToUnquotedPythonTokens() + " NEWLINE" }
    .flatMap { if (i.incrementAndGet() % 100 == 0) println(i); vanillaS2PCFG.parseForest(it).map { it.triples() }.flatten().stream() }
    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
    .toList().sortedByDescending { it.second }
    .also {
//      File("/scratch/b/bengioy/breandan/pcfg_bifi.csv")
      File(File("").absolutePath + "/src/main/resources/models/pcfg3_BIFI.csv").writeText(it.joinToString("\n") {
        (t, count) ->
        "${t.π1} ${t.π2} ${t.π3} ::: $count" })
    }.forEach { (a, b) ->
      println("${a.π1} ${a.π2} ${a.π3} ::: $b")
    }
}

fun collectPCFGQuintuples() {
  val i = AtomicInteger(0)
  streamBIFIContents()
    .map { it.mapToUnquotedPythonTokens() + " NEWLINE" }
    .flatMap { if (i.incrementAndGet() % 1000 == 0) println(i); vanillaS2PCFG.parseForest(it).map { it.quintuples() }.flatten().stream() }
    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
    .toList().sortedByDescending { it.second }
    .also {
//      File("/scratch/b/bengioy/breandan/pcfg_bifi.csv")
      File(File("").absolutePath + "/src/main/resources/models/pcfg5_BIFI.csv").writeText(it.joinToString("\n") {
          (t, count) ->
        "${t.π1} ${t.π2} ${t.π3} ${t.π4} ${t.π5} ::: $count" })
    }.forEach { (a, b) ->
      println("${a.π1} ${a.π2} ${a.π3} ${a.π4} ${a.π5} ::: $b")
    }
}

fun paperExample() {
  val broken = "f = f.f(1:, 1:)"
  val brokeLexed = broken.mapToUnquotedPythonTokens() + " NEWLINE"
  val fixed = "f = f.f[1:, 1:]"
  val fixedLexed = fixed.mapToUnquotedPythonTokens() + " NEWLINE"
  println("Valid pair: " + (brokeLexed !in vanillaS2PCFG.language && fixedLexed in vanillaS2PCFG.language))

  MAX_TOKENS = brokeLexed.tokenizeByWhitespace().size + 5

  val intGram = vanillaS2PCFG.jvmIntersectLevFSA(makeLevFSA(brokeLexed, 2))
  println(intGram.size)

  val clock = TimeSource.Monotonic.markNow()
  intGram.toPTree()
    .sampleDirectlyWOR(stoppingCriterion = { clock.elapsedNow() < 20.seconds })
    .distinct().forEach { println(levenshteinAlign(brokeLexed, it).printLaTeX()) }
}

fun readPy150() {
  println(P_BIFI_PY150.size)
}

/** See: [symbolCounts] */
fun Σᐩ.parseAndCountActiveSymbols(): Σᐩ {
  val ntCounts =
    lines().parallelStream().map { "$it NEWLINE" }
      .flatMap { vanillaS2PCFG.parseAll(it, false).flatMap { it.activeSymbols().toList() }.stream() }
      .toList().groupingBy { it }.eachCount()

  val postProc = ".lines().filter { it.isNotEmpty() }.map { it.split(\" ::: \") }.associate { (nt, count) -> nt.trim() to count.trim().toInt() }"

  return ntCounts.entries.sortedBy { it.value }
    .joinToString("\n", "val symbolCounts by lazy { \"\"\"\n", "\n\"\"\"$postProc }") {
      "  ${it.key} ::: ${it.value}"
    }
}

// Current distribution (1000 samples):
// 1: 37.90%, 2: 18.90%, 3: 9.50%, 4+: 33.70%
fun estimateLevenshteinDistanceDistribution() {
  var sampleSize = 0
  val totalByLevDist = AtomicLongMap.create<Int>()
  val lenDist = AtomicLongMap.create<Int>()
  preprocessStackOverflow(10, 1..100)
    .map { (a, _, c) -> a to c }.distinct().take(1000)
    .forEach { (broke, minfix) ->
      val brokeLexed = broke.lexToStrTypesAsPython()
      val minfixLexed = minfix.lexToStrTypesAsPython()
      val levDist = levenshtein(brokeLexed, minfixLexed)
      totalByLevDist.incrementAndGet(levDist)
      lenDist.incrementAndGet(brokeLexed.size)
      sampleSize++

      // Normalize and print the distribution
      print("Current distribution ($sampleSize samples):\n${totalByLevDist.asMap().entries.filter { it.key <= 3 }
        .sortedBy { it.key }.joinToString(", ") { (dist, count) -> "$dist: ${"%.2f".format(100.0 * count / sampleSize)}%" }}")
      // Now print the distribution > 3
      println(", 4+: ${"%.2f".format(100.0 * totalByLevDist.asMap().entries.filter { it.key > 3 }.sumOf { it.value } / sampleSize)}%")
//      println("Current distribution:\n${lenDist.asMap().entries.sortedBy { it.key }.joinToString(", ") { (dist, count) -> "$dist: ${"%.2f".format(100.0 * count / sampleSize)}%" }}")
    }
}

fun String.addNewLineIfMissing() = if (endsWith("NEWLINE")) this else "$this NEWLINE"

fun collectSyntheticRepairs() {
  readBIFIContents()
    .map { it.mapToUnquotedPythonTokens() }
    .filter { it.tokenizeByWhitespace().size in 3..MAX_TOKENS }
    .asStream().parallel()
    .flatMap { goodCodeTks ->
      val goodCode = "$goodCodeTks NEWLINE"
      goodCode.naturalPythonCorruptions().distinct().filter {
        levenshtein(goodCode, it) <= MAX_RADIUS &&
            it !in vanillaS2PCFG.language
      }.take(10).map { it to goodCode }.asStream()
    }
    .forEach { (a, c) -> println("$a\n$c\n") }
}

// Takes ~1.5 hrs to run on M1 serially, ~17 mins w/ parallel streaming
fun collectNaturallySmallRepairs() {
  MAX_PATCH_SIZE = 6
  val encoder = Base64.getEncoder()
  val filename = "src/main/resources/datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
    .also { File(it).also { if (it.exists()) it.delete(); it.createNewFile() } }

  var processed = 0
  preprocessStackOverflowStreaming(MAX_PATCH_SIZE, 3..100)
    .map { (a, c) ->
      if (processed++ % 100 == 0) println(processed)
      "${a.mapToUnquotedPythonTokens()}\n${c.mapToUnquotedPythonTokens()}\n" +
      "${encoder.encodeToString(a.toByteArray())}\n${encoder.encodeToString(c.toByteArray())}\n"
  }
    .distinct().toList().forEach { File(filename).appendText(it) }
}

fun collectPairwisePythonRepairs() {
  MAX_PATCH_SIZE = 3
  val lengthBounds = 20..40
  val (brokeSnips, fixedSnips) = mutableListOf<Σᐩ>() to mutableListOf<Σᐩ>()
  val errorMessages = mutableListOf<Σᐩ>()
  preprocessStackOverflow(MAX_PATCH_SIZE, lengthBounds)
    .map { (a, _, c) -> a to c }.distinct().take(1000)
    .forEach { (broke, minFix) ->
      broke.mapToUnquotedPythonTokens().also { brokeSnips.add(it) }
      minFix.mapToUnquotedPythonTokens().also { fixedSnips.add(it) }
      println("Parsing: $broke\nError message:${broke.getPythonErrorMessage().also { errorMessages.add(it) }}")
      println(levenshteinAlign(broke.mapToUnquotedPythonTokens(),
        minFix.mapToUnquotedPythonTokens()).paintANSIColors())
    }

  """
  // The following are length $lengthBounds Python statements with a human fix <=$MAX_PATCH_SIZE Levenshtein edits away
  val invalidLexedPythonStatements$MAX_PATCH_SIZE = ""${'"'}
     ${brokeSnips.joinToString("\n")} 
  ""${'"'}.trimIndent()

  val validLexedPythonStatements$MAX_PATCH_SIZE = ""${'"'}
     ${fixedSnips.joinToString("\n")} 
  ""${'"'}.trimIndent()
  
  ${""/*fixedSnips.joinToString("\n").parseAndCountActiveSymbols()*/}
  
  val errorMessages$MAX_PATCH_SIZE = ""${'"'}
     ${errorMessages.joinToString("\n")}
  ""${'"'}.trimIndent()
  """
    .trimIndent().also { it.alsoCopy() }
    .also {
      File("src/main/kotlin/edu/mcgill/cstk/experiments/repair/data/PairwisePythonRepairsL$MAX_PATCH_SIZE.kt")
        .writeText("""
          package edu.mcgill.cstk.experiments.repair.data

          $it
        """.trimIndent())
    }
}

fun testContextEditIssue() {
  ContextEdit(EditType.INS, Context("')'", "", "NEWLINE"), "':'".toPythonIntType())
    .let { it to it.toString() }
//  INS, (( ')' [')'] ',' // 53 [53] 54 ))
//  INS, (( ')' [':'] NEWLINE // 53 [55] 39 ))
//  INS, (( ')' [':'] NEWLINE // 53 [55] 39 ))
//  INS, (( ')' [':'] NEWLINE // 53 [55] 39 ))

  .also { println(it.second) }
    .also { println(it.first in contextCSV.allProbs) }
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

// Returns a distribution of snippet lengths in buckets of 10, up to maximum 100+
// 10, 5.00%
// 20, 10.00%
// 30, 15.00%
//    ...
// 100+, 100.00%
fun computeSnippetLengthDistribution() {
  val buckets: MutableMap<IntRange, Int> =
    0.rangeTo(10).associate { (it * 10)..(it * 10 + 9) to 0 }.toMutableMap()
  buckets[100..Int.MAX_VALUE] = 0

  var total = 0
  preprocessStackOverflowQuickly().limit(10_000).toList().map { it.first }
    .map { it.lexToStrTypesAsPython().size }
    .forEach { len ->
      val bucket = buckets.keys.first { len in it }
      buckets[bucket] = buckets.getValue(bucket) + 1
      total++
    }

//  buckets.toList().sortedBy { it.first.first }.joinToString("\n") { (range, count) ->
//    "${range.first}, ${"%.2f".format(100.0 * count / total.toDouble())}%"
//  }.also { println(it) }
  // Cumulative distribution

  buckets.toList().sortedBy { it.first.first }.runningFold(0 to 0) { (prev, prevCount), (range, count) ->
    range.first to (prevCount + count)
  }.map { (n, count) -> n to count.toDouble() / total.toDouble() }
    .joinToString("\n") { "${it.first}, ${"%.2f".format(100.0 * it.second)}%" }
    .also { println(it) }
}

fun computeLevDistDistribution( ){
  val buckets = 0.rangeTo(10).associate { it..it+1 to 0 }.toMutableMap()
  buckets[10..Int.MAX_VALUE] = 0

  var total = 0
  preprocessStackOverflowQuickly().limit(10_000).toList()
    .map { it.first.lexToStrTypesAsPython() to it.second.lexToStrTypesAsPython() }
    .map { (broke, minfix) ->
      if (broke.size - minfix.size > 10) Int.MAX_VALUE
      else levenshtein(broke, minfix)
    }
    .forEach { len ->
      val bucket = buckets.keys.first { len in it }
      buckets[bucket] = buckets.getValue(bucket) + 1
      total++
    }

  buckets.toList().sortedBy { it.first.first }.runningFold(0 to 0) { (prev, prevCount), (range, count) ->
    range.first to (prevCount + count)
  }.map { (n, count) -> n to count.toDouble() / total.toDouble() }
    .joinToString("\n") { "${it.first}, ${"%.2f".format(100.0 * it.second)}%" }
    .also { println(it) }
}

fun preprocessStackOverflowQuickly(
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
  fixedSnippets: Sequence<String> = readContents("parse_fixes.json"),
): Stream<Π2A<Σᐩ>> =
  brokeSnippets.zip(fixedSnippets).asStream().parallel()

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

fun readGoodBIFIAndCorrupt() =
  readBIFIContents().take(100_000)
    .map { "\n$it\n".lexToStrTypesAsPython().joinToString(" ") }
    .map { it.syntheticallyCorrupt() to it }

fun String.syntheticallyCorrupt(): String =
  tokenizeByWhitespace().toMutableList().let { tokens ->
    var corrupted = tokens.map { it.toPythonIntType() }

    val initREAs = contextCSV.relevantEditActions(corrupted)
    while (corrupted.isValidPython()) {
      corrupted = corrupted.sampleEditTrajectory(contextCSV, initREAs).first
    }

    corrupted.joinToString(" ") { it.toPyRuleName() }
  }

var progress = 0
fun computePatchTrigramStats(toTake: Int = 100000) =
  preprocessStackOverflowInParallel(take=toTake).map { (broke, _, minfix) ->
    val brokeLexed = listOf("BOS") + broke.lexToStrTypesAsPython() + listOf("EOS")
    val minfixLexed = listOf("BOS") + minfix.lexToStrTypesAsPython() + listOf("EOS")
    val patch: Patch = extractPatch(brokeLexed, minfixLexed)
    progress++.also { if (it % 100 == 0) println("Processed $it/$toTake patches") }
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

// Same as above, but with the reverse patch order (minfix -> broke) instead
fun corruptedTrigramStats(toTake: Int = 100000) =
  preprocessStackOverflowInParallel(take=toTake).map { (broke, _, minfix) ->
    val brokeLexed = listOf("BOS") + broke.lexToStrTypesAsPython() + listOf("EOS")
    val minfixLexed = listOf("BOS") + minfix.lexToStrTypesAsPython() + listOf("EOS")
    val patch: Patch = extractPatch(minfixLexed, brokeLexed)
    progress++.also { if (it % 100 == 0) println("Processed $it/$toTake patches") }
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
      File("context_typos.csv").apply {
        writeText(it.reformatCSVIntoPrettyColumns())
        println(readLines().take(100).joinToString("\n"))
        println("Typo trigrams written to: $absolutePath")
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

fun Sequence<Σᐩ>.train(csv: File) {
  measureTimedValue {
    println("Training $MARKOV_MEMORY Markov chain")
    asStream().parallel().map {
      "\n$it\n".mapToUnquotedPythonTokens().let { "BOS $it EOS" }
        .tokenizeByWhitespace().asSequence().toMarkovChain(MARKOV_MEMORY)
    }.reduce { t, u -> t + u }.get()
      .also { csv.also { println("Writing CSV to ${it.absolutePath}") }.writeText(it.toCSV()) }
  }.let { println("Trained $MARKOV_MEMORY-gram Markov chain on ${it.value.counter.total.get()} " +
      "PY150 tokens in ${it.duration.inWholeSeconds}s"); it.value }
}