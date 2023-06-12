package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.intersperse
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.types.*
import bijectiveRepair
import com.github.difflib.text.*
import edu.mcgill.cstk.experiments.repair.MSK
import java.util.stream.Stream
import kotlin.math.*
import kotlin.time.TimeSource

fun Σᐩ.constructPromptByMaskingRandomSyntax(
  eligibleTokensToMask: Set<Σᐩ> = COMMON_BRACKETS,
  numTokensToMask: Int = 1,
  tokenize: Σᐩ.() -> List<Σᐩ> = Σᐩ::defaultTokenizer
): Σᐩ =
  tokenize().toMutableList().let { codeTokens ->
//    println("Code tokens: $codeTokens")
//    println("Eligible tokens to mask: $eligibleTokensToMask")
    codeTokens.indices.filter { codeTokens[it].trim() in eligibleTokensToMask }
//      .also { println("Indices of eligible tokens to mask: $it") }
      .shuffled().take(numTokensToMask)
      .forEach { codeTokens[it] = MSK }
    codeTokens
  }.joinToString(" ")

val COMMON_BRACKETS = "()[]{}".map { "$it" }.toSet()
fun Σᐩ.defaultTokenizer(): List<Σᐩ> =
  split(Regex("[\\(\\)\\[\\]{}]|___".let { "((?<=($it))|(?=($it)))" }))

fun Stream<Π2A<Σᐩ>>.minimizeFix(tokenize: Σᐩ.() -> List<Σᐩ>) =
  map { (broke, fixed) ->
    val (brokeTokens, fixedTokens) = broke.tokenize() to fixed.tokenize()

//  val brokeJoin = brokeTokens.joinToString("")
    val fixedJoin = fixedTokens.joinToString("")
//  val pdiffTok = prettyDiffs(listOf(brokeJoin, fixedJoin), listOf("broken", "original fix"))

    val patch: Patch = extractPatch(brokeTokens, fixedTokens)
    val minEdit = deltaDebug(patch.changes()) { idxs -> patch.apply(idxs).isValidPython() }
    val minFix = patch.apply(minEdit)
//  val pdiff = prettyDiffs(listOf(brokeJoin, minFix), listOf("broken", "minimized fix"))
//  if(pdiff.any { it == '\u001B' } && pdiffTok.filter { !it.isWhitespace() } != pdiff.filter { !it.isWhitespace() }) println(pdiffTok + "\n\n" + pdiff)
    broke to fixedJoin to minFix
  }

typealias Edit = Π2A<Σᐩ>
typealias Patch = List<Edit>
val Edit.old: Σᐩ get() = first
val Edit.new: Σᐩ get() = second

fun Patch.changes(): List<Int> = indices.filter { this[it].old != this[it].new }

fun Patch.apply(indices: List<Int>) =
  mapIndexed { i, it -> if (i in indices) it.new else it.old }.joinToString("")

fun extractPatch(original: List<Σᐩ>, new: List<Σᐩ>): Patch =
  DiffRowGenerator.create().build()
    .generateDiffRows(original, new).mapIndexed { i, it ->
      when (it.tag) {
        DiffRow.Tag.INSERT -> ("" to it.newLine)
        DiffRow.Tag.CHANGE -> (it.oldLine to it.newLine)
        DiffRow.Tag.DELETE -> (it.oldLine to "")
        DiffRow.Tag.EQUAL ->  (it.oldLine to it.newLine)
      }
    }

fun <T> deltaDebug(elements: List<T>, n: Int = 2, checkValid: (List<T>) -> Boolean): List<T> {
  // If n granularity is greater than number of tests, then finished, simply return passed in tests
  if (elements.size < n) { return elements }

  // Cut the elements into n equal chunks and try each chunk
  val chunkSize = (elements.size.toDouble() / n).roundToInt()

  val chunks = elements.windowed(chunkSize, chunkSize, true)

  chunks.forEachIndexed { index, chunk ->
    val otherChunk = elements.subList(0, index*chunkSize) +
      elements.subList(min((index+1)*chunkSize, elements.size), elements.size)

    // Try to other, complement chunk first, with theory that valid elements are closer to end
    if (checkValid(otherChunk)) return deltaDebug(otherChunk, 2, checkValid)

    // Check if running this chunk works
    if (checkValid(chunk)) return deltaDebug(chunk, 2, checkValid)
  }

  // If size is equal to number of chunks, we are finished, cannot go down more
  if (elements.size == n) return elements

  // If not chunk/complement work, increase granularity and try again
  return if (elements.size < n * 2) deltaDebug(elements, elements.size, checkValid)
  else deltaDebug(elements, n * 2, checkValid)
}

// TODO: Generify to accept List<T>
fun parallelRepair(
  prompt: Σᐩ,
  fillers: Set<Σᐩ>,
  maxEdits: Int = 2,
  admissibilityFilter: Σᐩ.() -> Boolean,
  scoreEdit: ((Σᐩ) -> Double)? = null,
): List<Repair> {
  var bestRepair = Double.MAX_VALUE
  val delim = List(prompt.length) { "-" }.joinToString("")
  println("$delim\nBest repairs so far:\n$delim")
  // We intersperse the prompt with empty strings to enable the repair of the first and last token
  // as well as insertion of tokens by the repair algorithm, which only considers substitutions
  val promptTokens = prompt.tokenizeByWhitespace().intersperse(maxEdits.coerceAtMost(2))

  val clock: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow()
  return bijectiveRepair(
    promptTokens = promptTokens,
    fillers = fillers,
    maxEdits = maxEdits,
    takeMoreWhile = { clock.elapsedNow().inWholeMilliseconds < TIMEOUT_MS },
    admissibilityFilter = admissibilityFilter,
    scoreEdit = scoreEdit ?: { 0.0 },
    diagnostic =
    if (scoreEdit != null) {
      {
        val score = scoreEdit(it.result)
        if (score < bestRepair) {
          println("Δ=${it.scoreStr()} repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.result)}")
//          println("(LATEX) Δ=$score repair: ${latexDiffSingleLOC(prompt, it)}")
          bestRepair = score
        }
      }
    }
    else {
      {
        val levDiff = it.edit.size.toDouble()
        if (levDiff < bestRepair) {
          println("Δ=$levDiff repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.result)}")
//            println("(LATEX) Δ=$levDiff repair: ${latexDiffSingleLOC(prompt, it)}")
          bestRepair = levDiff
        }
      }
    }
  ).toList()
//    .parallelStream().map {
//      it.editSignatureEquivalenceClass(
//        tokens = (fillers + promptTokens).shuffled().toSet() - "\"",
//        filter =  { it in seq2parsePythonCFG.language },
//        score = { scoreEdit?.invoke(it) ?: 0.0 }
//      ).also { it.time = clock.elapsedNow().inWholeMilliseconds }
//    }.toList()
    .distinctBy { it.result }.toList()
    .sortedWith(compareBy({ it.edit.size }, { it.score }))
}

data class CodeSnippet(
  val originalCode: Σᐩ,
  val coarsened: Σᐩ,
  val errorMsg: Σᐩ,
  val groundTruth: Σᐩ? = null
) {
  val tokens = coarsened.split(" ")
}