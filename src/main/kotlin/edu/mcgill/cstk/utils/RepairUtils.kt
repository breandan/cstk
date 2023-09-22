package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sampling.choose
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.markovian.mcmc.*
import bijectiveRepair
import com.github.difflib.text.*
import edu.mcgill.cstk.experiments.repair.MSK
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

fun Sequence<Π2A<Σᐩ>>.minimizeFix(tokenize: Σᐩ.() -> List<Σᐩ>) =
  map { (broke, fixed) ->
//    val startTime = TimeSource.Monotonic.markNow()
    val (brokeTokens, fixedTokens) = broke.tokenize() to fixed.tokenize()

//  val brokeJoin = brokeTokens.joinToString("")
    val fixedJoin = fixedTokens.joinToString("")
//  val pdiffTok = prettyDiffs(listOf(brokeJoin, fixedJoin), listOf("broken", "original fix"))

    val patch: Patch = extractPatch(brokeTokens, fixedTokens)
    val minEdit = deltaDebug(patch.changes()) { idxs -> patch.apply(idxs).isValidPython() }
// deltaDebug only minimizes contiguous chunks, so here we find the minimal configuration of edits
//      .minimalSubpatch { patch.apply(this).isValidPython() }

//  val pdiff = prettyDiffs(listOf(brokeJoin, minFix), listOf("broken", "minimized fix"))
//  if(pdiff.any { it == '\u001B' } && pdiffTok.filter { !it.isWhitespace() } != pdiff.filter { !it.isWhitespace() }) println(pdiffTok + "\n\n" + pdiff)

//    println("Reduced from ${patch.changes().size} to ${minEdit.size} edits in ${startTime.elapsedNow().inWholeMilliseconds}ms")

//    if(!minFix.isValidPython()) println("Minimized fix is invalid Python: $minFix")

    val minfix= patch.apply(minEdit)

    broke to fixedJoin to minfix
  }

typealias Edit = Π2A<Σᐩ>
typealias Patch = List<Edit>
val Edit.old: Σᐩ get() = first
// If new is empty, then this is a deletion
val Edit.new: Σᐩ get() = second

// returns when there are at least two types of edits (insertions, deletions, changes) choose 2
fun Patch.isInteresting() = changes().let {ch ->
  filterIndexed { index, pair -> index in ch }
    .map { (a, b) -> if(b == "") "D" else if(a == "") "I" else "C" }
    .toSet().size > 1
}
fun Patch.changes(): List<Int> = indices.filter { this[it].old != this[it].new }

fun List<Int>.minimalSubpatch(filter: List<Int>.() -> Boolean): List<Int> =
  (1..size).asSequence().map { choose(it).map { it.toList() } }
  .map { it.filter { it.filter() } }.firstOrNull { it.any() }?.firstOrNull() ?: this

fun Patch.apply(indices: List<Int>, separator: Σᐩ = ""): Σᐩ =
  mapIndexed { i, it -> if (i in indices) it.new else it.old }.joinToString(separator)

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

fun CFG.metrizedRepair(refStr: List<Σᐩ>, mc: MarkovChain<Σᐩ>): List<Repair> =
  solve(List(refStr.size + 3) { "_" }) {
    levenshtein(it.tokens, refStr) * mc.score(listOf("BOS") + it.tokens + "EOS").toFloat()
  }.map {
    val tokens = it.tokenizeByWhitespace()
    Repair(refStr, listOf(), tokens, mc.score(listOf("BOS") + it + "EOS"))
  }

// TODO: Generify to accept List<T>
fun parallelRepair(
  prompt: Σᐩ,
  fillers: Collection<Σᐩ>,
  hints: List<Int> = emptyList(),
  maxEdits: Int = 2,
  admissibilityFilter: List<Σᐩ>.() -> Boolean,
  scoreEdit: ((List<Σᐩ>) -> Double)? = null,
): List<Repair> {
  var bestRepair = Double.MAX_VALUE
  val delim = List(prompt.length) { "-" }.joinToString("")
  println("$delim\nBest repairs so far:\n$delim")
  // We intersperse the prompt with empty strings to enable the repair of the first and last token
  // as well as insertion of tokens by the repair algorithm, which only considers substitutions
  val spacingLength = maxEdits.coerceAtMost(2)
  val promptTokens = prompt.tokenizeByWhitespace().intersperse(spacingLength)
  // Remap the hints to the new indices in the interspersed prompt tokens
  val remappedHints = hints.map { (spacingLength + 1) * it + 2 }

  val deck = fillers + promptTokens.toSet() - "\""

  val clock: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow()
  return bijectiveRepair(
    promptTokens = promptTokens,
    deck = deck,
    hints = remappedHints,
    maxEdits = maxEdits,
    takeMoreWhile = { clock.elapsedNow().inWholeMilliseconds < TIMEOUT_MS },
    admissibilityFilter = admissibilityFilter,
    scoreEdit = scoreEdit ?: { 0.0 },
    diagnostic =
      if (scoreEdit != null) ({
          val score = scoreEdit(it.result)
          if (score < bestRepair) {
            println("Δ=${it.scoreStr()} repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.resToStr())}")
//            println("(LATEX) Δ=$score repair: ${latexDiffSingleLOC(prompt, it)}")
            bestRepair = score
          }
      })
      else ({
          val levDiff = it.edit.size.toDouble()
          if (levDiff < bestRepair) {
            println("Δ=$levDiff repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.resToStr())}")
//              println("(LATEX) Δ=$levDiff repair: ${latexDiffSingleLOC(prompt, it)}")
            bestRepair = levDiff
          }
      })
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