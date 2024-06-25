package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.markovian.mcmc.MarkovChain
import bijectiveRepair
import edu.mcgill.cstk.experiments.repair.MSK
import java.util.stream.Stream
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

fun Stream<Π2A<Σᐩ>>.minimizeFix(
  tokenize: Σᐩ.() -> List<Σᐩ>,
  isValid: Σᐩ.() -> Boolean
): Stream<Π3A<Σᐩ>> =
  map { (broke, fixed) ->
    minimizeFix(broke, tokenize, fixed, "", isValid) }

fun Sequence<Π2A<Σᐩ>>.minimizeFix(
  tokenize: Σᐩ.() -> List<Σᐩ>,
  isValid: Σᐩ.() -> Boolean
): Sequence<Π3A<Σᐩ>> =
  map { (broke, fixed) ->
    minimizeFix(broke, tokenize, fixed, "", isValid) }

fun CFG.metrizedRepair(refStr: List<Σᐩ>, mc: MarkovChain<Σᐩ>): List<Repair> =
  solve(List(refStr.size + 3) { "_" }) {
    levenshtein(it.tokens, refStr) * mc.score(it.tokens).toFloat()
  }.map {
    val tokens = it.tokenizeByWhitespace()
    Repair(refStr, listOf(), tokens, mc.score(tokens))
  }

fun CFG.ptreeRepair(
  refStr: List<Σᐩ>,
  scoreEdit: ((List<Σᐩ>) -> Double),
  clock: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow()
) =
  solveSeq(List(refStr.size + 3) { "_" })
    .map {
      val tokens = it.tokenizeByWhitespace()
      Repair(refStr, listOf(), tokens, scoreEdit(tokens))
    }
    .filter { levenshtein(it.result, refStr) < 3 }
    .takeWhile { clock.elapsedNow().inWholeMilliseconds < TIMEOUT_MS }
    .sortedBy { it.score }.toList()

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
//          println("(LATEX) Δ=$score repair: ${latexDiffSingleLOC(prompt, it)}")
          bestRepair = score
        }
      })
      else ({
        val levDiff = it.edit.size.toDouble()
        if (levDiff < bestRepair) {
          println("Δ=$levDiff repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.resToStr())}")
//            println("(LATEX) Δ=$levDiff repair: ${latexDiffSingleLOC(prompt, it)}")
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
  val tokens = coarsened.split(' ')
}