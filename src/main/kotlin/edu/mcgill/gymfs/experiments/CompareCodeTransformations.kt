package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.kantorovich

fun main() {
  val syntaxAlteringTransformations =
    listOf(String::mutateSyntax, String::shuffleLines)

  val semanticsPreservingTxs =
    listOf(String::renameTokens, String::swapMultilineNoDeps, String::addDeadCode)

  val semanticsAlteringTxs =
    listOf(String::permuteArgumentOrder, String::fuzzLoopBoundaries, String::swapPlusMinus)

  // Measure average distributional shift introduced by each code transformation
  // as measured by wordmover distance metric. https://proceedings.mlr.press/v37/kusnerb15.pdf

  TEST_DIR.allFilesRecursively().allCodeFragments().map { (c, s) ->
    semanticsPreservingTxs.mapNotNull { tx ->
      val (original, transformed) = c.getContext(4).let { it to tx(it) }
      if (original == transformed) return@mapNotNull null
      val distance = kantorovich(matrixize(original), matrixize(transformed))
      println("${tx.name}:".padEnd(20, ' ') + distance)
      tx to distance
    }
  }.flatten().groupBy { it.first }
    // Average embedding distance across code transformation
    .mapValues { (_, v) -> v.map { it.second }.average() }
}