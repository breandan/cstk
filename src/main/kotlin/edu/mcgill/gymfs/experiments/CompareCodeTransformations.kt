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

  TEST_DIR.allFilesRecursively().allCodeFragments().forEach { (c, s) ->
    semanticsPreservingTxs.forEach { tx ->
      val (original, transformed) = c.getContext(4).let { it to tx(it) }
      println(kantorovich(matrixize(original), matrixize(transformed)))
    }
  }
}