package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.*
import edu.mcgill.kaliningraph.show
import java.io.File
import kotlin.math.absoluteValue
import kotlin.reflect.KFunction1

fun main() {
  val syntaxAlteringTransformations =
    listOf(String::mutateSyntax, String::shuffleLines)

  val semanticsPreservingTxs =
    listOf(String::renameTokens, String::swapMultilineNoDeps, String::addExtraLogging)

  val semanticsAlteringTxs =
    listOf(String::permuteArgumentOrder, String::fuzzLoopBoundaries, String::swapPlusMinus)

  //  compareDistributionalShift(semanticsPreservingTxs)
//  compareTsneEmbeddings(String::swapMultilineNoDeps)
//  compareTsneEmbeddings(String::addDeadCode)
//  compareTsneEmbeddings(String::renameTokens)

  analyzeDimensionalShift(String::swapMultilineNoDeps).let { println(it.joinToString(",")) }
  analyzeDimensionalShift(String::addExtraLogging).let { println(it.joinToString(",")) }
  analyzeDimensionalShift(String::renameTokens).let { println(it.joinToString(",")) }
}

private fun analyzeDimensionalShift(tx: KFunction1<String, String>) =
    TEST_DIR.allFilesRecursively().allCodeFragments().take(100).mapIndexed() { i, (c, s) ->
      val (original, transformed) = c.getContext(4).let { it to tx(it) }
      matrixize(original).average().zip(matrixize(transformed).average())
        .map { (a, b) -> (a - b).absoluteValue }.toDoubleArray()
    }.toList().toTypedArray().average().normalize()

private fun compareTsneEmbeddings(tx: KFunction1<String, String>) {
  val (vecs, labels) =
    TEST_DIR.allFilesRecursively().allCodeFragments().take(100).mapIndexed() { i, (c, s) ->
      val (original, transformed) = c.getContext(4).let { it to tx(it) }
      listOf(matrixize(original).average() to "o", matrixize(transformed).average() to "t")
    }.flatten().unzip()
  val d2vecs = vecs.toTypedArray().reduceDim()
  val plot = plotTsneEmbeddingsWithLabels(d2vecs, labels)
  File.createTempFile("compare_${tx.name}", ".html")
    .apply { writeText("<html>$plot</html>") }.show()
}

// Compare distributional shift introduced by each code transformation
// as measured by some distance metric. https://proceedings.mlr.press/v37/kusnerb15.pdf
private fun compareDistributionalShift(txs: List<KFunction1<String, String>>) {
  TEST_DIR.allFilesRecursively().allCodeFragments().map { (c, s) ->
    txs.mapNotNull { tx ->
      val (original, transformed) = c.getContext(4).let { it to tx(it) }
      if (original == transformed) return@mapNotNull null
//      val distance = kantorovich(matrixize(original), matrixize(transformed))
      val distance = euclidDist(
        matrixize(original).average(),
        matrixize(transformed).average()
      )
      println("${tx.name}:".padEnd(20, ' ') + distance)
      tx to distance
    }
  }.flatten().groupBy { it.first }
    // Average embedding distance across code transformation
    .mapValues { (_, v) -> v.map { it.second }.average() }
}