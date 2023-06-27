package edu.mcgill.cstk.experiments.rewriting

import ai.hypergraph.kaliningraph.types.cc
import ai.hypergraph.kaliningraph.visualization.show
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.search.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.rewriting.*
import edu.mcgill.cstk.utils.*
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

// Tries to identify the dimensions most influenced by each code transformation
private fun analyzeDimensionalShift(tx: KFunction1<String, String>) =
    DATA_DIR.allFilesRecursively().allCodeFragments().take(100).mapIndexed { i, (c, s) ->
      val (original, transformed) = c.getContext(4).let { it cc tx(it) }
      matrixize(original).mean().zip(matrixize(transformed).mean())
        .map { (a, b) -> (a - b).absoluteValue }.toDoubleArray()
    }.toList().toTypedArray().mean().normalize()

private fun compareTsneEmbeddings(tx: KFunction1<String, String>) {
  val (vecs, labels) =
    DATA_DIR.allFilesRecursively().allCodeFragments().take(100).mapIndexed { i, (c, s) ->
      val (original, transformed) = c.getContext(4).let { it cc tx(it) }
      listOf(matrixize(original).mean() to "o", matrixize(transformed).mean() to "t")
    }.flatten().unzip()
  val d2vecs = vecs.toTypedArray().reduceDim()
  val plot = plotTsneEmbeddingsWithLabels(d2vecs, labels)
  File.createTempFile("compare_${tx.name}", ".html")
    .apply { writeText("<html>$plot</html>") }.show()
}

// Compare distributional shift introduced by each code transformation
// as measured by some distance metric. https://proceedings.mlr.press/v37/kusnerb15.pdf
private fun compareDistributionalShift(txs: List<KFunction1<String, String>>) {
  DATA_DIR.allFilesRecursively().allCodeFragments().map { (c, s) ->
    txs.mapNotNull { tx ->
      val (original, transformed) = c.getContext(4).let { it cc tx(it) }
      if (original == transformed) return@mapNotNull null
//      val distance = kantorovich(matrixize(original), matrixize(transformed))
      val distance = euclidDist(
        matrixize(original).mean(),
        matrixize(transformed).mean()
      )
      println("${tx.name}:".padEnd(20, ' ') + distance)
      tx to distance
    }
  }.flatten().groupBy { it.first }
    // Average embedding distance across code transformation
    .mapValues { (_, v) -> v.map { it.second }.average() }
}