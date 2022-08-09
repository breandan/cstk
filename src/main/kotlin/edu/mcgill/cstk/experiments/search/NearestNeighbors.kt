package edu.mcgill.cstk.experiments.search

import com.github.jelmerk.knn.SearchResult
import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharSequenceNodeFactory
import com.googlecode.concurrenttrees.solver.LCSubstringSolver
import edu.mcgill.cstk.disk.TEST_DIR
import edu.mcgill.cstk.disk.indices.*

fun main() {
  val (labels, vectors) = fetchOrLoadSampleData(1000)

  val knnIndex = buildOrLoadVecIndex(rootDir = TEST_DIR)

//  val query = "private fun compareDistanceMetrics("
//  knnIndex.exactKNNSearch(vectorize(query), 10)
//    .forEach { println("${it.distance()}:${it.item().loc} / ${it.item()}\n\n") }

  val mostSimilar = labels.zip(vectors).mapIndexed { i, (l, v) ->
    Neighborhood(l, v, knnIndex.knn(v, 20))
  }.sortedBy { it.totalDistance }

  println("Nearest nearest neighbors by cumulative similarity")
  println("Angle brackets enclose longest common substring up to current result\n")

  mostSimilar.distinct()
    .filter { "import" !in it.origin && "package" !in it.origin }.take(20)
    .forEachIndexed { i, neighborhood ->
      println("$i.] ${neighborhood.origin}")
      neighborhood.prettyPrinted.forEachIndexed { j, it ->
        println("\t$i.$j] $it")
      }
      println()
      println()
    }
}

data class Neighborhood(
  val origin: String,
  val vector: DoubleArray,
  val nearestNeighbors: List<SearchResult<CodeEmbedding, Double>>,
) {
  val totalDistance by lazy { nearestNeighbors.sumOf { it.distance() } }

  val resultsSoFar by lazy {
    List(nearestNeighbors.size) { i ->
      nearestNeighbors.subList(0, i + 1).map { it.item().toString() }
    }
  }

  val longestCommonSubstringSoFar by lazy {
    resultsSoFar.map { allResultsUpToCurrent ->
      LCSubstringSolver(DefaultCharSequenceNodeFactory())
        .apply { allResultsUpToCurrent.forEach { if (it.isNotBlank()) add(it) } }
        .longestCommonSubstring.toString()
    }
  }

  val prettyPrinted by lazy {
    nearestNeighbors.zip(longestCommonSubstringSoFar)
      .map { (result, substring) ->
        if (substring.length < 2) result.item().toString()
        else result.item().toString().replace(substring, "《$substring》")
      }
  }
}