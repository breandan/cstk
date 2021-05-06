package edu.mcgill.gymfs.experiments

import com.github.jelmerk.knn.SearchResult
import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharSequenceNodeFactory
import com.googlecode.concurrenttrees.solver.LCSubstringSolver
import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.indices.*

fun main() {
  val (labels, vectors) = fetchOrLoadSampleData(1000)

  val knnIndex = buildOrLoadVecIndex(rootDir = TEST_DIR)

//  val query = "private fun compareDistanceMetrics("
//  knnIndex.exactKNNSearch(vectorize(query), 10)
//    .forEach { println("${it.distance()}:${it.item().loc} / ${it.item()}\n\n") }

  val mostSimilar = labels.zip(vectors).mapIndexed { i, (l, v) ->
    Neighborhood(l, knnIndex.nearestNonEmptyNeighbors(v, 20))
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

private fun VecIndex.nearestNonEmptyNeighbors(v: DoubleArray, i: Int) =
  findNearest(v, i + 10)
    .filter { !it.item().embedding.contentEquals(v) }
    .distinctBy { it.item().toString() }.take(i)

data class Neighborhood(
  val origin: String,
  val nearestNeighbors: List<SearchResult<CodeEmbedding, Double>>,
) {
  val totalDistance by lazy { nearestNeighbors.sumOf { it.distance() } }

  val resultsSoFar by lazy {
    nearestNeighbors.mapIndexed { i, _ ->
      nearestNeighbors.subList(0, i + 1).map { it.item().toString() }
    }
  }
  val longestCommonSubstringSoFar by lazy {
    resultsSoFar.map { allResultsUpToCurrent ->
      LCSubstringSolver(DefaultCharSequenceNodeFactory())
        .apply { allResultsUpToCurrent.forEach { if(it.isNotBlank()) add(it) } }
        .longestCommonSubstring.toString()
    }
  }

  val regexSoFar by lazy {
    resultsSoFar.map { allResultsUpToCurrent ->
      synthesizeRegex(*allResultsUpToCurrent.toTypedArray())
    }
  }

  val prettyPrinted by lazy {
    nearestNeighbors.zip(longestCommonSubstringSoFar).map { (result, substring) ->
      if(substring.length < 2) result.item().toString()
      else result.item().toString()
        .replace(substring, "《$substring》") //+ "// Regex: $regexSoFar"
    }
  }
}