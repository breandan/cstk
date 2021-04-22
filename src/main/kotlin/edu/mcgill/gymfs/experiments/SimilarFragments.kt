package edu.mcgill.gymfs.experiments

import com.github.jelmerk.knn.SearchResult
import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharSequenceNodeFactory
import com.googlecode.concurrenttrees.solver.LCSubstringSolver
import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.indices.*
import java.io.File

fun main() {
  val (labels, vectors) = fetchOrLoadSampleData()

  val knnIndex = File("knnindex.idx").deserialize() as VecIndex

//  val query = "private fun compareDistanceMetrics("
//  knnIndex.exactKNNSearch(vectorize(query), 10)
//    .forEach { println("${it.distance()}:${it.item().loc} / ${it.item()}\n\n") }

  val topK = 100
  val mostSimilar = labels.zip(vectors).map { (l, v) ->
    val knn = knnIndex.findNearest(v, topK)
      .filter { it.item().toString() != l }
      .distinctBy { it.item().toString() }
      .take(20)
    Neighborhood(l, knn)
  }.sortedBy { it.totalDistance }

  println("Nearest nearest neighbors by cumulative similarity")
  println("Angle brackets enclose longest common substring up to current result\n")

  mostSimilar.distinct()
    .filter { "import" !in it.origin }
    .take(20)
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
  val nearestNeighbors: List<SearchResult<Fragment, Double>>,
) {
  val totalDistance by lazy { nearestNeighbors.sumByDouble { it.distance() } }

  val longestCommonSubstringSoFar by lazy {
    nearestNeighbors.mapIndexed { i, _ ->
      nearestNeighbors.subList(0, i + 1).map { it.item().toString() }
    }.map { allResultsUpToCurrent ->
      if (allResultsUpToCurrent.size < 2) ""
      else LCSubstringSolver(DefaultCharSequenceNodeFactory())
        .apply { allResultsUpToCurrent.forEach { add(it) } }
        .longestCommonSubstring.toString()
    }
  }

  val prettyPrinted by lazy {
    nearestNeighbors.zip(longestCommonSubstringSoFar).map { (result, substring) ->
      if(substring.length < 2) result.item().toString()
      else result.item().toString().replace(substring, "《$substring》")
    }
  }
}