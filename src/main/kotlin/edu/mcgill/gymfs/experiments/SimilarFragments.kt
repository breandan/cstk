package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
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
      .take(10)
    Triple(l, knn, knn.sumByDouble { it.distance() })
  }.sortedBy { (_, _, totalDistance) -> totalDistance }

  println("Nearest nearest neighbors by cumulative similarity\n")

  mostSimilar.distinct()
    .filter { "import" !in it.first }
    .take(10)
    .forEachIndexed { i, (label, knn, totalDistance) ->
      println("$i.] $label")
      knn.forEachIndexed { j, it ->
        println("\t$i.$j.] " + it.item().toString())
      }
      println()
      println()
    }
}