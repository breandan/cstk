package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import java.io.File
import kotlin.math.*

fun main() {
  val (labels, vectors) = fetchOrLoadSampleData()

  val knnIndex = File("knnindex.idx").deserialize() as VecIndex

//    knnIndex.exactKNNSearch(vectorize("import ai.djl.training.loss.Loss"), 10)
//      .forEach { println("" + it.distance() + ":" + it.item().loc.getContext(5) + "\n\n") }

  val topK = 10
  val mostSimilar =
    labels.zip(vectors)
    .map { (l, v) ->
    val knn = knnIndex.exactKNNSearch(v, topK)
    Triple(l, knn, knn.sumByDouble { it.distance() })
  }.sortedBy { (_, _, totalDistance) -> totalDistance }

  mostSimilar.take(10).forEach { (label, knn, totalDistance) ->
    println("$totalDistance: $label")
    knn.forEach { println(it.item().toString()) }
    println()
  }
}