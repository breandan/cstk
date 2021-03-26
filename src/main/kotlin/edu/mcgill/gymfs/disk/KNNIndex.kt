package edu.mcgill.gymfs.disk

import com.github.jelmerk.knn.DistanceFunctions.*
import com.github.jelmerk.knn.Item
import com.github.jelmerk.knn.hnsw.HnswIndex
import info.debatty.java.stringsimilarity.*
import java.io.File
import kotlin.system.measureTimeMillis
import kotlin.time.*

/**
 * Retrieves the K-nearest fragment embeddings and
 * finds the longest common substring by embedding
 * distance.
 */

@OptIn(ExperimentalTime::class)
fun main() {
  val query = "* Retrieves the K-nearest fragment embeddings and"
  println("Query: $query\n\n")
  val embedding= vectorize(query)

  println("Fetched nearest neighbors in " + measureTime {
    val nearestNeighbors = knnIndex.findNearest(embedding, 1000)
      .map { it.item().loc.getContext(0) }

    val mostSimilarNeighbors = nearestNeighbors.sortedByDist(query)
    mostSimilarNeighbors.forEachIndexed { i, s -> println("$i.) $s") }
  }.inMilliseconds + "ms")
}

fun List<String>.sortedByDist(query: String, metric: LongestCommonSubsequence= LongestCommonSubsequence()) =
  sortedBy { metric.distance(it, query) }

val indexFile = File("knnindex.idx")

val knnIndex: VecIndex =
  if (indexFile.exists()) deserialize(indexFile) as VecIndex else rebuildIndex()

typealias VecIndex = HnswIndex<Location, FloatArray, Fragment, Float>

@OptIn(ExperimentalTime::class)
fun rebuildIndex(): VecIndex =
  HnswIndex.newBuilder(512, FLOAT_INNER_PRODUCT, 1000000)
    .withM(100).withEf(500).withEfConstruction(500)
    .build<Location, Fragment>().also { idx ->
      println("Rebuilt index in " + measureTime {
        ROOT_DIR.allFilesRecursively().allCodeFragments()
          .forEach { (loc, text) -> idx.add(Fragment(loc, vectorize(text))) }
      }.inMinutes + " minutes")
    }.also { it.serialize(indexFile) }

data class Fragment(val loc: Location, val vector: FloatArray):
  Item<Location, FloatArray> {
  override fun id(): Location = loc

  override fun vector(): FloatArray = vector

  override fun dimensions(): Int = vector.size
}