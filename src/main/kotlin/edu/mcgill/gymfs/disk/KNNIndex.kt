package edu.mcgill.gymfs.disk

import com.github.jelmerk.knn.DistanceFunctions.*
import com.github.jelmerk.knn.Item
import com.github.jelmerk.knn.hnsw.HnswIndex
import java.io.File

/**
 * Retrieves the K-nearest fragment embeddings and
 * finds the longest common substring by embedding
 * distance.
 */

fun main() {
  val query = vectorize("* Retrieves the K-nearest fragment embeddings and")
  println(query.drop(500).joinToString("/"))
  val nearestNeighbors = knnIndex.findNearest(query, 100)
  println(nearestNeighbors.joinToString("\n\n") {
    it.item().loc.getContext(0)
  })
}

val indexFile = File("knnindex.idx")

val knnIndex: VecIndex =
  if (indexFile.exists()) deserialize(indexFile) as VecIndex else rebuildIndex()

typealias VecIndex = HnswIndex<Location, FloatArray, Fragment, Float>

fun rebuildIndex(): VecIndex =
  HnswIndex.newBuilder(512, FLOAT_INNER_PRODUCT, 1000000)
    .withM(50).withEf(500).withEfConstruction(500)
    .build<Location, Fragment>().also { idx ->
      ROOT_DIR.allFilesRecursively().allCodeFragments()
        .forEach { (loc, text) -> idx.add(Fragment(loc, vectorize(text))) }
    }.also { it.serialize(indexFile) }

data class Fragment(val loc: Location, val vector: FloatArray):
  Item<Location, FloatArray> {
  override fun id(): Location = loc

  override fun vector(): FloatArray = vector

  override fun dimensions(): Int = vector.size
}