package edu.mcgill.gymfs.indices

import com.github.jelmerk.knn.*
import com.github.jelmerk.knn.DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE
import com.github.jelmerk.knn.hnsw.HnswIndex
import edu.mcgill.gymfs.disk.*
import java.io.File
import java.nio.file.Path
import kotlin.io.path.extension
import kotlin.time.*


fun buildOrLoadVecIndex(index: File, rootDir: Path): VecIndex =
  if (!index.exists()) rebuildIndex(index, rootDir).also { it.serialize(index) }
  else index.also { println("Loading index from ${it.absolutePath}") }
    .deserialize() as VecIndex

tailrec fun VecIndex.edges(
  seed: String? = null,
  queries: List<String> = if (seed == null) emptyList() else listOf(seed),
  depth: Int = 10,
  width: Int = 5,
  edges: List<Pair<String, String>> = emptyList(),
): List<Pair<String, String>> =
  if (queries.isEmpty() || depth == 0) edges
  else {
    val query = seed ?: queries.first()
    val nearestResults = findNearest(vectorize(query), 100)
      .map { it.item().loc.getContext(0) }
      .filter { it.isNotEmpty() && it != query }
      .take(width)

    val newEdges = nearestResults.map { query to it }

    edges(
      null,
      queries.drop(1) + nearestResults,
      depth - 1,
      width,
      edges + newEdges
    )
  }

// Compare various distance functions
@OptIn(ExperimentalTime::class, kotlin.io.path.ExperimentalPathApi::class)
fun rebuildIndex(index: File, path: Path): VecIndex =
  HnswIndex.newBuilder(
    BERT_EMBEDDING_SIZE,
    DOUBLE_EUCLIDEAN_DISTANCE, 1000000
  ).withM(100).withEf(500).withEfConstruction(500)
    .build<Location, Fragment>().also { idx ->
      println("Rebuilding index...")
      println("Rebuilt index in " + measureTime {
        path.allFilesRecursively().asSequence()
          .filter { it.extension == FILE_EXT }
          .allCodeFragments()
          .forEach { (loc, text) -> idx.add(Fragment(loc, vectorize(text))) }
      }.inMinutes + " minutes")
    }.also { it.serialize(index) }

typealias VecIndex = HnswIndex<Location, DoubleArray, Fragment, Double>

// Expensive, need to compute pairwise distances with all items in the index
fun VecIndex.exactKNNSearch(vq: DoubleArray, nearestNeighbors: Int) =
  asExactIndex().findNearest(vq, nearestNeighbors)

data class Fragment(val loc: Location, val embedding: DoubleArray):
  Item<Location, DoubleArray> {
  override fun id(): Location = loc

  override fun vector(): DoubleArray = embedding

  override fun dimensions(): Int = embedding.size

  override fun toString() = loc.getContext(0)
}