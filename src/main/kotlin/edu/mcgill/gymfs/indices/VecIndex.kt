package edu.mcgill.gymfs.indices

import com.github.jelmerk.knn.*
import com.github.jelmerk.knn.DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE
import com.github.jelmerk.knn.hnsw.HnswIndex
import edu.mcgill.gymfs.disk.*
import java.io.File
import java.net.URI
import kotlin.time.*


fun buildOrLoadVecIndex(
  index: File = File(DEFAULT_KNNINDEX_FILENAME),
  rootDir: URI = TEST_DIR
): VecIndex =
  if (!index.exists()) rebuildVecIndex(index, rootDir)
  else index.also { println("Loading index from ${it.absolutePath}") }
    .deserializeFrom()

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
@OptIn(ExperimentalTime::class)
fun rebuildVecIndex(indexFile: File, origin: URI): VecIndex =
  HnswIndex.newBuilder(
    BERT_EMBEDDING_SIZE,
    DOUBLE_EUCLIDEAN_DISTANCE, 1000000
  ).withM(100).withEf(500).withEfConstruction(500)
    .build<Location, CodeEmbedding>().also { idx ->
      println("Rebuilding vector index...")
      measureTimedValue {
        indexURI(origin) { line, loc ->
          try {
            idx.add(CodeEmbedding(loc, vectorize(line)))
          } catch (exception: Exception) {}
        }
      }.let { println("Rebuilt vector index in ${it.duration.inWholeMinutes} minutes") }
    }.also { it.serializeTo(indexFile) }

typealias VecIndex = HnswIndex<Location, DoubleArray, CodeEmbedding, Double>

val VecIndex.defaultFilename: String by lazy { "vector.idx" }

// Expensive, need to compute pairwise distances with all items in the index
fun VecIndex.exactKNNSearch(vq: DoubleArray, nearestNeighbors: Int) =
  asExactIndex().findNearest(vq, nearestNeighbors)

data class CodeEmbedding(val loc: Location, val embedding: DoubleArray):
  Item<Location, DoubleArray> {
  override fun id(): Location = loc

  override fun vector(): DoubleArray = embedding

  override fun dimensions(): Int = embedding.size

  override fun toString() = loc.getContext(0)
}

fun main() {
  buildOrLoadVecIndex()
}