package edu.mcgill.gymfs.indices

import com.github.jelmerk.knn.Item
import com.github.jelmerk.knn.hnsw.HnswIndex
import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.EMD
import edu.mcgill.gymfs.nlp.vectorize
import java.io.File
import java.net.URI
import kotlin.time.*


fun buildOrLoadVecIndex(
  indexFile: File = File(DEFAULT_KNNINDEX_FILENAME),
  rootDir: URI = DATA_DIR
): VecIndex =
  if (!indexFile.exists()) rebuildVecIndex(indexFile, rootDir)
  else indexFile.also { println("Loading index from ${it.absolutePath}") }
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
  HnswIndex.newBuilder(BERT_EMBEDDING_SIZE, EMD, 1000000)
    .withM(100).withEf(500).withEfConstruction(500)
    .build<Concordance, CodeEmbedding>().also { idx ->
      println("Rebuilding vector index...")
      measureTimedValue {
        // TODO: Does parallelization really help on single-GPU machine?
        origin.allFilesRecursively().toList().parallelStream().forEach { src ->
//          var last = ""
          indexURI(src) { line, loc ->
            try {
//              if (loc.uri.suffix() != last)
//                last = loc.uri.suffix().also { println(loc.fileSummary()) }
              idx.add(CodeEmbedding(loc, vectorize(line)))
            } catch (exception: Exception) {
            }
          }
        }
      }.let { println("Rebuilt vector index in ${it.duration.inWholeMinutes} minutes") }
    }.also { it.serializeTo(indexFile) }

typealias VecIndex = HnswIndex<Concordance, DoubleArray, CodeEmbedding, Double>

val VecIndex.defaultFilename: String by lazy { "vector.idx" }

// Expensive, need to compute pairwise distances with all items in the index
fun VecIndex.exactKNNSearch(vq: DoubleArray, nearestNeighbors: Int) =
  asExactIndex().findNearest(vq, nearestNeighbors)

data class CodeEmbedding constructor(val loc: Concordance, val embedding: DoubleArray):
  Item<Concordance, DoubleArray> {
  override fun id(): Concordance = loc

  override fun vector(): DoubleArray = embedding

  override fun dimensions(): Int = embedding.size

  override fun toString() = loc.getContext(0)
}

fun main() {
  buildOrLoadVecIndex()
}