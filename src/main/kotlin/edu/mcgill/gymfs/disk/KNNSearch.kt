package edu.mcgill.gymfs.disk

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.github.jelmerk.knn.*
import com.github.jelmerk.knn.DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE
import com.github.jelmerk.knn.DistanceFunctions.DOUBLE_INNER_PRODUCT
import com.github.jelmerk.knn.hnsw.HnswIndex
import edu.mcgill.gymfs.experiments.fetchOrLoadSampleData
import edu.mcgill.kaliningraph.show
import info.debatty.java.stringsimilarity.MetricLCS
import java.io.File
import java.nio.file.Path
import kotlin.math.pow
import kotlin.time.*

/**
 * Retrieves the K-nearest fragment embeddings and
 * finds the longest common substring by embedding
 * distance.
 */

class KNNSearch: CliktCommand() {
  val path by option("--path", help = "Root directory")
    .default(ROOT_DIR.toAbsolutePath().toString())

  val query by option(
    "--query",
    help = "Code fragment"
  ).default("const val MAX_GPUS = 1")

  val index by option(
    "--index",
    help = "Prebuilt index file"
  ).default("knnindex.idx")

  val graphs by option("--graphs", help = "Visualize graphs").default("")

  val knnIndex: VecIndex by lazy {
    if (File(index).exists()) File(index).deserialize() as VecIndex
    else rebuildIndex()
  }

  // Cheap index lookup using HNSW index
  fun approxKNNSearch(query: String, vq: DoubleArray = vectorize(query)) =
    knnIndex.findNearest(vq, 1000).map { it.item().loc.getContext(0) }

  @OptIn(ExperimentalTime::class)
  override fun run() {
    printQuery()
    graphs.toIntOrNull()?.let { generateGraphs(it) }
  }

  private fun generateGraphs(total: Int) {
    println("Regenerating $total graphs...")
    fetchOrLoadSampleData().first.take(total).forEach { query ->
        val id = query.hashCode().toString()
        edges(query)
          .toLabeledGraph()
          .also { it.A.show() }
          .apply { vertices.first { it.label == id }.occupied = true }
          .renderVKG()
          .show()
      }
  }

  @OptIn(ExperimentalTime::class)
  private fun printQuery() {
    println("\nSearching KNN index of size ${knnIndex.size()} for [?]=[$query]…\n")
    val nearestNeighbors = approxKNNSearch(query) //exactKNNSearch(query)

    println("\nFetched nearest neighbors in " + measureTime {
      nearestNeighbors.take(10).forEachIndexed { i, s -> println("$i.) $s") }
    }.inMilliseconds + "ms")

    val (metric, metricName) = MetricLCS().let { it to it::class.simpleName }
    val mostSimilarHits = nearestNeighbors.sortedByDist(query, metric)

    println("\nReranked nearest neighbors in " + measureTime {
      println(
        """
        
        |-----> Original index before reranking by $metricName
        |    |-----> Current index after reranking by $metricName
        |    |
      """.trimIndent()
      )
      mostSimilarHits.take(10).forEachIndexed { currentIndex, s ->
        val originalIndex = nearestNeighbors.indexOf(s).toString().padStart(3)
        println("${originalIndex}->$currentIndex.) $s")
      }
    }.inMilliseconds + "ms")
  }

  tailrec fun edges(
    seed: String? = null,
    queries: List<String> = if(seed == null) emptyList() else listOf(seed),
    depth: Int = 10,
    width: Int = 5,
    edges: List<Pair<String, String>> = emptyList(),
  ): List<Pair<String, String>> =
    if (queries.isEmpty() || depth == 0) edges
    else {
      val query = seed ?: queries.first()
      val nearestResults = knnIndex.findNearest(vectorize(query), 100)
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
  fun rebuildIndex(): VecIndex =
    HnswIndex.newBuilder(BERT_EMBEDDING_SIZE, DOUBLE_EUCLIDEAN_DISTANCE, 1000000)
      .withM(100).withEf(500).withEfConstruction(500)
      .build<Location, Fragment>().also { idx ->
        println("Rebuilt index in " + measureTime {
          Path.of(path).allFilesRecursively().allCodeFragments()
            .forEach { (loc, text) -> idx.add(Fragment(loc, vectorize(text))) }
        }.inMinutes + " minutes")
      }.also { it.serialize(File(index)) }
}

fun main(args: Array<String>) = KNNSearch().main(args)

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