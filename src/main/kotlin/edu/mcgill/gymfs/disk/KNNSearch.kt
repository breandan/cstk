package edu.mcgill.gymfs.disk

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.github.jelmerk.knn.DistanceFunctions.FLOAT_INNER_PRODUCT
import com.github.jelmerk.knn.Item
import com.github.jelmerk.knn.hnsw.HnswIndex
import info.debatty.java.stringsimilarity.MetricLCS
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import java.io.File
import java.nio.file.*
import kotlin.time.*

/**
 * Retrieves the K-nearest fragment embeddings and
 * finds the longest common substring by embedding
 * distance.
 */

class KNNSearch: CliktCommand() {
  val path by option("--path", help = "Root directory")
    .default(ROOT_DIR.toAbsolutePath().toString())

  val query by option("--query", help = "Code fragment").default("const val MAX_GPUS = 1")

  val index by option("--index", help = "Prebuilt index file").default("knnindex.idx")

  val knnIndex: VecIndex by lazy {
    if (File(index).exists()) deserialize(File(index)) as VecIndex
    else rebuildIndex()
  }

  fun search(query: String): List<String> =
    knnIndex.findNearest(vectorize(query), 1000)
      .map { it.item().loc.getContext(0) }

  fun List<String>.sortedByDist(query: String, metric: MetricStringDistance) =
    sortedBy { metric.distance(it, query) }

  @OptIn(ExperimentalTime::class)
  override fun run() {
    println("\nSearching index of size ${knnIndex.size()} for [?]=[$query]…\n")
    val nearestNeighbors = search(query)
    val (metric, metricName) = MetricLCS().let { it to it::class.simpleName }
    val mostSimilarHits = nearestNeighbors.sortedByDist(query, metric)

    println("\nFetched nearest neighbors in " + measureTime {
      println("""
        |-----> Original index before reranking by $metricName
        |    |-----> Current index after reranking by $metricName
        |    |
      """.trimIndent())
      mostSimilarHits.take(10).forEachIndexed { currentIndex, s ->
        val originalIndex = nearestNeighbors.indexOf(s).toString().padStart(3)
        println("${originalIndex}->$currentIndex.) $s")
      }
    }.inMilliseconds + "ms")
  }

  @OptIn(ExperimentalTime::class)
  fun rebuildIndex(): VecIndex =
    HnswIndex.newBuilder(512, FLOAT_INNER_PRODUCT, 1000000)
      .withM(100).withEf(500).withEfConstruction(500)
      .build<Location, Fragment>().also { idx ->
        println("Rebuilt index in " + measureTime {
          Path.of(path).allFilesRecursively().allCodeFragments()
            .forEach { (loc, text) -> idx.add(Fragment(loc, vectorize(text))) }
        }.inMinutes + " minutes")
      }.also { it.serialize(File(index)) }
}

fun main(args: Array<String>) = KNNSearch().main(args)

typealias VecIndex = HnswIndex<Location, FloatArray, Fragment, Float>

data class Fragment(val loc: Location, val embedding: FloatArray):
  Item<Location, FloatArray> {
  override fun id(): Location = loc

  override fun vector(): FloatArray = embedding

  override fun dimensions(): Int = embedding.size
}