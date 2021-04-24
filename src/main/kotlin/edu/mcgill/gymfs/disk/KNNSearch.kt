package edu.mcgill.gymfs.disk

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import edu.mcgill.gymfs.experiments.fetchOrLoadSampleData
import edu.mcgill.gymfs.indices.*
import edu.mcgill.kaliningraph.show
import info.debatty.java.stringsimilarity.MetricLCS
import java.io.File
import java.nio.file.Path
import kotlin.time.*

/**
 * Retrieves the K-nearest fragment embeddings and
 * finds the longest common substring by embedding
 * distance.
 */

class KNNSearch: CliktCommand() {
  val path by option("--path", help = "Root directory")
    .default(TEST_DIR.toAbsolutePath().toString())

  val query by option(
    "--query",
    help = "Code fragment"
  ).default("const val MAX_GPUS = 1")

  val index by option(
    "--index",
    help = "Prebuilt index file"
  ).default("vector.idx")

  val graphs by option("--graphs", help = "Visualize graphs").default("")

  val knnIndex: VecIndex by lazy { buildOrLoadVecIndex(File(index), Path.of(path)) }

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
          knnIndex.edges(query)
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

}

fun main(args: Array<String>) = KNNSearch().main(args)