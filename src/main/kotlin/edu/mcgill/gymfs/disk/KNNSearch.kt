package edu.mcgill.gymfs.disk

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.github.jelmerk.knn.DistanceFunctions.FLOAT_INNER_PRODUCT
import com.github.jelmerk.knn.Item
import com.github.jelmerk.knn.hnsw.HnswIndex
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

  val query by option("--query", help = "Code fragment").default("const val MAX_GPUS = 1")

  val index by option("--index", help = "Prebuilt index file").default("knnindex.idx")

  val knnIndex: VecIndex by lazy {
    if (File(index).exists()) deserialize(File(index)) as VecIndex
    else rebuildIndex()
  }

  // Cheap index lookup using HNSW index
  fun approxKNNSearch(query: String, vq: FloatArray = vectorize(query)) =
    knnIndex.findNearest(vq, 1000).map { it.item().loc.getContext(0) }

  // Expensive, need to compute pairwise distances with all items in the index
  fun exactKNNSearch(query: String, vq: FloatArray = vectorize(query)) =
    knnIndex.items().mapIndexed { i, it ->
      if (i % 100 == 0) println("Vectorized $i out of ${knnIndex.items().size}")
      it to vectorize(it.loc.getContext(0))
    }.sortedBy { (_, vx) ->
      vq.zip(vx).map { (x, y) -> x * x - y * y }.sum().pow(0.5f)
    }.map { it.first.loc.getContext(0) }

  @OptIn(ExperimentalTime::class)
  override fun run() {
    println("\nSearching KNN index of size ${knnIndex.size()} for [?]=[$query]…\n")
    val nearestNeighbors = approxKNNSearch(query) //exactKNNSearch(query)

    // TODO: Why are the KNN results so bad?
    // Hypothesis #1: Language mismatch (Kotlin/Java)
    // Hypothesis #2: Encoding issue with BERT vectors (MLM/dot product/...) <--
    // Hypothesis #3: Pretraining issue / contextual misalignment

    println("\nFetched nearest neighbors in " + measureTime {
      nearestNeighbors.take(10).forEachIndexed { i, s -> println("$i.) $s") }
    }.inMilliseconds + "ms")

    val (metric, metricName) = MetricLCS().let { it to it::class.simpleName }
    val mostSimilarHits = nearestNeighbors.sortedByDist(query, metric)

    println("\nReranked nearest neighbors in " + measureTime {
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

  // Compare various distance functions
  @OptIn(ExperimentalTime::class)
  fun rebuildIndex(): VecIndex =
    HnswIndex.newBuilder(BERT_EMBEDDING_SIZE, FLOAT_INNER_PRODUCT, 1000000)
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