package edu.mcgill.cstk.experiments

import ai.hypergraph.kaliningraph.visualization.show
import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.datalore.plot.PlotSvgExport
import jetbrains.letsPlot.*
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import smile.manifold.tsne
import java.io.File

fun main() {
  val (labels, vectors) = fetchOrLoadSampleData()

  val d2vecs = vectors.reduceDim()

  labels.forEachIndexed { i, l ->
    println("${l.length},${d2vecs[i][0]},${d2vecs[i][1]}")
  }

  val plot = plotTsneEmbeddingsWithLabels(d2vecs, labels.map { it.length.toString() } )

  File.createTempFile("clusters", ".html")
    .apply { writeText("<html>$plot</html>") }.show()
}

fun Array<DoubleArray>.reduceDim(
  outputDims: Int = 2,
  perplexity: Double = 10.0,
  iterations: Int = 99999
): Array<out DoubleArray> =
  tsne(this, d = outputDims, perplexity = perplexity, iterations = iterations).coordinates

fun plotTsneEmbeddingsWithLabels(
  embeddings: Array<out DoubleArray>,
  labels: List<String>
): String {
  val data = mapOf(
    "labels" to labels,
    "x" to embeddings.map { it[0] },
    "y" to embeddings.map { it[1] }
  )
  val plot = letsPlot(data) { x = "x"; y = "y"; color = "labels" } +
    ggsize(300, 250) + geomPoint(size = 6) +
    ggtitle("Lines by Structural Similarity") +
    theme(axisLine = "blank", axisTitle =  "blank", axisTicks = "blank", axisText = "blank")
  return PlotSvgExport.buildSvgImageFromRawSpecs(
    plotSpec = plot.toSpec(), plotSize = DoubleVector(1000.0, 500.0)
  )
}