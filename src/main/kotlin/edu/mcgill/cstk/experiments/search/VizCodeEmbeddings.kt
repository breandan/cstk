package edu.mcgill.cstk.experiments.search

import ai.hypergraph.kaliningraph.visualization.show
import org.jetbrains.letsPlot.*
import org.jetbrains.letsPlot.awt.plot.PlotSvgExport
import org.jetbrains.letsPlot.commons.geometry.DoubleVector
import org.jetbrains.letsPlot.themes.theme
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.intern.toSpec
import org.jetbrains.letsPlot.label.ggtitle
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
  tsne(this, d = outputDims, perplexity = perplexity, maxIter = iterations).coordinates()

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