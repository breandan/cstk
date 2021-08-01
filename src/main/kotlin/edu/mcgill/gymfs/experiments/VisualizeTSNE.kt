package edu.mcgill.gymfs.experiments

import com.jujutsu.tsne.TSne
import com.jujutsu.tsne.barneshut.ParallelBHTsne
import com.jujutsu.utils.TSneUtils
import edu.mcgill.kaliningraph.show
import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.datalore.plot.*
import jetbrains.letsPlot.*
import jetbrains.letsPlot.geom.*
import jetbrains.letsPlot.intern.*
import jetbrains.letsPlot.label.ggtitle
import java.io.File

fun main() {
  val (labels, vectors) = fetchOrLoadSampleData()

  val d2vecs = vectors.reduceDim()

  labels.forEachIndexed { i, l ->
    println("${l.length},${d2vecs[i][0]},${d2vecs[i][1]}")
  }

  val plot = plot(d2vecs, labels.map { it.length.toString() } )

  File.createTempFile("clusters", ".html")
    .apply { writeText("<html>$plot</html>") }.show()
}

private fun Array<DoubleArray>.reduceDim(
  outputDims: Int = 2,
  perplexity: Double = 10.0,
  tSne: TSne = ParallelBHTsne()
): Array<out DoubleArray> =
  tSne.tsne(TSneUtils.buildConfig(this, outputDims, size - 1, perplexity, 99999))

private fun plot(
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
    theme().axisLineBlank().axisTitleBlank()
      .axisTicksBlank().axisTextBlank()
  return PlotSvgExport.buildSvgImageFromRawSpecs(
    plotSpec = plot.toSpec(), plotSize = DoubleVector(1000.0, 500.0)
  )
}