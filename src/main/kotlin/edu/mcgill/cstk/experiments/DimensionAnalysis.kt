package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.math.*
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation

// Do certain dimensions correlate more strongly with string edit distance?
fun main() {
  val data = fetchOrLoadSampleData().let { (l, v) -> l.zip(v) }.take(100)
  // Compare correlation between string metrics and learned metric along individual dimensions
  println("dim,correlation")
  (0 until data.first().second.size - 1).map { i ->
    compareDistanceMetricsOnDim(data, (i..i + 1).toList())
  }.sortedBy { it.second }
    .forEach { println("${it.first.first()},${it.second}") }
}

private fun compareDistanceMetricsOnDim(
  data: List<Pair<String, DoubleArray>>,
  dims: Collection<Int>,
  stringMetric: MetricStringDistance = MetricCSNF,
): Pair<Collection<Int>, Double> = cartProd(data, data)
  .map { (s1, s2) ->
    stringMetric.distance(s1.first, s2.first) to
      euclidDist(s1.second.sliceArray(dims), s2.second.sliceArray(dims))
  }.map { (a, b) -> a to b }.unzip().let { (a, b) ->
    dims to PearsonsCorrelation().correlation(a.toDoubleArray(), b.toDoubleArray())
  }