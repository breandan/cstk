package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.*
import edu.mcgill.markovian.pmap
import info.debatty.java.stringsimilarity.*
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import kotlin.math.*

// Do certain dimensions correlate more strongly with string edit distance?
fun main() {
  val data = fetchOrLoadSampleData().let { (l, v) -> l.zip(v) }.take(100)
  //compare correlation between string metrics and learned metric along dimensions
  println("dim, correlation")
  (0 until data.first().second.size - 1).pmap { i ->
    println(i)
    compareDistanceMetricsOnDim(data, (i..i + 1).toList())
  }.sortedBy { it.second }
    .forEach { println("${it.first.first()} ${it.second}") }
}

private fun compareDistanceMetricsOnDim(
  data: List<Pair<String, DoubleArray>>,
  dims: Collection<Int>,
  stringMetric: MetricStringDistance = MetricCSNF,
): Pair<Collection<Int>, Double> = cartProd(data, data)
  .map { (s1, s2) ->
    (stringMetric.distance(s1.first, s2.first) * 100).toInt() to
      euclidDist(s1.second.sliceArray(dims), s2.second)
  }.groupBy(Pair<Int, Double>::first)
  .mapValues { it.value.map { (_, euclid) -> euclid }.variance() }
  .values.sum().let { dims to it }
