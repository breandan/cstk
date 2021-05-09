package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.*
import info.debatty.java.stringsimilarity.*
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import org.nield.kotlinstatistics.variance
import kotlin.math.*

// Does edit distance correlate with embedding distance?
fun main() {
  val data = fetchOrLoadSampleData().let { (l, v) -> l.zip(v) }
  println("strdist,embdist,variance")
  println(compareDistanceMetrics(data, MetricLCS())
    .joinToString("\n") { "" + it.first + "," + it.second + "," + it.third })
}

private fun compareDistanceMetrics(
  data: List<Pair<String, DoubleArray>>,
  stringMetric: MetricStringDistance = Levenshtein(),
) = cartProd(data, data)
  .map { (s1, s2) ->
    (stringMetric.distance(s1.first, s2.first) * 100).toInt() to
      euclidDist(s1.second, s2.second)
  }.groupBy(Pair<Int, Double>::first)
  .mapValues { (_, value) ->
    value.map { (_, euclid) -> euclid }
      .let { it.average() to it.variance() }
  }.toSortedMap()
  .map { (key, value) -> Triple(key, value.first, value.second) }
