package edu.mcgill.cstk.experiments

import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import info.debatty.java.stringsimilarity.*
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import kotlin.math.*

// Does edit distance correlate with embedding distance?
fun main() {
  val data = fetchOrLoadSampleData().let { (l, v) -> l.zip(v) }
  println("strdist,embdist,variance")
  println(compareDistanceMetrics(data.toSet(), MetricLCS())
    .joinToString("\n") { "" + it.π1 + "," + it.π2 + "," + it.π3 })
}

private fun compareDistanceMetrics(
  data: Set<Pair<String, DoubleArray>>,
  stringMetric: MetricStringDistance = Levenshtein(),
) = (data * data)
  .map { (s1, s2) ->
    (stringMetric.distance(s1.first, s2.first) * 100).toInt() to
      euclidDist(s1.second, s2.second)
  }.groupBy(Pair<Int, Double>::first)
  .mapValues { (_, value) ->
    value.map { (_, euclid) -> euclid }
      .let { it.average() to it.variance() }
  }.toSortedMap()
  .map { (key, value) -> Π(key, value.first, value.second) }
