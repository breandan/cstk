package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.*
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import org.nield.kotlinstatistics.variance
import java.io.File
import kotlin.math.*

// Does edit distance correlate with embedding distance?
fun main() {
  val data = fetchOrLoadData().let { (l, v) -> l.zip(v) }
  println("strdist,euclidist,var")
  println(compareDistanceMetrics(data, Damerau())
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

fun <T, U> cartProd(c1: Collection<T>, c2: Collection<U>): List<Pair<T, U>> =
  c1.flatMap { lhsElem -> c2.map { rhsElem -> lhsElem to rhsElem } }

fun euclidDist(f1: DoubleArray, f2: DoubleArray) =
  sqrt(f1.mapIndexed { i, f -> (f - f2[i]).pow(2) }.sum())
