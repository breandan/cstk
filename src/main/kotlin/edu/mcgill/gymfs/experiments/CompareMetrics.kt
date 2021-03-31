package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.*
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import org.nield.kotlinstatistics.variance
import java.io.File
import kotlin.math.*

// Does edit distance correlate with embedding distance?
fun main() {
  val data = fetchOrLoadData()
  println("levdist,euclidist,var")
  println(compareDistanceMetrics(data, Damerau())
    .joinToString("\n") { "" + it.first + "," + it.second + "," + it.third })
}

private fun fetchOrLoadData() = (File("sample1000.data")
  .let { if (it.exists()) it else null }
  ?.deserialize() as? List<Pair<String, FloatArray>>
  ?: ROOT_DIR
    .allFilesRecursively()
    .allCodeFragments()
    .shuffled()
    .take(1000)
    .map { it.second to vectorize(it.second) }
    .also { it.serialize(File("sample1000.data")) })

private fun compareDistanceMetrics(
  data: List<Pair<String, FloatArray>>,
  stringMetric: MetricStringDistance = Levenshtein(),
) =
  cartProd(data, data)
    .map { (s1, s2) ->
      (stringMetric.distance(s1.first, s2.first)*100).toInt() to
        euclidDist(s1.second, s2.second)
    }.groupBy(Pair<Int, Float>::first)
    .mapValues {
      it.value.map { it.second.toDouble() }
        .let { it.average() to it.variance() }
    }
    .toSortedMap().entries
    .map { (key, value) -> Triple(key,value.first, value.second) }

fun <T, U> cartProd(c1: Collection<T>, c2: Collection<U>): List<Pair<T, U>> {
  return c1.flatMap { lhsElem -> c2.map { rhsElem -> lhsElem to rhsElem } }
}

fun euclidDist(f1: FloatArray, f2: FloatArray) =
  sqrt(f1.mapIndexed { i, f -> (f - f2[i]).pow(2f) }.sum())
