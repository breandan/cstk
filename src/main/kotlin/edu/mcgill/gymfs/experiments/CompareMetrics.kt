package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.Levenshtein
import org.nield.kotlinstatistics.variance
import kotlin.math.*

// Does edit distance correlate with embedding distance?
fun main() {
  val key = "const val MAX_GPUS = 1"
  val data = ROOT_DIR
    .allFilesRecursively()
    .allCodeFragments()
    .shuffled()
    .take(1000)
    .map { it.second to vectorize(it.second) }
//    .groupBy { Levenshtein().distance(key, it.second) }
//    .keys.sorted().forEach { println(it) }

  cartProd(data, data)
  .map { (s1, s2) ->
    Levenshtein().distance(s1.first, s2.first).toInt() to
      euclidDist(s1.second, s2.second)
  }.groupBy(Pair<Int, Float>::first)
    .mapValues { it.value.map { it.second.toDouble() }.let { it.average() to it.variance() } }
    .toSortedMap().entries
    .forEach { (key, value) -> println("$key,${value.first}, ${value.second}") }
}

fun <T, U> cartProd(c1: Collection<T>, c2: Collection<U>): List<Pair<T, U>> {
  return c1.flatMap { lhsElem -> c2.map { rhsElem -> lhsElem to rhsElem } }
}

fun euclidDist(f1: FloatArray, f2: FloatArray) =
  sqrt(f1.mapIndexed { i, f -> (f - f2[i]).pow(2f) }.sum())
