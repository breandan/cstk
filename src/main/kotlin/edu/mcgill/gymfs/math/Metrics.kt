package edu.mcgill.gymfs.math

import com.github.jelmerk.knn.DistanceFunction
import com.google.ortools.Loader
import com.google.ortools.linearsolver.MPSolver
import kotlin.math.*
import kotlin.random.Random
import kotlin.time.*

fun <T, U> cartProd(c1: Iterable<T>, c2: Iterable<U>): List<Pair<T, U>> =
  c1.flatMap { lhsElem -> c2.map { rhsElem -> lhsElem to rhsElem } }

fun euclidDist(f1: DoubleArray, f2: DoubleArray) =
  sqrt(f1.zip(f2) { a, b -> (a - b).pow(2) }.sum())

// Computes elementwise means on a list of lists
fun Array<DoubleArray>.average(): DoubleArray =
  fold(DoubleArray(first().size)) { a, b ->
    a.zip(b).map { (i, j) -> i + j }.toDoubleArray()
  }.map { it / size }.toDoubleArray()

fun DoubleArray.normalize() = sum().let { sum -> map { it / sum }.toDoubleArray() }

//val t = Loader.loadNativeLibraries()

// https://github.com/stephenhky/PyWMD/blob/master/WordMoverDistanceDemo.ipynb
// http://proceedings.mlr.press/v37/kusnerb15.pdf#page=3
// https://www.youtube.com/watch?v=CDiol4LG2Ao
fun kantorovich(p1: Array<DoubleArray>, p2: Array<DoubleArray>): Double =
  if (p1.size == 1 && p2.size == 1) euclidDist(p1.first(), p2.first())
// https://developers.google.com/optimization/introduction/java#complete-program
// https://developers.google.com/optimization/lp/glop#entire_program
  else MPSolver.createSolver("GLOP").run {
    val (vars, dists) = cartProd(p1.indices, p2.indices)
      .mapIndexed { i, (j, k) ->
        makeNumVar(0.0, 1.0, "x$i") to euclidDist(p1[j], p2[k])
      }.unzip()

    val obj = objective()
    for (i in vars.indices) obj.setCoefficient(vars[i], dists[i])
    obj.setMinimization()

    // Ensure each row sums to 1
    makeConstraint(1.0, 1.0).apply {
      for (j in p1.indices)
        cartProd(p1.indices, p2.indices)
          .mapIndexedNotNull { i, (jj, _) -> if (j == jj) vars[i] else null }
          .forEach { setCoefficient(it, 1.0) }
    }

    // Ensure each col sums to 1
    makeConstraint(1.0, 1.0).apply {
      for (k in p2.indices)
        cartProd(p1.indices, p2.indices)
          .mapIndexedNotNull { i, (_, kk) -> if (k == kk) vars[i] else null }
          .forEach { setCoefficient(it, 1.0) }
    }

    // Ensure nonnegative transport
    vars.forEach {
      makeConstraint(0.0, MPSolver.infinity())
        .apply { setCoefficient(it, 1.0) }
    }

    solve()

    obj.value()
  }

object EMD: DistanceFunction<DoubleArray, Double> {
  override fun distance(u: DoubleArray, v: DoubleArray) =
    kantorovich(arrayOf(u), arrayOf(v))
}

@OptIn(ExperimentalTime::class)
fun main() {
  val (a, b) = Pair(randomMatrix(400, 768), randomMatrix(400, 768))
  println(measureTime { println(kantorovich(a, b)) })
}

fun randomMatrix(n: Int, m: Int) =
  Array(n) { DoubleArray(m) { Random.nextDouble() } }