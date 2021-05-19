package edu.mcgill.gymfs.experiments

import de.learnlib.algorithms.rpni.BlueFringeRPNIDFA
import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.indices.*
import edu.mcgill.gymfs.math.euclidDist
import net.automatalib.visualization.Visualization
import net.automatalib.words.*
import net.automatalib.words.impl.Alphabets
import kotlin.time.*

// TODO: DFA/RegEx or BoW query?


@OptIn(ExperimentalTime::class)
fun main() {
  val (strings, vectors) = fetchOrLoadSampleData(1000)
  val vecMap = strings.zip(vectors).toMap()

  val knnIndex = buildOrLoadVecIndex(rootDir = TEST_DIR)

  val (precisions, recalls) =
    vecMap.entries.take(100)
    .map { (s, v) -> calculuatePrecisionAndRecall(s, v, strings, knnIndex, vecMap) }
    .unzip()

  println()
  println("Mean precision: ${precisions.average()}")
  println("Mean recall:    ${recalls.average()}")
}

private fun calculuatePrecisionAndRecall(
  query: String,
  vector: DoubleArray,
  strings: List<String>,
  knnIndex: VecIndex,
  vecMap: Map<String, DoubleArray>
): Pair<Double, Double> {
  val neighbors = knnIndex.knn(vector, 100000)
  val neighborhood = Neighborhood(query, vector, neighbors)
  println("\nQuery:\n======\n${neighborhood.origin}")

  val numNearestNeighbors = 30
  val numFurthestNeighbors = 1000
  val nearestNeighbors = neighborhood.nearestNeighbors.take(numNearestNeighbors)
  val furthestNeighbors = neighborhood.nearestNeighbors.reversed().take(numFurthestNeighbors)

  val positiveExamples = nearestNeighbors.map { it.item().loc.getContext(0) }
    .filter { it.isNotBlank() }
//    .alsoSummarize("Positive examples (nearest neighbors)")

  val negativeExamples = furthestNeighbors.map { it.item().loc.getContext(0) }
//    .alsoSummarize("Negative examples (furthest neighbors)")

  val dfa = synthesizeDFA(positiveExamples, negativeExamples)

//  Visualization.visualize(dfa, DEFAULT_ALPHABET)

  val resultsOfDFAQuery = strings.filterByDFA(dfa)
    .sortedBy { euclidDist(neighborhood.vector, vecMap[it]!!) }
//    .alsoSummarize("Results of synthetic query:")

  // How many nearest neighbors not in the positive examples were retrieved?

  val testSetSize = 100
  val (truePositives, falseNegatives) =
    neighborhood.nearestNeighbors.drop(numNearestNeighbors).take(testSetSize)
      .map { it.item().loc.getContext(0) }.partition { it in resultsOfDFAQuery }

  // https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg

//  truePositives.alsoSummarize("True positives")
//  falseNegatives.alsoSummarize("False negatives")

  println()
  val precision = truePositives.size.toDouble() / resultsOfDFAQuery.size
  val recall = truePositives.size.toDouble() / testSetSize
  println("DFA-kNN Precision: ${truePositives.size}/${resultsOfDFAQuery.size} = $precision")
  println("DFA-kNN Recall:    ${truePositives.size}/$testSetSize = $recall")
  return Pair(precision, recall)
}

fun List<String>.alsoSummarize(title: String) = also {
  println("\n$title\n".let { it + it.map { "=" }.joinToString("") + "\n" })
  forEachIndexed { i, it ->
    if (i !in 6..(size - 5)) println("$i.) $it")
    if (i == 5) println("...")
  }
}

// Active: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example1.java
// Active: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example2.java
// Active: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example3.java
// Passive: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/passive/Example1.java

val DEFAULT_ALPHABET: Alphabet<Char> =
  (' '..'~').toList().let { Alphabets.fromCollection(it) }
fun synthesizeDFA(
  positiveSamples: List<String> = emptyList(),
  negativeSamples: List<String> = emptyList(),
  alphabet: Alphabet<Char> = DEFAULT_ALPHABET,
) =
//  https://www.ibisc.univ-evry.fr/~janodet/pub/tjs04.pdf
  BlueFringeRPNIDFA(alphabet).apply {
    addPositiveSamples(*positiveSamples.filter { it.all { DEFAULT_ALPHABET.containsSymbol(it) } }.toWords())
    addNegativeSamples(*negativeSamples.filter { it.all { DEFAULT_ALPHABET.containsSymbol(it) } }.toWords())
  }.computeModel()

fun List<String>.toWords() = map { Word.fromCharSequence(it) }.toTypedArray()