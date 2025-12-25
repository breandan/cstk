package edu.mcgill.cstk.experiments.search

import ai.hypergraph.kaliningraph.types.*
import de.learnlib.algorithm.rpni.BlueFringeRPNIDFA
import edu.mcgill.cstk.disk.TEST_DIR
import edu.mcgill.cstk.disk.indices.*
import edu.mcgill.cstk.math.euclidDist
import net.automatalib.alphabet.Alphabet
import net.automatalib.alphabet.impl.Alphabets
import net.automatalib.automaton.fsa.DFA
import net.automatalib.word.Word

// TODO: Look into extracting automata from the compiler
// Possible to extract SL_k (n-gram presence/absence)?
// https://arxiv.org/pdf/2509.22598

fun main() {
  val (strings, vectors) = fetchOrLoadSampleData(1000)
  val stringToVecMap = strings.zip(vectors).toMap()

  val knnIndex = buildOrLoadVecIndex(rootDir = TEST_DIR)

  val (precisions, recalls) =
    stringToVecMap.entries.take(100).mapNotNull { (s, v) ->
      try {
        calculuatePrecisionAndRecall(s, v, strings, knnIndex, stringToVecMap)
      } catch (e: Exception) {
        println("exception occured")
        null
      }
    }.unzip()

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
): V2<Double> {
  val neighbors = knnIndex.knn(vector, 100000)
  val neighborhood = Neighborhood(query, vector, neighbors)
  println("\nQuery:\n======\n${neighborhood.origin}")

  val numNearestNeighbors = 30
  val numFurthestNeighbors = 100
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
  return precision cc recall
}

fun List<String>.alsoSummarize(title: String) = also {
  println("\n$title\n".let { it + it.map { "=" }.joinToString("") + "\n" })
  forEachIndexed { i, it ->
    if (i !in 6..(size - 5)) println("$i.) $it")
    if (i == 5) println("...")
  }
}

fun List<String>.filterByDFA(dfa: DFA<*, Char>) = filter {
  try {
    dfa.accepts(it.toCharArray().toList())
  } catch (exception: Exception) {
    false
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
): DFA<*, Char> =
//  https://www.ibisc.univ-evry.fr/~janodet/pub/tjs04.pdf
  BlueFringeRPNIDFA(alphabet).apply {
    addPositiveSamples(*positiveSamples.filter { it.all { DEFAULT_ALPHABET.containsSymbol(it) } }.toWords())
    addNegativeSamples(*negativeSamples.filter { it.all { DEFAULT_ALPHABET.containsSymbol(it) } }.toWords())
  }.computeModel()

fun List<String>.toWords() = map { Word.fromCharSequence(it) }.toTypedArray()