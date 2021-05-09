package edu.mcgill.gymfs.experiments

import de.learnlib.algorithms.rpni.BlueFringeRPNIDFA
import edu.mcgill.gymfs.disk.TEST_DIR
import edu.mcgill.gymfs.indices.buildOrLoadVecIndex
import edu.mcgill.gymfs.math.euclidDist
import net.automatalib.words.*
import net.automatalib.words.impl.Alphabets
import kotlin.time.*

// TODO: DFA/RegEx or BoW query?

// Active: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example1.java
// Active: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example2.java
// Active: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example3.java
// Passive: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/passive/Example1.java

val alphabet: Alphabet<Char> =
  (' '..'~').toList().let { Alphabets.fromCollection(it) }

@OptIn(ExperimentalTime::class)
fun main() {
  val (labels, vectors) = fetchOrLoadSampleData(1000)

  val knnIndex = buildOrLoadVecIndex(rootDir = TEST_DIR)

  val mostSimilarSamples = measureTimedValue {
    labels.zip(vectors).take(1).mapIndexed { i, (l, v) ->
      Neighborhood(l, v, knnIndex.nearestNonEmptyNeighbors(v, 100000))
    }.sortedBy { it.totalDistance }
  }.let { println("Built KNN in:" + it.duration); it.value }

  val query = mostSimilarSamples.first()
  println("\nQuery:\n======\n${query.origin}")

  val nearestNeighbors = query.nearestNeighbors.take(10)
  val furthestNeighbors = query.nearestNeighbors.reversed().take(100)

  val positiveExamples = nearestNeighbors.map { it.item().loc.getContext(0) }
    .filter { it.isNotBlank() }
    .alsoSummarize("Positive examples (nearest neighbors)")

  val negativeExamples = furthestNeighbors.map { it.item().loc.getContext(0) }
    .alsoSummarize("Negative examples (furthest neighbors)")

  val synthesizedDFA =
    computeModel(alphabet, positiveExamples, negativeExamples)

//  Visualization.visualize(secondModel, alphabet)

  val resultsOfDFAQuery = labels.zip(vectors)
    .filter { (l, _) ->
      try {
        synthesizedDFA.accepts(l.toCharArray().toList())
      } catch (exception: Exception) {
        false
      }
    }.sortedBy { (_, v) -> euclidDist(query.vector, v) }
    .map { it.first }

  resultsOfDFAQuery.alsoSummarize("Synthesized query results:")
}

fun List<String>.alsoSummarize(title: String) = also {
  println("\n$title\n".let { it + it.map { "=" }.joinToString("") + "\n" })
  forEachIndexed { i, it ->
    if (i !in 6..(size - 5)) println("$i.) $it")
    if (i == 5) println("...")
  }
}

fun computeModel(
  alphabet: Alphabet<Char>,
  positiveSamples: List<String> = emptyList(),
  negativeSamples: List<String> = emptyList()
) =
//  https://www.ibisc.univ-evry.fr/~janodet/pub/tjs04.pdf
  BlueFringeRPNIDFA(alphabet).apply {
    addPositiveSamples(*positiveSamples.toWords())
    addNegativeSamples(*negativeSamples.toWords())
  }.computeModel()

fun List<String>.toWords() = map { Word.fromCharSequence(it) }.toTypedArray()