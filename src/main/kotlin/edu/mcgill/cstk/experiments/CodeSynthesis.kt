package edu.mcgill.cstk.experiments

import ai.hypergraph.markovian.mcmc.toMarkovChain
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.allLines
import kotlin.time.*

@ExperimentalTime
fun main() {
  val mc = measureTimedValue {
    TEST_DIR.allFilesRecursively(readCompressed = false).toList()
      .map { src ->
        try { src.allLines().joinToString("\n") } catch (e: Exception) { "" }
          .asSequence().chunked(3).toMarkovChain(2)
    }.toList().reduce { a, b -> a + b }
    // TODO: translate identifiers to placeholders for symbolic automata
  }.also { println("Training time: ${it.duration}") }.value

  println("Tokens: " + mc.size)

  measureTimedValue { mc.sample().map { it.joinToString("") }.take(200) }
    .also {
      println("Sample: " + it.value.joinToString(""))
      println("Sampling time: ${it.duration}")
    }
}