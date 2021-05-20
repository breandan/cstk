package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.markovian.mcmc.toMarkovChain
import java.io.File
import kotlin.time.*

@ExperimentalTime
fun main() {
  val mc = measureTimedValue {
    val data = DATA_DIR.allFilesRecursively(walkIntoCompressedFiles = true)
      .take(100).toList().joinToString("\n") {
        it.allLines().joinToString("\n")
//        .replace(Regex("[a-zA-Z0-9_]*"), "w")
      }.asSequence()

    // TODO: translate identifiers to placeholders for symbolic automata
    data.toMarkovChain(memory = 3)
  }.also { println("Training time: ${it.duration}") }.value

  println("Tokens: " + mc.size)
//  measureTimedValue {
//    println("Ergodic:" + mc.isErgodic())
//  }.also { println("Ergodicity time: ${it.duration}") }

  measureTimedValue { mc.sample().take(200).flatten().toList()
  }.also {
    println("Sample: " + it.value.joinToString(""))
    println("Sampling time: ${it.duration}")
  }
}