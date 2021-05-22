package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.markovian.mcmc.toMarkovChain
import kotlin.streams.toList
import kotlin.time.*

@ExperimentalTime
fun main() {
  val mc = measureTimedValue {
    DATA_DIR.allFilesRecursively().toList().parallelStream().map { src ->
      vfsManager.resolveFile("tgz:${src.path}").run {
        println("Indexing $name")
        findFiles(VFS_SELECTOR).mapNotNull {
            try {
//              println(it.uri)
              val text = it.uri.allLines().joinToString("\n")
              if (text.isEmpty()) null
              else text.asSequence().toMarkovChain()
            } catch (e: Exception) { null }
          }.toList()
      }
    }.toList().flatten().reduce { a, b -> a + b }
    // TODO: translate identifiers to placeholders for symbolic automata
  }.also { println("Training time: ${it.duration}") }.value

  println("Tokens: " + mc.size)
//  measureTimedValue {
//    println("Ergodic:" + mc.isErgodic())
//  }.also { println("Ergodicity time: ${it.duration}") }

  measureTimedValue { mc.sample().take(200)
  }.also {
    println("Sample: " + it.value.joinToString(""))
    println("Sampling time: ${it.duration}")
  }
}