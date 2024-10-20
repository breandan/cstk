package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.levenshteinAlign
import ai.hypergraph.kaliningraph.parsing.paintANSIColors
import edu.mcgill.cstk.experiments.repair.mapToBIFITokens
import edu.mcgill.cstk.experiments.repair.mapToUnquotedPythonTokens
import edu.mcgill.cstk.experiments.repair.naturallySmallRepairs
import edu.mcgill.cstk.experiments.repair.sizeAndDistBalancedRepairsUnminimized
import edu.mcgill.cstk.experiments.repair.vanillaS2PCFG
import edu.mcgill.cstk.llama3.Llama3
import edu.mcgill.cstk.utils.tokenizeAsPython

/*
./gradlew probeLLaMA --console=plain
 */
fun main() {
  val model = "Llama-3.1-8B-Instruct-Q4_0.gguf"
  val vocab = Llama3.getVocab(model)
  val iterator = StateCounter(vanillaS2PCFG, vocab)
  val sampler: (FloatArray) -> Int = { logits -> iterator.countStates(logits) }
  fun makePrompt(code: String) = """
    In the following task, you will be presented with a broken code snippet. Your job is to 
    output a list of the nearest most likely intended repairs by edit distance, enclosed in 
    tags <repair> and </repair>. Make as few changes as possible to the original code snippet 
    so it parses. Each repair must be distinct from every other. Do not explain your thinking, 
    do not stop before outputting three repairs, do not output the same repair more than once,
    and do not use any English or non-Python tokens in the output except for the tags.
    
    Here is the broken code snippet that was rejected by the Python parser:
    
    <broken_code>$code</broken_code>
    
    And here is are the three distinct nearest most likely intended repairs:
    """.trimIndent()

  var total = 0
  var correct = 0

  sizeAndDistBalancedRepairsUnminimized.filter {
    it.π1.split(" ").size < 30 &&
        // count the number of NEWLINE tokens
        it.π1.split(" ").count { it == "NEWLINE" } == 1
  }.forEach {
    println("Original: ${it.π3.trim()}")
    println("Abstract: ${it.π1.trim()}\n")
    val repairs = Llama3.prompt(model, makePrompt(it.π3), sampler)
    val abstracted = extractRepairs(repairs).map { it.mapToUnquotedPythonTokens() + " NEWLINE" }
    val pretty = abstracted.mapIndexed { i , s -> "$i.) " + levenshteinAlign(it.π1, s).paintANSIColors() }
    println("\nPredicted repairs:\n${pretty.joinToString("\n")}\n")
    println("True repair: ${it.π4.trim()}")
    println("True repair: ${levenshteinAlign(it.π1, it.π2).paintANSIColors()}\n")
    val index = abstracted.indexOf(it.π2)
    total++
    if (-1 == index) println("True repair missed.")
    else { correct++; println("True repair found! ($index)") }
    println("Accuracy: $correct / $total = ${correct.toDouble() / total}\n")
  }
}

// Takes a string containing <repair>...</repair> tags and extracts the repairs
fun extractRepairs(repairs: String): List<String> {
  val repairRegex = "<repair>(.*?)</repair>".toRegex()
  return repairRegex.findAll(repairs).map { it.groupValues[1] }.toList()
}

class StateCounter(val g: CFG, val vocab: Array<String>) {
//  val dfa = g.startPTree("_ _ _".split(" "))!!.toDFA()!!
//  var state = dfa.initialState

  fun countStates(logits: FloatArray): Int {
    val n = 1
    return logits.withIndex()
      .sortedByDescending { it.value }
//      .filter { 'e' !in vocab[it.index] }
      .take(n).last().index
  }
}