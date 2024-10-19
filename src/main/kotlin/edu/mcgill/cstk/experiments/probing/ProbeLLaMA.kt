package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.automata.toDFA
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.startPTree
import edu.mcgill.cstk.experiments.repair.vanillaS2PCFG
import edu.mcgill.cstk.llama3.Llama3

/*
./gradlew probeLLaMA --console=plain
 */
fun main() {
  val model = "Llama-3.2-3B-Instruct-Q4_0.gguf"
  val vocab = Llama3.getVocab(model)
  val iterator = StateCounter(vanillaS2PCFG, vocab)
  val t: (FloatArray) -> Int = { logits -> iterator.countStates(logits) }
  val prompt = """
    In the following task, you will be presented with a broken code snippet. Your job is to 
    output a list of the nearest most likely intended repairs by edit distance, enclosed in 
    tags <begin_repair> and </end_repair>. Make as few changes as possible to the original 
    code snippet so that it parses. Each repair must be distinct from the others. Do not 
    explain your rationale, and do not use any English or other tokens in the output. 
    
    Here is the broken code snippet that was rejected by the Python parser:
    
    <begin_code>
    v = df.iloc(5:, 2:)
    </end_code>
    
    And here are three of the nearest most likely intended distinct repairs:
    """.trimIndent()

  Llama3.prompt(model, prompt, t)
}

class StateCounter(val g: CFG, val vocab: Array<String>) {
//  val dfa = g.startPTree("_ _ _".split(" "))!!.toDFA()!!
//  var state = dfa.initialState

  fun countStates(logits: FloatArray): Int {
    val n = 1
    return logits.withIndex()
      .sortedByDescending { it.value }
      .filter { 'e' !in vocab[it.index] }
      .take(n).last().index
  }
}