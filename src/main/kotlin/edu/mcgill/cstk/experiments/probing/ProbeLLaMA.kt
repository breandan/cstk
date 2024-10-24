package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.levenshteinAlign
import ai.hypergraph.kaliningraph.parsing.paintANSIColors
import ai.hypergraph.kaliningraph.parsing.patchSize
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import edu.mcgill.cstk.experiments.repair.LEN_BUCKET_INTERVAL
import edu.mcgill.cstk.experiments.repair.S2PMetrics
import edu.mcgill.cstk.experiments.repair.mapToBIFITokens
import edu.mcgill.cstk.experiments.repair.mapToUnquotedPythonTokens
import edu.mcgill.cstk.experiments.repair.naturallySmallRepairs
import edu.mcgill.cstk.experiments.repair.sizeAndDistBalancedRepairsUnminimized
import edu.mcgill.cstk.experiments.repair.summarizeLenAndDist
import edu.mcgill.cstk.experiments.repair.vanillaS2PCFG
import edu.mcgill.cstk.llama3.Llama3
import edu.mcgill.cstk.utils.tokenizeAsPython
import kotlin.time.TimeSource

/**
 * Precision@1
 * ===========
 * |σ|∈[0, 10): Top-1/total: 81 / 198 = 0.4090909090909091
 * |σ|∈[10, 20): Top-1/total: 66 / 171 = 0.38596491228070173
 * |σ|∈[20, 30): Top-1/total: 39 / 112 = 0.3482142857142857
 * Δ(1)= Top-1/total: 101 / 201 = 0.5024875621890548
 * Δ(2)= Top-1/total: 53 / 175 = 0.3028571428571429
 * Δ(3)= Top-1/total: 32 / 105 = 0.3047619047619048
 * (|σ|∈[0, 10), Δ=1): Top-1/total: 40 / 87 = 0.45977011494252873
 * (|σ|∈[0, 10), Δ=2): Top-1/total: 27 / 80 = 0.3375
 * (|σ|∈[0, 10), Δ=3): Top-1/total: 14 / 31 = 0.45161290322580644
 * (|σ|∈[10, 20), Δ=1): Top-1/total: 33 / 64 = 0.515625
 * (|σ|∈[10, 20), Δ=2): Top-1/total: 19 / 59 = 0.3220338983050847
 * (|σ|∈[10, 20), Δ=3): Top-1/total: 14 / 48 = 0.2916666666666667
 * (|σ|∈[20, 30), Δ=1): Top-1/total: 28 / 50 = 0.56
 * (|σ|∈[20, 30), Δ=2): Top-1/total: 7 / 36 = 0.19444444444444445
 * (|σ|∈[20, 30), Δ=3): Top-1/total: 4 / 26 = 0.15384615384615385
 */

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

  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_AllByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()

  sizeAndDistBalancedRepairsUnminimized.filter {
    it.π1.split(" ").size < 30 &&
        it.π3.length < 180 &&
        // count the number of NEWLINE tokens
        it.π1.split(" ").count { it == "NEWLINE" } == 1
  }.forEach { (invalidTokens, validTokens, invalidCode, validCode) ->
    val toRepair = invalidTokens.tokenizeByWhitespace()
    val humanRepair = validTokens.tokenizeByWhitespace()
    val levDist = levenshteinAlign(toRepair, humanRepair).patchSize()
    val lenBucket = (toRepair.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    total++

    println("Original: ${invalidCode.trim()}")
    println("Abstract: ${invalidTokens.trim()}\n")
    val repairs = Llama3.prompt(model, makePrompt(invalidCode), sampler)
    val abstracted = extractRepairs(repairs).map { it.mapToUnquotedPythonTokens() + " NEWLINE" }
    val pretty = abstracted.mapIndexed { i , s -> "$i.) " + levenshteinAlign(invalidTokens, s).paintANSIColors() }
    println("\nPredicted repairs:\n${pretty.joinToString("\n")}\n")
    println("True repair: ${validCode.trim()}")
    println("True repair: ${levenshteinAlign(invalidTokens, validTokens).paintANSIColors()}\n")

    val index = abstracted.indexOf(validTokens)
    if (-1 == index) println("True repair missed.")
    else { correct++; println("True repair found! ($index)")
      if (index == 0) P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
    }

    println("Accuracy: $correct / $total = ${correct.toDouble() / total}\n")

    println()
    println("Precision@1\n===========")
    println(P_1ByLevDist.summarizeLenAndDist())
    println("Precision@All\n=============")
    println(P_AllByLevDist.summarizeLenAndDist())
    println()
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