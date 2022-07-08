package edu.mcgill.cstk.experiments.probing

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*

/**
./gradlew completeSyntax
 */
fun main() {
// In this experiment, we first sample statements with balanced parentheses,
// chop after last open parenthesis, then sample autoregressively from the model
// until an end of statement token is emitted and test how many samples have
// balanced parentheses.
  DATA_DIR
    .also { println("Evaluating syntax completion using $MODELS on $it...") }
    .allFilesRecursively().allMethods().map { it.first.lineSequence() }
    .flatten()
    .filter { it.endsWith(";") && 2 < it.count { it == '(' } && it.count { it == '(' } == it.count { it == ')' } }
    .map { it to it.substringBeforeLast('(').trim() + "(" }
    .forEach {
      println("Ground truth: ${it.first}")
      MODELS.forEach { model ->
        val completion = model.complete(it.second + model.mask, maxTokens = 50)
        if(completion.endsWith(";"))
//        println("Sample ($model): " + )
        println("Balanced parens ($model): ${completion.count { it == '(' }} / ${completion.count { it == ')' }}")
      }
    }
}
