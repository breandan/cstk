package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*

/**
 * In this experiment, we sample nontrivial single-line statements with balanced
 * bracket from MiniGithub, delete a random bracket without telling the location
 * to the model, and ask it to predict the repair.
 *
 * This will produce scores for each model, i.e., how many repairs it predicted
 * correctly out of the total number of samples tested:
 *
 *     TBD
 */

/*
./gradlew unlocalizedSyntaxRepair
 */

fun main() {
  val tidyparse = Model("tidyparse")
  val cfg = """S -> w | ( ) | [ ] | < > | { } | ( S ) | [ S ] | < S > | { S } | S S""".parseCFG()
  val modelScores: Map<Model, Pair<Int, Int>> =
    (MODELS + tidyparse).associateWith { (0 to 0) }

  DATA_DIR.also { println("Evaluating syntax repair using $MODELS on $it...") }
    .allFilesRecursively().allMethods()
    .map { it.first.lineSequence() }.flatten()
    .filter(String::isANontrivialStatementWithBalancedBrackets)
    .map { it to it.constructPrompt() }
    .runningFold(modelScores) { scores, (groundTruth, prompt) ->
      (MODELS + tidyparse).associateWith { model ->
        var query: String
        var completion: String

        if (model == tidyparse) {
          println("Prompt: $prompt")
          query = prompt.coarsen()
          completion = query
            .synthesizeIncrementally(cfg, allowNTs = false,
              enablePruning = true, skipWhen = { 20 < it.size })
            .firstOrNull()?.uncoarsen(prompt) ?: prompt
          println("Completion: $completion")
          scores[model]!!.let { (n, d) -> // numerator / denominator
//            if (completion.hasBalancedBrackets())
            if (completion == groundTruth)
              (n + 1) to (d + 1) else n to (d + 1)
          }
        } else {
          query = prompt.replace(MSK, model.mask)
          completion = model.complete(query)

          scores[model]!!.let { (n, d) ->  // numerator / denominator
            if (groundTruth == completion) (n + 1) to (d + 1) else n to (d + 1)
          }
        }
      }
    }
//    .filterIndexed { i, _ -> i % 10 == 0 }
    .forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}