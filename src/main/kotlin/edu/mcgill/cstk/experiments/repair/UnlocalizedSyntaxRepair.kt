package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*

/**
 * Synthetic errors in natural data with unlocalized repair
 *
 * In this experiment, we sample nontrivial single-line statements with balanced
 * bracket from MiniGithub, delete a random bracket without telling the location
 * to the model, and ask it to predict the repair. If the ground truth is in the
 * repair set, it gets a 1 else 0.
 */

/*
./gradlew unlocalizedSyntaxRepair
 */

fun main() {
  val models = setOf(tidyparse)
  val modelScores: Scores = models.associateWith { (0 to 0) }

  DATA_DIR.also { println("Evaluating syntax repair using $models on $it...") }
    .allFilesRecursively().allMethods()
    .map { it.first.lineSequence() }.flatten()
    .map { it.trim() }
    .filter(String::isANontrivialStatementWithBalancedBrackets)
    .filter { cfg.parse(it.coarsen()) != null }
    .map { it to it.constructPrompt().replace(MSK, "") }
    .runningFold(modelScores) { scores, (groundTruth, prompt) ->
      models.associateWith { model ->
        val repairs: List<String> = prompt.dispatchTo(model, cfg)
        diagnoseSyntheticErrorUnlocalizedRepair(groundTruth, prompt, repairs)
        scores[model]!!.let { (n, d) -> // numerator / denominator
          if (groundTruth in repairs) (n + 1) to (d + 1) else n to (d + 1)
        }
      }
    }
    .forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

private fun diagnoseSyntheticErrorUnlocalizedRepair(
  code: String,
  prompt: String,
  repairs: List<String>
) {
  println("""
Original code:

$code

Prompt:

$prompt

Repairs:

${repairs.mapIndexed { i, it -> (if(it == code) "*" else "") + "$i.) ${it.trim()}" }.joinToString("\n")}

"""
  )
}