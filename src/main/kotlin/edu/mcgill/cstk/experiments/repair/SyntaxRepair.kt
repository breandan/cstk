package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.hasBalancedBrackets
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*

/**
 * In this experiment, we sample nontrivial single-line statements with balanced
 * parentheses from MiniGithub, truncate after the last open parenthesis, sample
 * autoregressively from the model under test until an end of statement token is
 * emitted, then measure how many samples are syntactically well-formed.
 *
 * This will produce scores for each model, i.e., how many samples are
 * syntactically well-formed out of the total number of samples tested:
 *
 *      Scores [model=(valid, total)]:
 *      microsoft/codebert-base-mlm=(1423, 3559)
 *      huggingface/CodeBERTa-small-v1=(768, 3681)
 *      microsoft/graphcodebert-base=(1008, 3571)
 *      dbernsohn/roberta-java=(434, 2924)
 *      ...
 */

/*
./gradlew repairSyntax
 */

val MSK = "___"
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
            .synthesizeIncrementally(cfg, allowNTs = false, enablePruning = true, skipWhen = { 20 < it.size })
            .firstOrNull()?.uncoarsen(prompt) ?: prompt
          println("Completion: $completion")
          scores[model]!!.let { (n, d) -> // numerator / denominator
            if (completion.hasBalancedBrackets()) (n + 1) to (d + 1) else n to (d + 1)
          }
        } else {
          0 to 0
//          query = prompt.replace(MSK, model.mask)
//          completion = query.replace(model.mask, model.makeQuery(prompt).first())
//
//          scores[model]!!.let { (n, d) ->  // numerator / denominator
//            if (groundTruth == completion) n to d else n to (d + 1)
//          }
        }
      }
    }
//    .filterIndexed { i, _ -> i % 10 == 0 }
    .forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

fun String.coarsen() =
  tokenize().joinToString(" ") {
    if (it.isBracket()) it else if (it == MSK) "_" else "w"
  }

fun String.uncoarsen(originalString: String) =
  originalString.tokenize().zip(tokenize())
    .joinToString("") { (a, b) -> if (a == MSK) b else a }

fun String.isBracket() = length == 1 && this in brackets

private fun String.constructPrompt(): String =
  tokenize().toMutableList().let { tokens ->
    val index = tokens.indices.filter { tokens[it].isBracket() }.random()
    tokens[index] = MSK
    tokens
  }.joinToString("")

val brackets = "()[]{}<>"
fun String.tokenize() =
  split(Regex("[\\(\\)\\[\\]{}<>]|___".let { "((?<=($it))|(?=($it)))" }))

fun String.isANontrivialStatementWithBalancedBrackets(
  parensAndDepth: Pair<Int, Int> = countBracketsAndMaxDepth(),
) =
  trim().endsWith(';')
    && parensAndDepth.let { (p, d) -> p == 0 && 2 < d }
    && hasBalancedBrackets()