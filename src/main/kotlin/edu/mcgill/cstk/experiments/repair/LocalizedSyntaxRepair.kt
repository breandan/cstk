package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*

/**
 * In this experiment, we sample nontrivial single-line statements with balanced
 * bracket from MiniGithub, delete a random bracket and feed the location to the
 * model, ask it to predict what the deleted token was.
 *
 * This will produce scores for each model, i.e., how many tokens it predicted
 * correctly out of the total number of samples tested:
 *
 *     Scores [model=(valid, total)]:
 *     microsoft/codebert-base-mlm=(1225, 3510)
 *     huggingface/CodeBERTa-small-v1=(827, 3510)
 *     microsoft/graphcodebert-base=(1021, 3510)
 *     dbernsohn/roberta-java=(715, 3510)
 *     tidyparse=(1113, 3510)
 */

/*
./gradlew localizedSyntaxRepair
 */

val MSK = "___"

// Our model predicts all possible locations as well as the fix
// Fine tune model to predict whether it is missing parentheses
// or not (single-bit prediction for error location)
// If results are comparable (unmodified) in terms of # of fixes,
// binary classification task
// Easy setting: everyone knows the location
// Hard setting: location unknown (maybe BIFI/NLM get confused)

// TODO:
//     Design custom grammars to fix each (sub)category of errors
//     Goal is to minimize the grammar size (no need to be complicated)
//     Evaluate Tidyparse with grammar to compare with accuracy of BIFI

// Read paper carefully and try to understand why the categories
// do not match raw count, if cannot resolve why, write to authors

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
            if (completion == groundTruth) (n + 1) to (d + 1) else n to (d + 1)
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

fun String.coarsen(): String =
  tokenize().joinToString(" ") {
    if (it.isBracket()) it else if (it == MSK) "_" else "w"
  }

fun String.uncoarsen(originalString: String) =
  originalString.tokenize().zip(tokenize())
    .joinToString("") { (a, b) -> if (a == MSK) b else a }

fun String.isBracket() = length == 1 && this in brackets

fun String.constructPrompt(): String =
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

fun String.findRepairs(cfg: CFG, exclusions: Set<Int>, fishyLocations: List<Int>, maxResults: Int = 10): List<String> =
  queryModel(
    cfg = cfg,
    tokens = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
    maxResults = maxResults,
    variations = listOf {
      it.multiTokenSubstitutionsAndInsertions(
        numberOfEdits = 2,
        exclusions = exclusions,
        fishyLocations = fishyLocations
      )
    }
  )

fun String.queryModel(
  cfg: CFG,
  tokens: List<String> = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
  sanitized: String = tokens.joinToString(" "),
  maxResults: Int = 20,
  variations: List<(String) -> Sequence<String>> =
    listOf(
      String::everySingleHoleConfig,
      String::increasingLengthChunks
    ),
): List<String> =
  sanitized.synthesizeIncrementally(cfg = cfg, variations = variations)
  .take(maxResults).toList().sortedWith(
    compareBy(tokenwiseEdits(tokens)).thenBy { it.length }
  )

private fun tokenwiseEdits(tokens: List<String>): (String) -> Comparable<*> =
  { levenshtein(tokens.filterNot { it.containsHole() }, it.tokenizeByWhitespace()) }