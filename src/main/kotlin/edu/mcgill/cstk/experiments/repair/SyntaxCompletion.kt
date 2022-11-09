package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.hasBalancedBrackets
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
./gradlew completeSyntax
 */

fun main() {
  val models = MODELS// + tidyparse
  DATA_DIR.also { println("Evaluating syntax completion using $models on $it...") }
    .allFilesRecursively().allMethods().map { it.first.lineSequence() }.flatten()
    .filter(String::isANontrivialStatementWithBalancedParentheses)
    .map { it to it.constructLevelOneHalfRepair() }
    .runningFold(models.associateWith { (0 to 0) }) { scores, (groundTruth, prompt) ->
      models.associateWith { model ->
        val completion = model.completeUntilStopChar(prompt + model.mask, maxTokens = 50)
        scores[model]!!.let { (n, d) ->
          if (!completion.endsWith(";")) n to d
          else (n + if (completion.hasBalancedBrackets()) 1 else 0) to (d + 1)
        }
      }
    }
    .filterIndexed { i, _ -> i % 10 == 0 }
    .forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

// Can model reconstruct a syntactically valid snippet from its truncated form?
fun String.constructLevelOneHalfRepair() =
  substringBeforeLast('(').trim() + '('

fun String.isANontrivialStatementWithBalancedParentheses(
  parensAndDepth: Pair<Int, Int> = countBracketsAndMaxDepth(),
) = trim().endsWith(';') && hasBalancedBrackets() &&
  parensAndDepth.let { (p, d) -> p == 0 && 2 < d }