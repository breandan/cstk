package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.types.times
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.allMethods
import edu.mcgill.cstk.rewriting.*

/**
./gradlew allTasks
 */
fun main() {
  val codeTxs = arrayOf(
    String::renameTokens,
    String::permuteArgumentOrder,
    String::swapMultilineNoDeps,
    String::addExtraLogging,
//      String::fuzzLoopBoundaries,
//      String::mutateSyntax,
//      String::shuffleLines
  )
  DATA_DIR
    .also { println("Evaluating variable misuse detection using $MODELS on $it...") }
    .allFilesRecursively().allMethods()
    .map { it.first to it.second }
    .flatMap { (method, origin) ->
      (setOf(method) * codeTxs.toSet() * MODELS).map { (method, codeTx, model) ->
        CodeSnippetToEvaluate(
          method = method,
          origin = origin,
          sct = codeTx,
          model = model
        )
      }
    }
    .filter { it.method != it.variant }
    .forEachIndexed { i, snippet ->
      snippet.evaluateMultimask()?.let { csByMultimaskPrediction[snippet] = it }
      snippet.evaluateMRR()?.let { csByMRR[snippet] = it }
      /** TODO: include [rougeScores]*/

      if (i < 20 || i % 20 == 0) {
        csByMultimaskPrediction.reportResults("code_completion")
        csByMRR.reportResults("variable_misuse")
      }
    }
}