package edu.mcgill.cstk.experiments.probing

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import edu.mcgill.cstk.rewriting.*

// Broadly can be considered as a mutlti-choice QA task
fun main() {
    evaluateTransformations(
      validationSet = DATA_DIR
        .also { println("Evaluating  using $MODELS on $it...") }
        .allFilesRecursively().allMethods()
        .map { it.first to it.second }
      // Ensure tokenized method fits within attention
      //.filter { defaultTokenizer.tokenize(it).size < 500 }
      ,
      evaluation = CodeSnippetToEvaluate::evaluateMRR,
      codeTxs = arrayOf(
        String::renameTokens,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps,
        String::addExtraLogging,
//      String::fuzzLoopBoundaries,
//      String::mutateSyntax,
//      String::shuffleLines
      )
    )
}

// https://en.wikipedia.org/wiki/Mean_reciprocal_rank
fun CodeSnippetToEvaluate.evaluateMRR(): Pair<Double, Double>? =
  (model.evaluateMultimaskMC(method) to model.evaluateMultimaskMC(variant))
    .let { (a, b) ->
      if (a.second > 0 && b.second > 0)
        (a.first.toDouble() / a.second.toDouble()) to
          (b.first.toDouble() / b.second.toDouble())
      else null
    }

/** Multiple choice version of [Model.evaluateMultimask]. */
fun Model.evaluateMultimaskMC(code: String, SAMPLES: Int = 200): Pair<Int, Int> =
  dists.get(code) {
    code.maskIdentifiers().shuffled(DEFAULT_RAND).take(SAMPLES)
      .mapNotNull { (maskedMethod, trueToken) ->
        val distractors = code.getDistractors(trueToken)
        val choices = (distractors + trueToken).toSet().toList()
        val (completion, score) = completeAndScoreMC(trueToken, maskedMethod, choices)
//        logDiffs(code, maskedMethod, trueToken, completion)
        if (completion == ERR || completion.isEmpty()) null else score
      }.fold(0 to 0) { (correct, total), it ->
        if (it > 0) correct + 1 to total + 1 else correct to total + 1
      }
  }

fun String.getDistractors(trueToken: String): List<String> =
  TODO("Sample distractors from parent context")

/** Multiple choice version of [Model.completeAndScore]. */
fun Model.completeAndScoreMC(
  correctToken: String,
  maskedSeqeunce: String,
  hints: List<String>
): Pair<String, Int> =
//   complete(maskedSeqeunce).let { it to if (correctToken.startsWith(it.trim())) 1.0 else 0.0 }
  makeQuery(maskedSeqeunce, hints).let {
    // Sometimes the source code token starts with the correct sequence, but
    // since source code tokens can be comprised of multiple BERT tokens, we
    // assume that if the prefix matches the ground truth, it is "correct".
    // Since the model returns its top-5 predictions, this is equivalent to
    // top-5 accuracy. This might be invalid if there are multiple tokens
    // with the same prefix.
    it.firstOrNull { correctToken.startsWith(it.trim()) }
      ?.let { it to 1 }
      ?: (it.first() to 0)
  }
