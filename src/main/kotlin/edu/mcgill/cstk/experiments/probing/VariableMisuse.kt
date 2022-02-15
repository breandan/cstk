package edu.mcgill.cstk.experiments.probing

import com.github.benmanes.caffeine.cache.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import edu.mcgill.cstk.rewriting.*
import qa.qf.qcri.iyas.evaluation.ir.MeanReciprocalRank
import java.net.URI
import kotlin.reflect.KFunction1

// Broadly can be considered as a multi-choice QA task
fun main() {
    evaluateTransformations(
      validationSet = DATA_DIR
        .also { println("Evaluating variable misuse detection using $MODELS on $it...") }
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

/** Multiple choice version of [evaluateTransformations]. */
fun evaluateMCTransformations(
  validationSet: Sequence<Pair<String, URI>>,
  evaluation: KFunction1<CodeSnippetToEvaluate, Pair<Double, Double>?>,
  vararg codeTxs: KFunction1<String, String>
) =
  validationSet
    .flatMap { (method, origin) ->
      (setOf(method) * codeTxs.toSet() * MODELS).map { (method, codeTx, model) ->
        CodeSnippetToEvaluate(method = method, origin = origin, sct = codeTx, model = model)
      }
    }
    .filter { it.method != it.variant }
    .forEachIndexed { i, snippet ->
      evaluation(snippet)?.let { csByMRR[snippet] = it }

      if (i < 20 || i % 200 == 0) csByMRR.reportResults("variable_misuse")
    }


val csByMRR =
  CodeSnippetAttributeScoresTable<Pair<Double, Double>>(::tTest, ::sideBySide)

// https://en.wikipedia.org/wiki/Mean_reciprocal_rank
fun CodeSnippetToEvaluate.evaluateMRR(): Pair<Double, Double> =
  (model.evaluateMultimaskMC(method) to model.evaluateMultimaskMC(variant))

val mrrDists: Cache<String, Double> = Caffeine.newBuilder().maximumSize(100).build()

/** Multiple choice version of [Model.evaluateMultimask]. */
fun Model.evaluateMultimaskMC(code: String, SAMPLES: Int = 200): Double =
  mrrDists.get(code) {
    code.maskIdentifiers().shuffled(DEFAULT_RAND).take(SAMPLES)
      .mapNotNull { (maskedMethod, trueToken) ->
        val distractors = code.getDistractors(trueToken)
        val choices = (distractors + trueToken).toSet().toList()
        val results = makeQuery(maskedMethod, choices)
        val gold = results.associateWith { (it == trueToken) }
        if (results.isEmpty()) null else results to gold
      }.let {
        val (rankings, gold) = it.unzip()
        MeanReciprocalRank.computeWithRankingList(rankings, gold)
      }
  }

// Returns distractors
fun String.getDistractors(trueToken: String): List<String> =
  split(Regex("((?<=\\W)|(?=\\W))"))
    .filter { it.isVariable() && it != trueToken }
    .groupingBy { it }.eachCount().entries
    .sortedBy { (_, v) -> v }.map { it.key }.take(5)

fun String.isVariable(): Boolean =
  length > 1 && all(Char::isJavaIdentifierPart) && this !in reservedWords