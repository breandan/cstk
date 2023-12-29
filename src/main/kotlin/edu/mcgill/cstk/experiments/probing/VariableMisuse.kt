package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.types.times
import com.github.benmanes.caffeine.cache.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.rewriting.*
import edu.mcgill.cstk.utils.*
import qa.qf.qcri.iyas.evaluation.ir.MeanReciprocalRank
import java.net.URI
import kotlin.reflect.KFunction1

/**
./gradlew varMisuse
 */
// Broadly can be considered as a multi-choice QA task
fun main() {
  evaluateMCTransformations(
    validationSet = DATA_DIR
      .also { println("Evaluating variable misuse detection using $MODELS on $it...") }
      .allFilesRecursively().allMethods()
    // Ensure tokenized method fits within attention
    //.filter { defaultTokenizer.tokenize(it).size < 500 }
    ,
    evaluation = CodeSnippetToEvaluate::evaluateMRR,
    codeTxs = arrayOf(
      String::renameTokens,
      String::permuteArgumentOrder,
      String::swapMultilineNoDeps,
      String::addExtraLogging,
//    String::fuzzLoopBoundaries,
//    String::mutateSyntax,
//    String::shuffleLines
    )
  )
}

/** Multiple choice version of [evaluateTransformations]. */
fun evaluateMCTransformations(
  validationSet: Sequence<Pair<String, URI>>,
  evaluation: KFunction1<CodeSnippetToEvaluate, V2<Double>?>,
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

      if (i < 20 || i % 20 == 0) csByMRR.reportResults("variable_misuse")
    }

val csByMRR =
  CodeSnippetAttributeScoresTable(::tTest, ::sideBySide)

// https://en.wikipedia.org/wiki/Mean_reciprocal_rank
fun CodeSnippetToEvaluate.evaluateMRR(): V2<Double>? =
  (model.evaluateMultimaskMC(method) cc model.evaluateMultimaskMC(variant))
    .let { (a, b) -> if ((a + b).isNaN()) null else a cc b }

val mrrDists: Cache<String, Double> = Caffeine.newBuilder().maximumSize(100).build()

/** Multiple choice version of [Model.evaluateMultimask]. */
fun Model.evaluateMultimaskMC(code: String, SAMPLES: Int = 200): Double =
  mrrDists.get(code) {
    code.maskIdentifiers().shuffled(DEFAULT_RAND).take(SAMPLES)
      .mapNotNull { (maskedMethod, trueToken) ->
        val distractors = code.getDistractors(trueToken)
        if (distractors.size < 3) return@mapNotNull null
        val choices = (distractors + trueToken).shuffled()
        val results = makeQuery(maskedMethod, choices)
        println("Hints: " + choices.joinToString(",", "[", "]") { if (it == trueToken) "*$it*" else it})
        println("Results" + results.joinToString(",", "[", "]") { if (trueToken.startsWith(it)) "*$it*" else it})
        logDiffs(this, code, maskedMethod, trueToken, results.first(), choices, 1.0)
        val gold = results.associateWith { (trueToken.startsWith(it)) }
        if (results.isEmpty()) null else results to gold
      }.let {
        val (rankings, gold) = it.unzip()
        MeanReciprocalRank.computeWithRankingList(rankings, gold)
      }
  }

// Returns distractors
fun String.getDistractors(trueToken: String): List<String> =
  splitByNonWordChars()
    // only use distractors where first char differs in case token not in vocab
    .filter { it.isVariable() && it.first() != trueToken.first() }
    .groupingBy { it }.eachCount().entries
    .sortedBy { (_, v) -> v }.map { it.key }.take(5)

fun String.isVariable(): Boolean =
  length > 1 && all(Char::isJavaIdentifierPart) && this !in reservedWords