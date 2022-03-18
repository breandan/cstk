package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.types.times
import com.github.benmanes.caffeine.cache.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.approxCyclomatic
import edu.mcgill.cstk.nlp.*
import edu.mcgill.cstk.rewriting.*
import org.hipparchus.stat.inference.*
import java.io.File
import java.net.URI
import kotlin.math.round
import kotlin.random.Random
import kotlin.reflect.KFunction1

data class CodeSnippetToEvaluate constructor(
  val method: String,
  val origin: URI? = null,
  val complexity: Int = binByComplexity(method.approxCyclomatic()),
  val sct: KFunction1<String, String>, // Source code transformation
  val variant: String = sct(method),
  val model: Model = defaultModel
) {
  companion object {
    const val BINSIZE = 5
    fun binByComplexity(complexity: Int) =
      round(complexity.toDouble() / BINSIZE).toInt()
    fun dummy(sct: KFunction1<String, String>, model: Model) =
      CodeSnippetToEvaluate("", null, 0, sct=sct, "", model=model)
  }
  override fun hashCode() = model.hashCode() + sct.name.hashCode()
  fun print() = printSideBySide(method, variant)
}

/**
./gradlew completeCode
 */
fun main() {
  evaluateTransformations(
    validationSet = DATA_DIR
        .also { println("Evaluating code completion using $MODELS on $it...") }
        .allFilesRecursively().allMethods()
        .map { it.first to it.second }
    // Ensure tokenized method fits within attention
    //.filter { defaultTokenizer.tokenize(it).size < 500 }
    ,
    evaluation = CodeSnippetToEvaluate::evaluateMultimask,
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

val defaultTokenizer = BasicTokenizer(false)
fun evaluateTransformations(
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
      evaluation(snippet)?.let { csByMultimaskPrediction[snippet] = it }

      if (i < 20 || i % 10 == 0) csByMultimaskPrediction.reportResults("code_completion")
    }

fun tTest(it: List<Pair<Double, Double>>): String =
  it.unzip().let { (a, b) ->
    if (2 < a.size && 2 < b.size) // t statistic requires at least two
      TTest().pairedTTest(
        a.toDoubleArray(),
        b.toDoubleArray()
      ).toString().take(5) + " (${it.size})"
    else ""
  }

fun sideBySide(it: List<Pair<Double, Double>>) =
  it.unzip().let { (before, after) ->
    before.joinToString(", ", "\t\"before\": [", "],\n") {
      it.toString().take(5).padEnd(5)
    } + after.joinToString(", ", "\t\"after\":  [", "]\n") {
      it.toString().take(5).padEnd(5)
    }
  }

val csByMultimaskPrediction =
  CodeSnippetAttributeScoresTable<Pair<Double, Double>>(::tTest, ::sideBySide)

class CodeSnippetAttributeScoresTable<V>(
  // Renders the significance test for a single configuration
  val significanceTest: (List<V>) -> String,
  // Renders the distribution for each independent variable configuration
  val distToString: (List<V>) -> String = { it.joinToString() },
) {
  val scoreByCodeSnippet = mutableMapOf<Int, MutableList<V>>()
  val complexities = mutableSetOf<Int>()
  val transformations = mutableSetOf<KFunction1<String, String>>()

  // Adds a given score to the table
  operator fun set(snippet: CodeSnippetToEvaluate, metric: V) {
    scoreByCodeSnippet.getOrPut(snippet.hashCode()) { mutableListOf() }.add(metric)
    complexities += snippet.complexity
    transformations += snippet.sct
//    println("Put ${metric.toString().take(6)} in (model = ${defaultModel}, complexity=${snippet.complexity}, SCT=${snippet.sct.name})")
  }

  operator fun get(snippet: CodeSnippetToEvaluate): List<V> =
    scoreByCodeSnippet[snippet.hashCode()] ?: emptyList()

  /* Example of table output:
   %\begin{table}[H]
   %  \begin{tabular}{l|cccc}
   %    Model                         & renameTokens                  & permuteArgument               & swapMultilineNo               & addExtraLogging               \\\hline\\
   %    microsoft/codebert-base-mlm   & 1.232 (3467)                  & 1.324 (1683)                  & 0.421 (2642)                  & 4.378 (1188)                  \\
   %    microsoft/graphcodebert-base  & 2.977 (3474)                  & 0.632 (1695)                  & 0.656 (2642)                  & 5.639 (1198)                  \\
   %    dbernsohn/roberta-java        & 0.0 (3479)                    & 1.808 (1693)                  & 0.705 (2642)                  & 2.638 (1216)
   %  \end{tabular}
   %\end{table}
   */
  fun toLatexTable(colWidth: Int = 30) =
    ("""
      \begin{table}[H]
      \begin{tabular}{l|${"c".repeat(transformations.size)}}
      
      """.trimIndent() +
      transformations.joinToString(
        "& ",
        "Model".padEnd(colWidth) + "& ",
        "\\\\\\hline\\\\\n"
      ) { tx -> tx.name.take(15).padEnd(colWidth) } +
      MODELS.joinToString("\\\\\n") { model ->
        model.name.padEnd(colWidth) + "& " +
          transformations.joinToString("& ") { tx ->
              // Construct a fake code snippet with the same hash code as this cell
              // to retrieve all matching code snippet data from this cell
              this[CodeSnippetToEvaluate.dummy(tx, model)]
                .let { significanceTest(it).padEnd(colWidth) }
            }
      } +
//      complexities.toSortedSet().joinToString("\\\\\n") { cplx ->
//        (cplx * 10).let { "$it-" + (it + 10) }.padEnd(colWidth) + "& " +
//          transformations.toSortedSet(compareBy { it.name })
//            .joinToString("& ") { tx ->
//              summarizer(this[CodeSnippetToEvaluate("", cplx, tx, "")])
//                .padEnd(colWidth)
//            }
//      } +
      """
        
      \end{tabular}
      \end{table}
      """.trimIndent()).lines().joinToString("\n") { "%$it" }

  fun JSONify() =
    """
      {
      ${
        MODELS.joinToString(",\n") { model ->
          """
            "$model": {
              ${
                transformations.joinToString(",\n") { tx ->
                  "\t\t\"${tx.name}\": {\n" +
                    distToString(this[CodeSnippetToEvaluate.dummy(tx, model)]) +
                       "}"
                }
              }
            }
          """.trimIndent().prependIndent("\t")
        }
      }
      }
    """.trimIndent()

  fun reportResults(filename: String) {
    println(toLatexTable())
    val outputFile = File("$filename.json")
    JSONify().let { outputFile.writeText(it) }
    val fullPath = outputFile.absolutePath
    println("% For full distribution, see: $fullPath")
  }
}

// https://en.wikipedia.org/wiki/Relative_change_and_difference
fun CodeSnippetToEvaluate.evaluateMultimask(): Pair<Double, Double>? =
  (model.evaluateMultimask(method) to model.evaluateMultimask(variant))
    .let { (a, b) ->
      if (a.second > 0 && b.second > 0)
        (a.first.toDouble() / a.second.toDouble()) to
          (b.first.toDouble() / b.second.toDouble())
      else null
    }

val dists: Cache<String, Pair<Int, Int>> = Caffeine.newBuilder().maximumSize(100).build()

// Masking all identifiers in all snippets is too expensive,
// so instead we sample a small number of mask positions
fun Model.evaluateMultimask(code: String, SAMPLES: Int = 200): Pair<Int, Int> =
  dists.get(code) {
    code.maskIdentifiers().shuffled(DEFAULT_RAND).take(SAMPLES)
      .mapNotNull { (maskedMethod, trueToken) ->
        val (completion, score) = completeAndScore(trueToken, maskedMethod)
        logDiffs(this, code, maskedMethod, trueToken, completion)
        if (completion == ERR || completion.isEmpty()) null else score
      }.fold(0 to 0) { (correct, total), it ->
        if (it > 0) correct + 1 to total + 1 else correct to total + 1
      }
  }

fun logDiffs(
  model: Model, original: String, maskedSequence: String,
  correctToken: String, completion: String,
  hints: Collection<String> = listOf(),
  logFrequency: Double = 0.1,
) {
  if (Random.nextDouble() < (1.0 - logFrequency)) return
  val maskedSequenceWithOptionalHints =
    maskedSequence.replace(
      model.mask, if (hints.isNotEmpty())
        "<mask hints=[${hints.joinToString()}]>"
      else model.mask
    )
  // only compare line of masked token
  val maskedLines = maskedSequenceWithOptionalHints.lines()

  // Show some examples which are reasonably sized for CLI
  if (maskedLines.all { it.length < 80 } && maskedLines.size in 3..10) {
    val maskPattern = Regex("<mask( hints=\\[.*])?>")
    val maskedLineNo = maskedLines.indexOfFirst { maskPattern.containsMatchIn(it) }
    val maskedLine = maskedLines[maskedLineNo].trimStart()
    val actualLine = maskedLine.replace(maskPattern, correctToken)
    val predictedLine = maskedLine.replace(maskPattern, completion)

    printSideBySide(original, maskedSequenceWithOptionalHints, "original", "masked")
    printSideBySide(actualLine, predictedLine, "ground truth", "prediction")
    println("".padEnd(167, '=') + "\n\n")
  }
}

fun Model.completeAndScore(
  correctToken: String,
  maskedSeqeunce: String,
): Pair<String, Int> =
//   complete(maskedSeqeunce).let { it to if (correctToken.startsWith(it.trim())) 1.0 else 0.0 }
  makeQuery(maskedSeqeunce).let {
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

// Returns various maskings with the masked word
fun String.maskIdentifiers(): List<Pair<String, String>> =
  split(Regex("((?<=\\W)|(?=\\W))")).let {
    it.mapIndexed { index, maskedWord -> index to maskedWord }
      .filter { (_, token) ->
        token.isVariable() && 2 < split(token).size // More than two occurrences
      }.map { indexAndMask ->
        it.foldIndexed("") { i, acc, tok ->
          acc + if (i == indexAndMask.first) "<mask>" else tok
        } to indexAndMask.second
      }
  }