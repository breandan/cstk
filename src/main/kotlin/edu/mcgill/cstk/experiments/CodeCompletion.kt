package edu.mcgill.cstk.experiments

import com.github.benmanes.caffeine.cache.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import kotlin.math.*
import kotlin.reflect.KFunction1

data class CodeSnippet(
  val original: String,
  val complexity: Int = binByComplexity(original.approxCyclomatic()),
  val sct: KFunction1<String, String>, // Source code transformation
  val variant: String = sct(original)
) {
  companion object {
    const val BINSIZE = 5
    fun binByComplexity(complexity: Int) =
      round(complexity.toDouble() / BINSIZE).toInt()
  }
  override fun hashCode() = complexity.hashCode() + sct.name.hashCode()
  fun print() = printSideBySide(original, variant)
}

fun main() {
  val validationSet = DATA_DIR
    .also { println("Evaluating code completion with $MODEL on $it...") }
    .allFilesRecursively().allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 200 }

  evaluateTransformations(
    validationSet = validationSet,
    evaluation = CodeSnippet::evaluateMultimask,
    codeTxs = arrayOf(
//      String::renameTokens,
//      String::permuteArgumentOrder,
//      String::swapMultilineNoDeps,
//      String::addExtraLogging,
      String::fuzzLoopBoundaries,
//      String::mutateSyntax,
      String::shuffleLines
    )
  )
}

val defaultTokenizer = BasicTokenizer(false)
fun evaluateTransformations(
  validationSet: Sequence<String>,
  evaluation: KFunction1<CodeSnippet, Double>,
  vararg codeTxs: KFunction1<String, String>
) =
  validationSet
    .flatMap { method -> setOf(method) * codeTxs.toSet() }
    .map { (method, codeTx) -> CodeSnippet(original = method, sct = codeTx) }
    .filter { it.original != it.variant }
    .mapNotNull { snippet -> evaluation(snippet).let { if (it.isFinite()) snippet to it else null} }
    .forEach { (snippet, metric) ->
      csByMultimaskPrediction[snippet] = metric
      println(csByMultimaskPrediction.toLatexTable())
    }

val csByMultimaskPrediction = CodeSnippetAttributeScoresTable()

class CodeSnippetAttributeScoresTable {
  val scoreByCodeSnippet = mutableMapOf<Int, MutableList<Double>>()
  val complexities = mutableSetOf<Int>()
  val transformations = mutableSetOf<KFunction1<String, String>>()

  operator fun set(snippet: CodeSnippet, metric: Double) {
    scoreByCodeSnippet.getOrPut(snippet.hashCode()) { mutableListOf() }.add(metric)
    complexities += snippet.complexity
    transformations += snippet.sct
    println("Put ${metric.toString().take(6)} in (complexity=${snippet.complexity}, SCT=${snippet.sct.name})")
  }

  operator fun get(snippet: CodeSnippet): List<Double> =
    scoreByCodeSnippet[snippet.hashCode()] ?: emptyList()

  /* Example of table output:
\begin{table}[H]
\begin{tabular}{l|ccc}
Complexity          & renameTokens        & permuteArgument     & swapMultilineNo     \\\hline\
10-20               & 0.011 ± 0.003 (594) & 0.046 ± 0.016 (539) & 1.683 ± 1.680 (594) \\
20-30               & 0.031 ± 0.003 (442) & 0.056 ± 0.012 (441) & 0.004 ± 6.042 (442) \\
30-40               & 0.023 ± 0.003 (243) & 0.086 ± 0.016 (242) & 0.003 ± 4.389 (243) \\
40-50               & 0.029 ± 0.003 (147) & 0.071 ± 0.016 (147) & 0.014 ± 0.001 (147) \\
50-60               & 0.027 ± 0.003 (286) & 0.091 ± 0.011 (286) & 0.014 ± 0.002 (286) \\
60-70               & 0.034 ± 0.004 (149) & 0.082 ± 0.009 (149) & 0.024 ± 0.003 (149) \\
70-80               & 0.045 ± 0.005 (49)  & 0.084 ± 0.009 (49)  & 0.078 ± 0.009 (49)  \\
80-90               & 0.054 ± 0.005 (57)  & 0.105 ± 0.016 (57)  & 0.077 ± 0.010 (57)  \\
90-100              & 0.062 ± 0.008 (40)  & 0.085 ± 0.010 (40)  & 0.080 ± 0.007 (40)  \\
100-110             & 0.022 ± 0.001 (22)  & 0.054 ± 0.010 (22)  & 0.036 ± 0.004 (22)  \\
110-120             & 0.073 ± 0.009 (34)  & 0.091 ± 0.007 (34)  & 0.064 ± 0.008 (34)  \\
120-130             & 0.032 ± 0.002 (25)  & 0.092 ± 0.011 (25)  & 0.044 ± 0.005 (25)  \\
130-140             & 0.037 ± 0.002 (27)  & 0.055 ± 0.005 (27)  & 0.077 ± 0.007 (27)  \\
140-150             & 0.065 ± 0.005 (23)  & 0.078 ± 0.013 (23)  & 0.095 ± 0.009 (23)  \\
150-160             & 0.004 ± 3.840 (25)  & 0.016 ± 0.002 (25)  & 0.012 ± 0.001 (25)  \\
160-170             & 0.030 ± 0.003 (13)  & 0.023 ± 0.001 (13)  & 0.007 ± 7.100 (13)  \\
\end{tabular}
\end{table}
   */
  fun toLatexTable(colWidth: Int = 20) =
    ("""
      \begin{table}[H]
      \begin{tabular}{l|${"c".repeat(transformations.size)}}
      
      """.trimIndent() +
      transformations.joinToString(
        "& ",
        "Complexity ".padEnd(colWidth) + "& ",
        "\\\\\\hline\\\\\n"
      ) { it.name.take(15).padEnd(colWidth) } +
      complexities.toSortedSet().joinToString("\\\\\n") { cplx ->
        (cplx * 10).let { "$it-" + (it + 10) }.padEnd(colWidth) + "& " +
          transformations.toSortedSet(compareBy { it.name })
            .joinToString("& ") { tx ->
              this[CodeSnippet("", cplx, tx, "")]
                .let {
                  it.average().toString().take(5) + " ± " +
                    it.variance().toString().take(5) + " (${it.size})"
                }.padEnd(colWidth)
            }
      } +
      """
        
      \end{tabular}
      \end{table}
      """.trimIndent()).lines().joinToString("\n") { "%$it" }
}

// https://en.wikipedia.org/wiki/Relative_change_and_difference
fun CodeSnippet.evaluateMultimask(): Double =
  (original.evaluateMultimask() to variant.evaluateMultimask())
    .let { (a, b) -> (a - b) / max(a, b) }

val dists: Cache<String, Double> = Caffeine.newBuilder().maximumSize(100).build()

// Masking all identifiers in all snippets is too expensive,
// so instead we sample a small number of mask positions
fun String.evaluateMultimask(SAMPLES: Int = 10): Double =
  dists.get(this) {
    maskIdentifiers().shuffled().take(SAMPLES)
      .mapNotNull { (maskedMethod, trueToken) ->
        val (completion, score) = completeAndScore(trueToken, maskedMethod)
      logDiffs(this, maskedMethod, trueToken, completion)
        if (completion == ERR) null else score
      }.average()
  }

fun logDiffs(original: String, maskedSequence: String,
             correctToken: String, completion: String) {
  // only compare line of masked token
  val maskedLines = maskedSequence.lines()

  // Show some examples which are reasonably sized for CLI
  if (maskedLines.all { it.length < 80 } && maskedLines.size in 3..10) {
    val maskedLineNo = maskedLines.indexOfFirst { MSK in it }
    val maskedLine = maskedLines[maskedLineNo].trimStart()
    val actualLine = maskedLine.replace(MSK, correctToken)
    val predictedLine = maskedLine.replace(MSK, completion)

    printSideBySide(original, maskedSequence, "original", "masked")
    printSideBySide(actualLine, predictedLine, "ground truth", "prediction")
    println("".padEnd(167, '=') + "\n\n")
  }
}

fun completeAndScore(correctToken: String, maskedSeqeunce: String): Pair<String, Double> =
//   complete(maskedSeqeunce).let { it to if (correctToken.startsWith(it.trim())) 1.0 else 0.0 }
  getPredictions(maskedSeqeunce).let {
    // Sometimes the source code token starts with the correct sequence, but
    // since source code tokens can be comprised of multiple BERT tokens, we
    // assume that if the prefix matches the ground truth, it is "correct".
    // Since the model returns its top-5 predictions, this is equivalent to
    // top-5 accuracy. This might be invalid if there are multiple tokens
    // with the same prefix.
    it.firstOrNull { correctToken.startsWith(it.trim()) }
      ?.let { it to 1.0 }
      ?: (it.first() to 0.0)
  }

// Returns various maskings with the masked word
fun String.maskIdentifiers(): List<Pair<String, String>> =
  split(Regex("((?<=[^\\w])|(?=[^\\w]))")).let {
    it.mapIndexed { index, maskedWord -> index to maskedWord }
      .filter { (_, token) ->
        token.length > 1
          && token.all(Char::isJavaIdentifierPart)
          // Not learning syntax
          && token !in reservedWords
          // Singleton tokens are impossible to predict in zero-shot setting
          && 1 < split(token).size - 1
      }.map { indexAndMask ->
        it.foldIndexed("") { i, acc, tok ->
          acc + if (i == indexAndMask.first) "<mask>" else tok
        } to indexAndMask.second
      }
  }