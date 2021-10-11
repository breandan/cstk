package edu.mcgill.cstk.experiments

import com.github.benmanes.caffeine.cache.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.CodeSnippet.Companion.BINSIZE
import edu.mcgill.cstk.experiments.CodeSnippet.Companion.binComplexity
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import edu.mcgill.markovian.mcmc.Dist
import kotlin.math.*
import kotlin.reflect.KFunction1

data class CodeSnippet(
  val original: String,
  val complexity: Int = binComplexity(original.approxCyclomatic()), // Cyclomatic complexity
  val sct: KFunction1<String, String>, // Source code transformation
  val variant: String = sct(original)
) {
  companion object {
    const val BINSIZE = 5
    fun binComplexity(complexity: Int) = round(complexity.toDouble() / BINSIZE).toInt()
  }
  override fun hashCode() = complexity.hashCode() + sct.name.hashCode()
  fun print() = printSideBySide(original, variant)
}

fun main() {
  val validationSet = DATA_DIR
    .also { println("Evaluating doc completion with $MODEL on $it...") }
    .allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention

  evaluateTransformations(
    validationSet = validationSet,
    evaluation = CodeSnippet::evaluateMultimask,
    codeTxs = arrayOf(
      String::renameTokens,
      String::permuteArgumentOrder,
      String::swapMultilineNoDeps
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
    .mapNotNull { snippet -> evaluation(snippet).let { if (it.isNaN()) null else snippet to it } }
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
    println("Put $metric in (${snippet.complexity}, ${snippet.sct})")
  }
  operator fun get(snippet: CodeSnippet): List<Double> =
    scoreByCodeSnippet[snippet.hashCode()] ?: emptyList<Double>()

  /* Example of table output:
\begin{table}[H]
\begin{tabular}{l|ccc}
Complexity &        renameTokens         & shuffleLines         & permuteArgument      & swapMultilineNo     \\\hline\
4                   & -0.85 ± 0.050 (7)         &             -0.56 ± 0.157 (7)         &             -0.85 ± 0.050 (7)         &             -0.85 ± 0.050 (7)   \\
5                   & -1.0 ± 0.0 (1)            &             -0.98 ± 0.0 (1)           &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
7                   & -0.99 ± 9.467 (2)         &             -0.92 ± 0.001 (2)         &             -0.99 ± 9.467 (2)         &             -0.99 ± 9.467 (2)   \\
9                   & -0.99 ± 1.291 (2)         &             -0.66 ± 0.112 (2)         &             -0.99 ± 1.291 (2)         &             -0.99 ± 1.291 (2)   \\
10                  & -0.95 ± 0.0 (1)           &             -0.35 ± 0.0 (1)           &             -0.95 ± 0.0 (1)           &             -0.95 ± 0.0 (1)     \\
12                  & -0.99 ± 8.199 (2)         &             -0.99 ± 5.771 (2)         &             -0.99 ± 8.199 (2)         &             -0.99 ± 8.199 (2)   \\
13                  & -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
14                  & -0.99 ± 0.0 (1)           &             -0.22 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
16                  & -0.84 ± 0.046 (3)         &             -0.51 ± 0.161 (3)         &             -0.84 ± 0.046 (3)         &             -0.84 ± 0.046 (3)   \\
17                  & -0.98 ± 0.0 (1)           &             -0.98 ± 0.0 (1)           &             -0.98 ± 0.0 (1)           &             -0.98 ± 0.0 (1)     \\
18                  & -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
19                  & -1.0 ± 0.0 (1)            &             -0.89 ± 0.0 (1)           &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
20                  & -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
22                  & -0.99 ± 8.641 (2)         &             -0.94 ± 0.002 (2)         &             -0.99 ± 8.641 (2)         &             -0.99 ± 8.641 (2)   \\
24                  & -0.98 ± 9.514 (3)         &             -0.98 ± 8.439 (3)         &             -0.98 ± 9.514 (3)         &             -0.98 ± 9.514 (3)   \\
25                  & -0.98 ± 7.149 (2)         &             -0.98 ± 7.149 (2)         &             -0.98 ± 7.149 (2)         &             -0.98 ± 7.149 (2)   \\
26                  & -0.96 ± 0.0 (1)           &             -0.96 ± 0.0 (1)           &             -0.96 ± 0.0 (1)           &             -0.96 ± 0.0 (1)     \\
27                  & -0.78 ± 0.038 (2)         &             -0.59 ± 0.146 (2)         &             -0.78 ± 0.038 (2)         &             -0.78 ± 0.038 (2)   \\
29                  & -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
30                  & NaN ± NaN (0)             &             -0.99 ± 0.0 (1)           &             NaN ± NaN (0)             &             NaN ± NaN (0)       \\
31                  & -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
35                  & -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
36                  & -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
39                  & -0.96 ± 0.0 (1)           &             -0.94 ± 0.0 (1)           &             -0.96 ± 0.0 (1)           &             -0.96 ± 0.0 (1)     \\
40                  & -0.94 ± 0.002 (2)         &             -0.94 ± 0.002 (2)         &             -0.94 ± 0.002 (2)         &             -0.94 ± 0.002 (2)   \\
42                  & -0.96 ± 2.567 (2)         &             -0.96 ± 1.620 (2)         &             -0.96 ± 2.567 (2)         &             -0.96 ± 2.567 (2)   \\
43                  & -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
44                  & -0.98 ± 0.0 (1)           &             -0.98 ± 0.0 (1)           &             -0.98 ± 0.0 (1)           &             -0.98 ± 0.0 (1)     \\
50                  & -0.99 ± 0.0 (1)           &             -0.98 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
54                  & -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)      \\
59                  & -0.99 ± 0.0 (1)           &             -0.95 ± 0.0 (1)           &             -0.99 ± 0.0 (1)           &             -0.99 ± 0.0 (1)     \\
60                  & -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)            &             -1.0 ± 0.0 (1)
\end{table}
\end{tabular}
   */
  fun toLatexTable(colWidth: Int = 20) =
    """
      \begin{table}[H]
      \begin{tabular}{l|ccc}
      
      """.trimIndent() +
      transformations.joinToString(
        "& ",
        "Complexity ".padEnd(colWidth) + "& ",
        "\\\\\\hline\\\n"
      ) { it.name.take(15).padEnd(colWidth) } +
      complexities.toSortedSet().joinToString("\\\\\n") { cplx ->
        (cplx * 10).let { "$it-" + (it + 10) }.padEnd(colWidth) + "& " +
          transformations.toSortedSet(compareBy { it.name })
            .joinToString("& ") { tx ->
              this[CodeSnippet("", cplx * 10, tx, "")]
                .let {
                  it.average().toString().take(5) + " ± " +
                    it.variance().toString().take(5) + " (${it.size})"
                }.padEnd(colWidth)
            }
      } +
      """
        
      \end{table}
      \end{tabular}
      """.trimIndent()
}

// Masking all identifiers in all snippets is too expensive,
// so instead we sample a small number of mask positions
val SAMPLES = 10
fun CodeSnippet.evaluateMultimask(): Double =
    (original.evaluateMultimask() - variant.evaluateMultimask()).absoluteValue

val dists: Cache<String, Double> = Caffeine.newBuilder().maximumSize(100).build()

fun String.evaluateMultimask(): Double =
  dists.get(this) {
  maskIdentifiers().shuffled().take(SAMPLES)
    .mapNotNull { (maskedMethod, trueToken) ->
      val (completion, score) = completeAndScore(trueToken, maskedMethod)
//      logDiffs(this, maskedMethod, trueToken, completion)
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

fun completeAndScore(correctToken: String, maskedSeqeunce: String) =
   complete(maskedSeqeunce).let { it to if (it == correctToken) 1.0 else 0.0 }

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