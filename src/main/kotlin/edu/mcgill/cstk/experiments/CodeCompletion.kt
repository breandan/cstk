package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import kotlin.math.absoluteValue
import kotlin.reflect.KFunction1

data class CodeSnippet(
  val original: String,
  val complexity: Int = original.approxCyclomatic(), // Cyclomatic complexity
  val sct: KFunction1<String, String>, // Source code transformation
  val variant: String = sct(original)
) {
  override fun hashCode() = complexity.hashCode() + sct.hashCode()
  fun print() = printSideBySide(original, variant)
}

fun main() {
  val validationSet = DATA_DIR.allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }

  evaluateTransformations(
    validationSet = validationSet,
    evaluation = CodeSnippet::evaluateMultimask,
    codeTxs = arrayOf(
      String::renameTokens,
      String::shuffleLines,
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
  }

  fun toLatexTable(colWidth: Int = 20) =
      """
      \begin{table}[H]
      \begin{tabular}{l|ccc}
      
      """.trimIndent() +
        transformations.joinToString(
          " & ",
          "Complexity & ".padEnd(colWidth),
          "\\\\\n\\hline\n"
        ) { it.name.take(15).padEnd(colWidth) } +
        complexities.toSortedSet().joinToString("\\\\\n") { cplx ->
          "$cplx ".padEnd(colWidth) + "& " + transformations.toSortedSet(compareBy { it.name })
            .joinToString("      &".padEnd(colWidth)) { tx ->
              (
                (scoreByCodeSnippet[CodeSnippet("", cplx, tx, "").hashCode()] ?: listOf())
                  .let {
                    it.average().toString().take(5) + " ± " +
                      it.variance().toString().take(5) + " (${it.size})"
                  }
              ).padEnd(colWidth)
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

fun String.evaluateMultimask(): Double =
  maskIdentifiers().shuffled().take(SAMPLES)
    .mapNotNull { (maskedMethod, trueToken) ->
      val (completion, score) = completeAndScore(trueToken, maskedMethod)
//      logDiffs(this, maskedMethod, trueToken, completion)
      if (completion == ERR) null else score
    }.average()

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