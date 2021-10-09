package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.*
import edu.mcgill.gymfs.nlp.*
import kotlin.math.absoluteValue
import kotlin.reflect.KFunction1

data class CodeSnippet(
  val original: String,
  val complexity: Int = original.approxCyclomatic(), // Cyclomatic complexity
  val sct: KFunction1<String, String>, // Source code transformation
  val variant: String = sct(original)
) {
  override fun hashCode() = complexity.hashCode() // TODO: + sct.hashCode()
}

fun main() {
  val validationSet = DATA_DIR.allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }
    .take(100).toList().shuffled()
//    .also { printOriginalVsTransformed(it) }

  evaluateTransformations(validationSet,
    evaluation = CodeSnippet::evaluateMultimask,
    String::addExtraLogging, String::renameTokens, String::same,
    String::swapMultilineNoDeps, String::permuteArgumentOrder
  )
}

val defaultTokenizer = BasicTokenizer(false)
fun evaluateTransformations(
  validationSet: List<String>,
  evaluation: KFunction1<CodeSnippet, Double>,
  vararg codeTxs: KFunction1<String, String>
) =
  validationSet.asSequence()
    .map { method -> setOf(method) * codeTxs.toSet() }.flatten()
    .map { (method, codeTx) -> CodeSnippet(original = method, sct = codeTx) }
    .map { snippet -> snippet to snippet.evaluateMultimask() }
    .forEach { (snippet, metric) ->
      snippet to metric.also {
        csByMultimaskPrediction.getOrPut(snippet) { mutableListOf() }
          .add(metric)
      }
    }
//    .fold(0.0 to 0.0) { (total, sum), rougeScore ->
//      (total + 1.0 to sum + rougeScore).also { (total, sum) ->
//        val runningAverage = (sum / total).toString().take(6)
//        println("Running average ROUGE 2.0 score difference " +
////          "between original Javadoc and synthetic Javadoc before and after refactoring " +
//          "of $MODEL on document synthesis: $runningAverage"
//        )
//
//        rougeScoreByCyclomaticComplexity.toSortedMap(compareBy { it.complexity })
//          .forEach { (cc, rs) -> println("${cc.complexity}, ${rs.average()}, ${rs.variance()}") }
//      }
//    }
//    .fold(0.0 to 0.0) { (total, sum), (snippet, score) ->
//      (total + score to sum + mtdScores.sum()).also { (total, sum) ->
//        val runningAverage = (sum / total).toString().take(6)
//        println(
//          "Running accuracy of $MODEL with [${snippet.sct.name}] " +
//            "transformation ($total samples): $runningAverage\n"
//        )
//      }
//    }

val csByMultimaskPrediction = mutableMapOf<CodeSnippet, MutableList<Double>>()

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