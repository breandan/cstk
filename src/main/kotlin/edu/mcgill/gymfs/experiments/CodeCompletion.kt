package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.Levenshtein
import kotlin.math.abs
import kotlin.reflect.KFunction1

fun main() {
  val validationSet = TEST_DIR.allFilesRecursively().allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 750 }.take(1000)
//    .also { printOriginalVsTransformed(it) }

  evaluateTransformation(validationSet, String::same, String::renameTokens)
}

val defaultTokenizer = BasicTokenizer(false)
fun evaluateTransformation(
  validationSet: Sequence<String>,
  vararg codeTxs: KFunction1<String, String>
) =
  codeTxs.forEach { codeTx ->
    validationSet.map { method -> method to codeTx(method) }
      .map { (original, variant) ->
        variant.maskIdentifiers().shuffled().take(3)
          .mapNotNull { masked -> scoreCompletion(variant, masked) }
      }.fold(0.0 to 0.0) { (total, sum), mtdScores ->
        (total + mtdScores.size to sum + mtdScores.sum()).also { (total, sum) ->
          val runningAverage = (sum / total).toString().take(6)
          println("Running accuracy of $MODEL with [${codeTx.name}] " +
            "transformation ($total samples): $runningAverage\n")
        }
      }
  }

fun scoreCompletion(groundTruth: String, maskedSeqeunce: String): Double? {
  val completion = complete(maskedSeqeunce)
  val completionLines = completion.lines()
  val groundTruthLines = groundTruth.lines()
  val maskedLines = maskedSeqeunce.lines()

  // TODO: sometimes masking modifies other lines, why?
  if (completionLines.size != maskedLines.size ||
    maskedLines.size != groundTruthLines.size) {
//    System.err.println("\n\nError: mismatched lines!\n\n")
    return null
  }

  // TODO: instead of just one, mask multiple tokens method and compare
  // only compare line of masked token
  val maskedLineNo = maskedSeqeunce.lines().indexOfFirst { "<mask>" in it }
  val maskedLine = maskedLines[maskedLineNo]
  val actualLine = groundTruthLines[maskedLineNo]
  val predictedLine = completionLines[maskedLineNo]

  val leven = Levenshtein().distance(predictedLine, actualLine).toInt()
  val abs = abs(maskedLine.length - actualLine.length)

  // Show some examples which are reasonably sized for CLI
  if (groundTruthLines.all { it.length < 80 } && maskedLines.size in 3..10) {
    printSideBySide(groundTruth, maskedSeqeunce, "synthetic variant", "masked")
    printSideBySide(actualLine, predictedLine, "ground truth", "prediction")
    println("".padEnd(167, '=') + "\n\n")
  }

  return if (leven == abs) 1.0 else 0.0
}

fun String.maskIdentifiers(): List<String> =
  split(Regex("((?<=[^\\w])|(?=[^\\w]))")).let {
    it.mapIndexed { index, s -> index to s }
      .filter { (_, token) ->
        token.length > 1
          && token.all(Char::isJavaIdentifierPart)
          // Not learning syntax
          && token !in reservedWords
          // Singleton tokens are impossible to predict in zero-shot setting
          && 1 < split(token).size - 1
      }.map { maskIndex ->
        it.foldIndexed("") { i, acc, tok ->
          acc + if (i == maskIndex.first) "<mask>" else tok
        }
      }
  }