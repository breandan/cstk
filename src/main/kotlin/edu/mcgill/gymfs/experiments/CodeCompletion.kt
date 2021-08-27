package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import kotlin.reflect.KFunction1

fun main() {
  val validationSet = DATA_DIR.allFilesRecursively(walkIntoCompressedFiles = true)
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }.take(300).shuffled()
//    .also { printOriginalVsTransformed(it) }

  evaluateTransformations(validationSet, String::renameTokens, String::same)
}

val defaultTokenizer = BasicTokenizer(false)
fun evaluateTransformations(
  validationSet: Sequence<String>,
  vararg codeTxs: KFunction1<String, String>
) =
  codeTxs.forEach { codeTx ->
    validationSet.map { method -> method to codeTx(method) }
      .map { (original, variant) ->
        // Masking all identifiers in all snippets is too expensive,
        // so instead we sample a small number of mask positions
        variant.maskIdentifiers().shuffled().take(10)
          .mapNotNull { (maskedMethod, trueToken) ->
            val (completion, score) = completeAndScore(trueToken, maskedMethod)
            if (completion == ERR) return@mapNotNull null
//            logDiffs(original, maskedMethod, trueToken, completion)
            score
          }
      }.fold(0.0 to 0.0) { (total, sum), mtdScores ->
        (total + mtdScores.size to sum + mtdScores.sum()).also { (total, sum) ->
          val runningAverage = (sum / total).toString().take(6)
          println("Running accuracy of $MODEL with [${codeTx.name}] " +
            "transformation ($total samples): $runningAverage\n")
        }
      }
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