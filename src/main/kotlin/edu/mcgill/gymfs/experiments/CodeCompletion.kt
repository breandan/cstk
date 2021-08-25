package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.Levenshtein
import kotlin.math.abs

fun main() {
//  val query = "System.out.$MSK"
//  println("Query: $query")
//  println("Completion: " + complete(query, 3))

  val tokenizer = FullTokenizer()

  val validationSet = TEST_DIR.allFilesRecursively().allMethods()
    // Ensure tokenized method fits within attention
    .filter { tokenizer.tokenize(it).size < 750 }.take(1000)
    .map { method -> method to method.renameTokens() }
    .toList().map { it }
//    .also { printOriginalVsTransformed(it) }

  val accuracy = validationSet.mapNotNull { (original, refactored) ->
    refactored.maskIdentifiers().map { masked ->

    val completion = complete(masked)
    val completionLines = completion.lines()
    val refactoredLines = refactored.lines()
    val originalLines = original.lines()
    val maskedLines = masked.lines()

    // TODO: sometimes masking modifies other lines, why?
    if (completionLines.size != maskedLines.size ||
      maskedLines.size != refactoredLines.size ||
      refactoredLines.size != originalLines.size
    ) return@mapNotNull null

    // TODO: better heuristic for correct selection
    // TODO: instead of just one, mask multiple tokens method and compare
    // only compare line of masked token
    val maskedLineNo = masked.lines().indexOfFirst { "<mask>" in it }
    val maskedLine = maskedLines[maskedLineNo]
    val unmaskedLine = refactoredLines[maskedLineNo]
    val predictedLine = completionLines[maskedLineNo]

    val leven = Levenshtein().distance(predictedLine, unmaskedLine).toInt()
    val abs = abs(maskedLine.length - unmaskedLine.length)

    // Show some examples
    if(refactoredLines.all { it.length < 80 } && maskedLines.size in 3..10){
    printSideBySide(original, refactored, leftHeading = "original", rightHeading = "synthetic variant")
    printSideBySide(refactored, masked, leftHeading = "synthetic variant", rightHeading = "masked")
    printSideBySide(unmaskedLine, predictedLine, leftHeading = "ground truth", rightHeading = "predicted line")
    println("".padEnd(167, '=') + "\n\n")
    }

    if (leven == abs) 1.0 else 0.0
  }

  }.flatten().average() // average across all mask positions for each code snippet
  println("Accuracy: $accuracy")
}

val defaultTokenizer = BasicTokenizer(false)
fun String.maskIdentifiers(): List<String> =
  split(Regex("((?<=[^\\w])|(?=[^\\w]))")).let {
    it.mapIndexed { index, s -> index to s }
      .filter { it.second.length > 1 && it.second.all { it.isJavaIdentifierPart() } }
      .map { maskIndex ->
        it.foldIndexed("") { i, acc, tok ->
          acc + if (i == maskIndex.first) "<mask>" else tok
        }
      }
  }