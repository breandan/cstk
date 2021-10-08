package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.*
import edu.mcgill.gymfs.nlp.*
import kotlin.reflect.*

fun main() {

  DATA_DIR.allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }
    .filter { docCriteria(it.lines().first()) }
    .flatMap {
      fun String.sct() = renameTokens().swapMultilineNoDeps().permuteArgumentOrder()
      setOf(
        String::renameTokens,
        String::shuffleLines,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps
//          String::sct
      ).map { sct -> CodeSnippet(snippet=it, sct=sct) }
    }.mapNotNull { snippet ->
      val (originalDoc, originalCode) = snippet.snippet.splitDocAndCode()
      val originalCodeWithSyntheticJavadoc = originalCode.prependJavadoc()
      val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
      val refactoredCodeWithSyntheticJavadoc = snippet.variant.prependJavadoc()
      val syntheticJavadocForRefactoredCode = refactoredCodeWithSyntheticJavadoc.getDoc()

      val rougeScoreWithoutRefactoring = rougeSynonym(originalDoc, syntheticJavadocForOriginalCode)
      val rougeScoreWithRefactoring = rougeSynonym(originalDoc, syntheticJavadocForRefactoredCode)

      if (rougeScoreWithoutRefactoring == 0.0) null else {
        printSideBySide(snippet.snippet, originalCodeWithSyntheticJavadoc,
          leftHeading = "original doc", rightHeading = "synthetic doc before refactoring")
        printSideBySide(snippet.snippet, refactoredCodeWithSyntheticJavadoc,
          leftHeading = "original doc", rightHeading = "synthetic doc after refactoring")
        println("Rouge score before refactoring: $rougeScoreWithoutRefactoring")
        println("Rouge score after refactoring: $rougeScoreWithRefactoring")
        rougeScoreWithoutRefactoring - rougeScoreWithRefactoring
      }?.also {
        rougeScoreByCyclomaticComplexity.getOrPut(snippet.complexity) { mutableListOf() }.add(it)
      }
    }.fold(0.0 to 0.0) { (total, sum), rougeScore ->
      (total + 1.0 to sum + rougeScore).also { (total, sum) ->
        val runningAverage = (sum / total).toString().take(6)
        println("Running average ROUGE 2.0 score difference " +
//          "between original Javadoc and synthetic Javadoc before and after refactoring " +
          "of $MODEL on document synthesis: $runningAverage"
        )

        rougeScoreByCyclomaticComplexity.toSortedMap()
          .forEach { (cc, rs) -> println("$cc, ${rs.average()}, ${rs.variance()}") }
      }
    }
}

val rougeScoreByCyclomaticComplexity = mutableMapOf<Int, MutableList<Double>>()

val docCriteria: (String) -> Boolean = {
  val line = it.trim()
  line.startsWith("/*") ||
//    line.startsWith("//") ||
    line.startsWith("*")
}

fun String.splitDocAndCode() =
  lines().partition(docCriteria).let { (doc, code) ->
    doc.joinToString("\n") to code.joinToString("\n")
  }

fun String.dropDoc() = lines().filterNot(docCriteria).joinToString("\n")
fun String.getDoc() = lines().filter(docCriteria).joinToString("\n")
