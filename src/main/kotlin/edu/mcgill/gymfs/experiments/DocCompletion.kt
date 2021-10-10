package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.*
import edu.mcgill.gymfs.nlp.*

fun main() =
  DATA_DIR.allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }
    .filter { docCriteria(it.lines().first()) }
    .shuffled()
    .flatMap {
      setOf(
        String::renameTokens,
        String::shuffleLines,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps
      ).map { sct -> CodeSnippet(original=it, sct=sct) }
    }
    .map { it to evaluateDocSynthesis(it)}
    .forEach { (snippet, score) ->
      rougeScoreByCyclomaticComplexity[snippet] = score
      println(rougeScoreByCyclomaticComplexity.toLatexTable())
    }

fun evaluateDocSynthesis(snippet: CodeSnippet): Double {
  val (originalDoc, originalCode) = snippet.original.splitDocAndCode()
  val originalCodeWithSyntheticJavadoc = originalCode.prependJavadoc()
  val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
  val refactoredCodeWithSyntheticJavadoc = snippet.variant.prependJavadoc()
  val syntheticJavadocForRefactoredCode = refactoredCodeWithSyntheticJavadoc.getDoc()

  val rougeScoreWithoutRefactoring = rougeSynonym(originalDoc, syntheticJavadocForOriginalCode)
  val rougeScoreWithRefactoring = rougeSynonym(originalDoc, syntheticJavadocForRefactoredCode)

  if (rougeScoreWithoutRefactoring != 0.0) {
    printSideBySide(snippet.original, originalCodeWithSyntheticJavadoc,
      leftHeading = "original doc", rightHeading = "synthetic doc before refactoring")
    printSideBySide(snippet.original, refactoredCodeWithSyntheticJavadoc,
      leftHeading = "original doc", rightHeading = "synthetic doc after refactoring")
    println("Rouge score before refactoring: $rougeScoreWithoutRefactoring")
    println("Rouge score after refactoring: $rougeScoreWithRefactoring")
  }
  return rougeScoreWithoutRefactoring - rougeScoreWithRefactoring
}

val rougeScoreByCyclomaticComplexity = CodeSnippetAttributeScoresTable()

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
