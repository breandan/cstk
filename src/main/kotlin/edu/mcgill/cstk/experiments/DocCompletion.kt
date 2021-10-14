package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*

fun main() =
  DATA_DIR.also { println("Evaluating doc completion with $MODEL on $it...") }
    .allFilesRecursively()
    .toList().shuffled().asSequence().allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }
    .filter { docCriteria(it.lines().first()) }
    .flatMap {
      setOf(
        String::renameTokens,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps
      ).map { sct -> CodeSnippet(original=it, sct=sct) }
    }
    .mapNotNull { snp -> evaluateDocSynthesis(snp)?.let { snp to it } }
    .forEachIndexed { i, (snippet, score) ->
      rougeScoreByCyclomaticComplexity[snippet] = score
      println(rougeScoreByCyclomaticComplexity.toLatexTable())
    }

fun evaluateDocSynthesis(snippet: CodeSnippet): Double? {
  val (originalDoc, originalCode) = snippet.original.splitDocAndCode()
  val refactoredCode = snippet.sct(originalCode)
  if (refactoredCode == originalCode) return null
  val originalCodeWithSyntheticJavadoc = originalCode.prependJavadoc()
  val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
  val refactoredCodeWithSyntheticJavadoc = snippet.sct(originalCode).prependJavadoc()
  val syntheticJavadocForRefactoredCode = refactoredCodeWithSyntheticJavadoc.getDoc()

  val rougeScoreWithoutRefactoring = rougeSynonym(originalDoc, syntheticJavadocForOriginalCode)
  val rougeScoreWithRefactoring = rougeSynonym(originalDoc, syntheticJavadocForRefactoredCode)

  if (rougeScoreWithoutRefactoring != 0.0) {
    printSideBySide(snippet.original, originalCodeWithSyntheticJavadoc,
      leftHeading = "original doc", rightHeading = "synthetic doc before refactoring")
    printSideBySide(snippet.original, refactoredCodeWithSyntheticJavadoc,
      leftHeading = "original doc", rightHeading = "synthetic doc after refactoring")
    println("=--Original")
    println(snippet.original)
    println("=--SyntheticJDoc")
    println(originalCodeWithSyntheticJavadoc)
    println("=--RefactoredSyntheticJdoc")
    println(refactoredCodeWithSyntheticJavadoc)
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