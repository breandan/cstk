package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import java.lang.Double.max
import kotlin.math.absoluteValue

//fun main() =
//  DATA_DIR.also { println("Evaluating doc completion with $MODEL on $it...") }
//    .allFilesRecursively()
//    .allMethods()
//    // Ensure tokenized method fits within attention
//    .filter { defaultTokenizer.tokenize(it).size < 400 }
//    .filter { it.lines().any { docCriteria(it) } }
//    .flatMap {
//      setOf(
//        String::renameTokens,
//        String::permuteArgumentOrder,
//        String::swapMultilineNoDeps,
////        String::addExtraLogging
//      ).map { sct -> CodeSnippet(original = it, sct = sct) }
//    }
//    .filter { it.original != it.variant }
//    .mapNotNull { snp -> evaluateDocSynthesis(snp)?.let { snp to it } }
//    .forEachIndexed { i, (snippet, score) ->
//      rougeScoreByCyclomaticComplexity[snippet] = score
//      println(rougeScoreByCyclomaticComplexity.toLatexTable())
//    }
//
//fun evaluateDocSynthesis(snippet: CodeSnippet): Double? {
//  val (originalDoc, originalCode) = snippet.original.splitDocAndCode()
//  val refactoredCode = snippet.sct(originalCode)
//  if (refactoredCode == originalCode) return null
//  val originalCodeWithSyntheticJavadoc = originalCode.prependJavadoc()
//  val syntheticJavadocForOriginalCode =
//    originalCodeWithSyntheticJavadoc.getDoc()
//  val refactoredCodeWithSyntheticJavadoc =
//    snippet.sct(originalCode).prependJavadoc()
//  val syntheticJavadocForRefactoredCode =
//    refactoredCodeWithSyntheticJavadoc.getDoc()
//
//  val rougeScoreWithoutRefactoring =
//    rougeSynonym(originalDoc, syntheticJavadocForOriginalCode)
//  val rougeScoreWithRefactoring =
//    rougeSynonym(originalDoc, syntheticJavadocForRefactoredCode)
//
//  if (rougeScoreWithoutRefactoring != 0.0) {
//    printSideBySide(snippet.original, originalCodeWithSyntheticJavadoc,
//      leftHeading = "original doc", rightHeading = "synthetic doc before refactoring")
//    printSideBySide(snippet.original, refactoredCodeWithSyntheticJavadoc,
//      leftHeading = "original doc", rightHeading = "synthetic doc after refactoring")
//    println("=--Original")
//    println(snippet.original)
//    println("=--SyntheticJDoc")
//    println(originalCodeWithSyntheticJavadoc)
//    println("=--RefactoredSyntheticJdoc")
//    println(refactoredCodeWithSyntheticJavadoc)
//    println("Rouge score before refactoring: $rougeScoreWithoutRefactoring")
//    println("Rouge score after refactoring: $rougeScoreWithRefactoring")
//  }
//  val relativeDifference =
//    (rougeScoreWithoutRefactoring - rougeScoreWithRefactoring) /
//      max(rougeScoreWithRefactoring, rougeScoreWithRefactoring)
//  return if (relativeDifference.isFinite()) relativeDifference else null
//}
//
//val rougeScoreByCyclomaticComplexity = CodeSnippetAttributeScoresTable()
//
//val docCriteria: (String) -> Boolean = {
//  val line = it.trim()
//  line.startsWith("/*") ||
//    line.startsWith("//") ||
//    line.startsWith("*")
//}
//
//fun String.splitDocAndCode() =
//  lines().partition(docCriteria).let { (doc, code) ->
//    doc.joinToString("\n") to code.joinToString("\n")
//  }
//
//fun String.dropDoc() = lines().filterNot(docCriteria).joinToString("\n")
//fun String.getDoc() = lines().filter(docCriteria).joinToString("\n")

fun main() {
  DATA_DIR.allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 && it.approxCyclomatic() < 15 }
    .filter { it.lines().any { docCriteria(it) } }
    .flatMap {
      setOf(
        String::renameTokens,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps,
        String::addExtraLogging
      ).map { sct -> it to sct }
    }
    .forEach { (originalMethod, sct) ->
      val cyclomaticComplexity = originalMethod.approxCyclomatic()
//      println("" + cyclomaticComplexity + " " + sct.name + "..." )
      val groundTruth = originalMethod.getDoc()
      val originalCodeWithSyntheticJavadoc = originalMethod.fillFirstDoc() ?: return@forEach
      val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
      val refactoredCodeWithSyntheticJavadoc = sct(originalMethod).fillFirstDoc() ?: return@forEach
      val syntheticJavadocForRefactoredCode = refactoredCodeWithSyntheticJavadoc.getDoc()

      val rougeScoreWithoutRefactoring = rougeSynonym(groundTruth, syntheticJavadocForOriginalCode)
      val rougeScoreWithRefactoring = rougeSynonym(groundTruth, syntheticJavadocForRefactoredCode)

      val relativeDifference = (rougeScoreWithoutRefactoring - rougeScoreWithRefactoring) /
            max(rougeScoreWithRefactoring, rougeScoreWithRefactoring)
      if(relativeDifference.isFinite() && relativeDifference.absoluteValue > 0.0) {
        println("Ground truth doc: ${groundTruth.substringAfter(("//"))}")
        println("Synth origin doc: ${syntheticJavadocForOriginalCode.substringAfter(("//"))}")
        println("Synth refact doc: ${syntheticJavadocForRefactoredCode.substringAfter(("//"))}")

        printSideBySide(originalMethod, originalCodeWithSyntheticJavadoc,
          leftHeading = "original doc", rightHeading = "synthetic doc before refactoring")
        printSideBySide(originalMethod, refactoredCodeWithSyntheticJavadoc,
          leftHeading = "original doc", rightHeading = "synthetic doc after refactoring")

        println("Rouge score before refactoring: $rougeScoreWithoutRefactoring")
        println("Rouge score after refactoring: $rougeScoreWithRefactoring")
        println("Relative difference: $relativeDifference")

        val snippet = CodeSnippet(originalMethod, cyclomaticComplexity, sct, refactoredCodeWithSyntheticJavadoc)
        rougeScoreByCyclomaticComplexity[snippet] = relativeDifference
        println(rougeScoreByCyclomaticComplexity.toLatexTable())
      }
    }
}

val rougeScoreByCyclomaticComplexity = CodeSnippetAttributeScoresTable()

val docCriteria: (String) -> Boolean = {
  val line = it.trim()
//  line.startsWith("/*") ||
    line.startsWith("//")// ||
//    line.startsWith("*")
}

fun String.splitDocAndCode() =
  lines().partition(docCriteria).let { (doc, code) ->
    doc.joinToString("\n") to code.joinToString("\n")
  }

fun String.dropDoc() = lines().filterNot(docCriteria).joinToString("\n")
fun String.getDoc() = lines().first { docCriteria(it) }
//.filter(docCriteria).joinToString("\n")