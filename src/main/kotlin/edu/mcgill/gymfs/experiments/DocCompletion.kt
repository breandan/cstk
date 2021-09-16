package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import edu.mcgill.gymfs.math.rougeSynonym
import edu.mcgill.gymfs.nlp.*

fun main() {
  DATA_DIR.allFilesRecursively()
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }
    .filter { docCriteria(it.lines().first()) }
    .mapNotNull { originalMethod ->
      val (originalDoc, originalCode) = originalMethod.splitDocAndCode()
      val originalCodeWithSyntheticJavadoc = originalCode.prependJavadoc()
      val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
      val refactoredCodeWithSyntheticJavadoc = originalCode.renameTokens()
        .swapMultilineNoDeps().permuteArgumentOrder().prependJavadoc()
      val syntheticJavadocForRefactoredCode = refactoredCodeWithSyntheticJavadoc.getDoc()

      val rougeScoreWithoutRefactoring = rougeSynonym(originalDoc, syntheticJavadocForOriginalCode)
      val rougeScoreWithRefactoring = rougeSynonym(originalDoc, syntheticJavadocForRefactoredCode)

      if(rougeScoreWithoutRefactoring == 0.0) null else {
        printSideBySide(originalMethod, originalCodeWithSyntheticJavadoc,
          leftHeading = "original doc", rightHeading = "synthetic doc before refactoring")
        printSideBySide(originalMethod, refactoredCodeWithSyntheticJavadoc,
          leftHeading = "original doc", rightHeading = "synthetic doc after refactoring")
        println("Rouge score before refactoring: $rougeScoreWithoutRefactoring")
        println("Rouge score after refactoring: $rougeScoreWithRefactoring")
        rougeScoreWithoutRefactoring - rougeScoreWithRefactoring
      }
    }.fold(0.0 to 0.0) { (total, sum), rougeScore ->
      (total + 1.0 to sum + rougeScore).also { (total, sum) ->
        val runningAverage = (sum / total).toString().take(6)
        println("Running average ROUGE 2.0 score difference " +
//          "between original Javadoc and synthetic Javadoc before and after refactoring " +
          "of $MODEL on document synthesis: $runningAverage"
        )
      }
    }
}

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
