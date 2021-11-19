package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import java.lang.Double.max
import kotlin.math.absoluteValue

fun main() {
  DATA_DIR.allFilesRecursively()
    .allMethods()
    .map { it.first.toString() to it.second }
    // Ensure tokenized method fits within attention
    .filter { (method, origin) ->
      defaultTokenizer.tokenize(method).size < 500 &&
        method.approxCyclomatic() < 15 &&
        method.lines().any { line -> docCriteria(line) }
    }
    .flatMap {
      setOf(
        String::renameTokens,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps,
        String::addExtraLogging,
        String::swapPlusMinus
      ).map { sct -> it to sct }
    }
    .forEach { (methodAndOrigin, sct) ->
      val (method, origin) = methodAndOrigin
      val groundTruth = method.getDoc()
      val originalCodeWithSyntheticJavadoc = method.fillFirstDoc() ?: return@forEach
      val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
      val refactoredCodeWithSyntheticJavadoc = sct(method).fillFirstDoc() ?: return@forEach
      val syntheticJavadocForRefactoredCode = refactoredCodeWithSyntheticJavadoc.getDoc()

      val rougeScoreWithoutRefactoring = rougeSynonym(groundTruth, syntheticJavadocForOriginalCode)
      val rougeScoreWithRefactoring = rougeSynonym(groundTruth, syntheticJavadocForRefactoredCode)

      val relativeDifference = (rougeScoreWithoutRefactoring - rougeScoreWithRefactoring) /
            max(rougeScoreWithRefactoring, rougeScoreWithRefactoring)

      // Original doc
      val oc = groundTruth.substringAfter("//")
      // Synthetic doc before refactoring
      val sc = syntheticJavadocForOriginalCode.substringAfter("//")
      // Synthetic doc after refactoring
      val rc = syntheticJavadocForRefactoredCode.substringAfter("//")

      // Only report nontrivial lines (i.e. should contain some text)
      // when there is a variance between the synthetic docs after refactoring
      if (oc.length > 30 && sc.isNotBlank() && sc != rc) {
        printLatexSummary(
          summary = """
            Ground truth doc: $oc
            Synth origin doc: $sc
            Synth refact doc: $rc
            """.trimIndent(),
          original = method,
          synthetic = originalCodeWithSyntheticJavadoc,
          variant = refactoredCodeWithSyntheticJavadoc,
          discrepancy = """
            Rouge score before refactoring: $rougeScoreWithoutRefactoring
            Rouge score after refactoring: $rougeScoreWithRefactoring
            Relative difference: $relativeDifference""".trimIndent()
        )
        println("% Origin: $origin")
      }

      // Only record nonzero and finite relative differences in comparison
      if (relativeDifference.run { isFinite() && absoluteValue > 0.0 }) {
        val snippet = CodeSnippet(original = method, sct = sct, variant = refactoredCodeWithSyntheticJavadoc)
        rougeScoreByCyclomaticComplexity[snippet] = relativeDifference
        println(rougeScoreByCyclomaticComplexity.toLatexTable())
      }
    }
}

val rougeScoreByCyclomaticComplexity = CodeSnippetAttributeScoresTable<Double> {
  it.average().toString().take(5) + " ± " +
    it.variance().toString().take(5) + " (${it.size})"
}

val docCriteria: (String) -> Boolean = {
  val line = it.trim()
//  line.startsWith("/*") ||
    line.startsWith("//") ||
    line.startsWith(" *")
}

fun String.splitDocAndCode() =
  lines().partition(docCriteria).let { (doc, code) ->
    doc.joinToString("\n") to code.joinToString("\n")
  }

fun String.dropDoc() = lines().filterNot(docCriteria).joinToString("\n")
fun String.getDoc() = lines().first { docCriteria(it) }
//.filter(docCriteria).joinToString("\n")