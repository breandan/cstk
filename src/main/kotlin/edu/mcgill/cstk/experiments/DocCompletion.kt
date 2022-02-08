package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import java.lang.Double.max

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
    .flatMap { (method, origin) ->
      val codeTxs = setOf(
        String::renameTokens,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps,
//        String::addExtraLogging,
//        String::swapPlusMinus
      )
      (codeTxs * MODELS).map { (sct, model) -> (method to origin) to (sct to model) }
    }
    .forEach { (methodAndOrigin, sctAndModel) ->
      val (method, origin) = methodAndOrigin
      val (sct, model) = sctAndModel
      val groundTruth = method.getDoc()
      val (originalCode, transformedCode) = method to sct(method)
      if (originalCode == transformedCode) return@forEach
      val originalCodeWithSyntheticJavadoc = model.fillFirstDoc(originalCode) ?: return@forEach
      val transformedCodeWithSyntheticJavadoc = model.fillFirstDoc(transformedCode) ?: return@forEach
      val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
      val syntheticJavadocForRefactoredCode = transformedCodeWithSyntheticJavadoc.getDoc()
      val snippet = CodeSnippetToEvaluate(
        method = method, origin = origin, sct = sct,
        variant = transformedCodeWithSyntheticJavadoc, model = model
      )

      val rougeScoreWithoutRefactoring = rougeSynonym(groundTruth, syntheticJavadocForOriginalCode)
      val rougeScoreWithRefactoring = rougeSynonym(groundTruth, syntheticJavadocForRefactoredCode)

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
          variant = transformedCodeWithSyntheticJavadoc,
          discrepancy = """
            Rouge score before refactoring: $rougeScoreWithoutRefactoring
            Rouge score after refactoring: $rougeScoreWithRefactoring
            """.trimIndent()
        )
        println("% Origin: $origin")
      }

      rougeScores[snippet] = rougeScoreWithRefactoring to rougeScoreWithoutRefactoring
      println(rougeScores.toLatexTable())
    }
}

val rougeScores =
  CodeSnippetAttributeScoresTable<Pair<Double, Double>>(
    significanceTest = ::tTest,
    distToString = ::sideBySide
  )

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