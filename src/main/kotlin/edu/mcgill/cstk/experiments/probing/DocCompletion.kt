package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.math.*
import edu.mcgill.cstk.nlp.*
import edu.mcgill.cstk.rewriting.*

/**
./gradlew completeDoc
 */
fun main() {
  DATA_DIR.allFilesRecursively()
    .allMethods()
    .map { it.first pp it.second }
    // Ensure tokenized method fits within attention
    .filter { (method, origin) ->
      defaultTokenizer.tokenize(method).size < 500 &&
//        method.approxCyclomatic() < 15 &&
        method.lines().any { line -> docCriteria(line) }
    }
    .flatMap { (method, origin) ->
      val codeTxs = setOf(
        String::renameTokens,
        String::permuteArgumentOrder,
        String::swapMultilineNoDeps,
        String::addExtraLogging,
//        String::swapPlusMinus
      )
      (codeTxs * MODELS).map { (sct, model) -> method pp origin pp sct pp model }
    }
    .forEachIndexed { i, (method, origin, sct, model) ->
      val groundTruth = method.getDoc()
      val (originalCode, transformedCode) = method cc sct(method)
      // Do not evaluate snippets which remain unchanged after transformation
      if (originalCode == transformedCode) return@forEachIndexed

      val originalCodeWithSyntheticJavadoc = model.fillFirstDoc(originalCode) ?: return@forEachIndexed
      val transformedCodeWithSyntheticJavadoc = model.fillFirstDoc(transformedCode) ?: return@forEachIndexed
      val syntheticJavadocForOriginalCode = originalCodeWithSyntheticJavadoc.getDoc()
      val syntheticJavadocForRefactoredCode = transformedCodeWithSyntheticJavadoc.getDoc()
      val snippet = CodeSnippetToEvaluate(
        method = method, origin = origin, sct = sct,
        variant = transformedCodeWithSyntheticJavadoc, model = model
      )

      val rougeScoreWithoutRefactoring = rougeSynonym(groundTruth, syntheticJavadocForOriginalCode)
      val rougeScoreWithRefactoring = rougeSynonym(groundTruth, syntheticJavadocForRefactoredCode)

      // Original doc
      val od = groundTruth.substringAfter("//").trim()
      // Synthetic doc before refactoring
      val sd = syntheticJavadocForOriginalCode.substringAfter("//").trim()
      // Synthetic doc after refactoring
      val rd = syntheticJavadocForRefactoredCode.substringAfter("//").trim()

      // Only report nontrivial lines (i.e. should contain some text)
      // when there is a variance between the synthetic docs after refactoring
      if (od.length > 30 && sd.isNotBlank() && sd != rd) {
        printLatexSummary(
          summary = """
            Generative model: $model
            Ground truth doc: $od
            Synth origin doc: $sd
            Synth refact doc: $rd
            """.trimIndent(),
          original = method,
          synthetic = originalCodeWithSyntheticJavadoc,
          variant = transformedCodeWithSyntheticJavadoc,
          discrepancy = """
            Rouge score before refactoring: $rougeScoreWithoutRefactoring
            Rouge score after refactoring: $rougeScoreWithRefactoring
            """.trimIndent()
        )
        println("% Origin: $origin\n")
      }

      rougeScores[snippet] = rougeScoreWithRefactoring cc rougeScoreWithoutRefactoring

      if (i < 20 || i % 100 == 0) rougeScores.reportResults("doc_completion")
    }
}

val rougeScores =
  CodeSnippetAttributeScoresTable<V2<Double>>(
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
    doc.joinToString("\n") cc code.joinToString("\n")
  }

fun String.dropDoc() = lines().filterNot(docCriteria).joinToString("\n")
fun String.getDoc() = lines().first { docCriteria(it) }
//.filter(docCriteria).joinToString("\n")