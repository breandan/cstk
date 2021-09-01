package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import org.apache.commons.lang3.StringUtils.splitByCharacterTypeCamelCase

fun main() {
  DATA_DIR.allFilesRecursively(walkIntoCompressedFiles = true)
    .allMethods()
    // Ensure tokenized method fits within attention
    .filter { defaultTokenizer.tokenize(it).size < 500 }
    .filter { docCriteria(it.lines().first()) }
    .map { originalMethod ->
      val (originalDoc, originalCode) = originalMethod.splitDocAndCode()
      val syntheticMethod = originalCode.prependJavadoc()
      val syntheticDoc = syntheticMethod.getDoc()
      printSideBySide(originalDoc, syntheticDoc,
        leftHeading = "original doc", rightHeading = "synthetic doc")
      val rougeScore = rouge(originalDoc.synonymCloud(), syntheticDoc.synonymCloud())
      println("Rouge score: $rougeScore")
      rougeScore
    }.fold(0.0 to 0.0) { (total, sum), rougeScore ->
      (total + 1.0 to sum + rougeScore).also { (total, sum) ->
        val runningAverage = (sum / total).toString().take(6)
        println("Running average ROUGE 2.0 score of $MODEL on document synthesis: $runningAverage")
      }
    }
}

fun rouge(reference: Set<String>, candidate: Set<String>) =
  reference.intersect(candidate).size.toDouble() / reference.size.toDouble()


val docCriteria: (String) -> Boolean = {
  val line = it.trim()
  line.startsWith("/*") ||
    line.startsWith("//") ||
    line.startsWith("*")
}

fun String.splitDocAndCode() =
  lines().partition(docCriteria).let { (doc, code) ->
    doc.joinToString("\n") to code.joinToString("\n")
  }

fun String.dropDoc() = lines().filterNot(docCriteria).joinToString("\n")
fun String.getDoc() = lines().filter(docCriteria).joinToString("\n")

fun String.synonymCloud(): Set<String> =
  splitByCharacterTypeCamelCase(this)
    .filter { it.all(Char::isLetter) }
    .map { it.synonyms() }
    .flatten().toSet()