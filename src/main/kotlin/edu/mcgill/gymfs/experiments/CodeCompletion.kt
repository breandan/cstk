package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.Levenshtein
import kotlin.math.abs
import kotlin.streams.toList

fun main() {
  val query = "System.out.$MSK"
  println("Query: $query")
  println("Completion: " + complete(query, 3))

  val validationSet = TEST_DIR.allFilesRecursively().allMethods().take(1000)
    .map { method ->
      val (variant, masked) = method.renameTokensAndMask()
      if (variant == method) null else Triple(method, variant, masked)
    }.toList().mapNotNull { it }
//    .also { printOriginalVsTransformed(it) }

  val accuracy = validationSet.map { (original, refactored, masked) ->
    val completion = complete(masked)
    if (Levenshtein().distance(completion, refactored).toInt() ==
      abs(completion.length - refactored.length)) 1.0 else 0.0
  }.average()

  println("Accuracy: $accuracy")
}

fun printOriginalVsTransformed(methodPairs: List<Triple<String, String, String>>) =
  methodPairs.forEach { (original, variant, _) ->
    if(original != variant) {
      val maxLen = 70
      val maxLines = 10
      val methodLines = original.lines()
      val variantLines = variant.lines()
      if(methodLines.all { it.length < maxLen } && methodLines.size < maxLines) {
        methodLines.forEachIndexed { i, l ->
          println(
            l.padEnd(maxLen, ' ') + "|    " +
              variantLines[i].padEnd(maxLen, ' ')
          )
        }
        println(List(maxLen * 2){'='}.joinToString(""))
      }
    }
  }