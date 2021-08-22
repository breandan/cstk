package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import kotlin.streams.toList

fun main() {
//  val query = "System.out.$MSK"
//  println("Query: $query")
//  println("Completion: " + complete(query, 3))

  val validationSet = TEST_DIR.allFilesRecursively().allCodeFragments().take(99)
    .toList().parallelStream().map {
      val snippet = it.first.getContext(3)
      val (variant, masked) = snippet.renameTokensAndMask()
      if (variant == snippet) null
      else Triple(snippet, variant, masked)
    }.toList().mapNotNull { it }

  val accuracy = validationSet.map { (original, refactored, masked) ->
    val completion = complete(masked)
    if (completion == refactored) 1.0 else 0.0
  }.average()

  println("Accuracy: $accuracy")
}

fun String.maskLastToken(token: String) =
  reversed().replaceFirst(token.reversed(), MSK.reversed()).reversed()

fun String.histogram(): Map<String, Int> =
  split(Regex("[^A-Za-z]")).filter { it.length > 2 }.groupingBy { it }.eachCount()