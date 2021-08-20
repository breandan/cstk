package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*

fun main() {
//  val query = "System.out.$MSK"
//  println("Query: $query")
//  println("Completion: " + complete(query, 3))

  val validationSet = TEST_DIR.allFilesRecursively().allCodeFragments().take(99)
    .mapNotNull {
      val snippet = it.first.getContext(3)
      val histogram = snippet.histogram()
      val mostFreqToken = histogram.maxByOrNull { it.value }
      if (mostFreqToken?.value?.let { it < 2 } != false) null
      else snippet.replace(mostFreqToken.key, "xx").let {
        it to it.reversed().replaceFirst("xx", MSK.reversed()).reversed()
      }
    }

  val accuracy = validationSet.map { (truth, masked) ->
    val completion = complete(masked)
    if (completion == truth) 1.0 else 0.0
  }.average()

  println("Accuracy: $accuracy")
}

fun String.histogram(): Map<String, Int> =
  split(Regex("[^A-Za-z]")).filter { it.length > 2 }.groupingBy { it }.eachCount()