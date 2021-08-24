package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import info.debatty.java.stringsimilarity.Levenshtein
import kotlin.math.abs

fun main() {
//  val query = "System.out.$MSK"
//  println("Query: $query")
//  println("Completion: " + complete(query, 3))

  val validationSet = TEST_DIR.allFilesRecursively().allMethods().take(1000)
    .map { method ->
      val variant = method.renameTokens()
      val masked = variant.maskRandomIdentifier()
      if (variant == method) null else Triple(method, variant, masked)
    }.toList().mapNotNull { it }
//    .also { printOriginalVsTransformed(it) }

  val accuracy = validationSet.map { (original, refactored, masked) ->
    val completion = complete(masked)
    // TODO: better heuristic for correct selection
    if (Levenshtein().distance(completion, refactored).toInt() ==
      abs(completion.length - refactored.length)) 1.0 else 0.0
  }.average()

  println("Accuracy: $accuracy")
}

val defaultTokenizer = BasicTokenizer(false)
fun String.maskRandomIdentifier(): String =
//  defaultTokenizer.tokenize(this).let {
//    val rand = it.indices.random()
//    it.foldIndexed("") { i, acc, tok -> acc + if(i == rand) "<mask>" else tok }
//  }
  split(Regex("((?<=[^\\w])|(?=[^\\w]))")).let {
    val indexToMask = it.mapIndexed { index, s -> index to s }
      .filter { it.second.length > 1 && it.second.all { it.isJavaIdentifierPart() } }.random().also { println(it.second) }.first
    it.foldIndexed("") { i, acc, tok -> acc + if(i == indexToMask) "<mask>" else tok }
  }