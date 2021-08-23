package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*

fun main() {
  TEST_DIR.allFilesRecursively().allMethods().take(1000)
    .map { method ->
      val (variant, masked) = method.renameTokensAndMask()
      if (variant == method) null else method to variant
    }.toList().mapNotNull { it }
    .let { printOriginalVsTransformed(it) }
}

fun printOriginalVsTransformed(methodPairs: List<Pair<String, String>>) =
  methodPairs.forEach { (original, variant) ->
    if (original != variant) {
      val maxLen = 70
      val maxLines = 10
      val methodLines = original.lines()
      val variantLines = variant.lines()
      if (methodLines.all { it.length < maxLen } && methodLines.size < maxLines) {
        methodLines.forEachIndexed { i, l ->
          println(
            l.padEnd(maxLen, ' ') + "|    " +
              variantLines[i].padEnd(maxLen, ' ')
          )
        }
        println(List(maxLen * 2) { '=' }.joinToString(""))
      }
    }
  }