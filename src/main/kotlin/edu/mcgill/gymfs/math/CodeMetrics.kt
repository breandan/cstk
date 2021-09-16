package edu.mcgill.gymfs.math

import edu.mcgill.gymfs.nlp.*
import info.debatty.java.stringsimilarity.Levenshtein
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import org.apache.commons.lang3.StringUtils

fun String.allTokensDeduped(): List<String> =
  split(Regex("((?<=[^\\w])|(?=[^\\w]))"))

// Approximates CC metric without parsing
fun String.approxCyclomatic(
  tokens: List<String> = (openParens + closeParens).map { it.toString() }.toList() + controlFlowKeywords
) = tokens.mapLeftJoin(allTokensDeduped()).sumOf { it.second.size }


object MetricCSNF: MetricStringDistance {
  /**
   * NF1, NF2 := CSNF(S1 + S2)
   * CSNFΔ(SN1, SN2) := LEVΔ(NF1, NF2)
   */
  override fun distance(s1: String, s2: String) =
    codeSnippetNormalForm(s1 to s2).let { (a, b) -> Levenshtein().distance(a, b) }

  fun codeSnippetNormalForm(pair: Pair<String, String>): Pair<String, String> =
    (StringUtils.splitByCharacterTypeCamelCase(pair.first).toList() to
      StringUtils.splitByCharacterTypeCamelCase(pair.second).toList()).let { (c, d) ->
      val vocab = (c.toSet() + d.toSet()).mapIndexed { i, s -> s to i }.toMap()
      c.map { vocab[it] }.joinToString("") to d.map { vocab[it] }.joinToString("")
    }
}

fun main() {
  println(kantorovich(matrixize("test  a ing 123"), matrixize("{}{{}{{{}}}{asdf g")))
  println(kantorovich(matrixize("test  a ing 123"), matrixize("open ing 222")))
}