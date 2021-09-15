package edu.mcgill.gymfs.math

import edu.mcgill.gymfs.disk.*

fun String.allTokensDeduped(): List<String> =
  split(Regex("((?<=[^\\w])|(?=[^\\w]))"))

// Approximates CC metric without parsing
fun String.approxCyclomatic(
  tokens: List<String> = (openParens + closeParens).map { it.toString() }.toList() + controlFlowKeywords
) = tokens.mapLeftJoin(allTokensDeduped()).sumOf { it.second.size }