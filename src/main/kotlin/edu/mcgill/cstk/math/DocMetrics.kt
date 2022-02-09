package edu.mcgill.cstk.math

import edu.mcgill.cstk.experiments.synonyms
import org.apache.commons.lang3.StringUtils

fun rouge(reference: Set<String>, candidate: Set<String>) =
  (reference intersect candidate).size.toDouble() / reference.size.toDouble().coerceAtLeast(1.0)

fun rougeSynonym(originalDoc: String, candidateDoc: String) =
  rouge(originalDoc.synonymCloud(), candidateDoc.synonymCloud())

fun String.synonymCloud(): Set<String> =
  StringUtils.splitByCharacterTypeCamelCase(this)
    .filter { it.all(Char::isLetter) }
    .map { it.synonyms() }
    .flatten().toSet()