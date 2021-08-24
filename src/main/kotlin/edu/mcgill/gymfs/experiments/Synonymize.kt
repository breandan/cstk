package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import net.sf.extjwnl.data.PointerUtils.*
import net.sf.extjwnl.dictionary.Dictionary
import org.apache.commons.lang3.StringUtils

fun main() {
  TEST_DIR.allFilesRecursively().allMethods().take(1000)
    .map { method ->
      val variant = method.renameTokens()
      if (variant == method) null else method to variant
    }.toList().mapNotNull { it }
    .let { printOriginalVsTransformed(it) }
}

fun synonymize(token: String): String =
  StringUtils.splitByCharacterTypeCamelCase(token).joinToString("") { old ->
    (synonyms(old) + old).random().let { new ->
      if (old.first().isLowerCase()) new
      else "" + new[0].uppercaseChar() + new.drop(1)
    }
  }

// Returns single-word synonyms
fun synonyms(
  word: String,
  synonymDepth: Int = 3,
  dict: Dictionary = Dictionary.getDefaultResourceInstance()
): Set<String> =
  dict.lookupAllIndexWords(word).indexWordArray.map {
    it.senses.map { sense ->
      (getSynonymTree(sense, synonymDepth).toList() +
        listOf(getDirectHyponyms(sense), getDirectHypernyms(sense)))
        .flatten().map { it.synset.words }
        .flatten().mapNotNull { it.lemma }
    }.flatten()
  }.flatten().filter { " " !in it }.toSet()

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