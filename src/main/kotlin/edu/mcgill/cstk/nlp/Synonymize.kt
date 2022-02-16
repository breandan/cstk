package edu.mcgill.cstk.nlp

import edu.mcgill.cstk.disk.RESERVED_TOKENS
import net.sf.extjwnl.data.PointerUtils.*
import net.sf.extjwnl.dictionary.Dictionary
import org.apache.commons.lang3.StringUtils

fun synonymize(token: String): String =
  StringUtils.splitByCharacterTypeCamelCase(token).joinToString("") { old ->
    old.synonyms().filter { it !in RESERVED_TOKENS }.ifEmpty { setOf(old) }
      .random().let { new ->
        if (old.first().isLowerCase()) new
        else "" + new[0].uppercaseChar() + new.drop(1)
      }
  }

val defaultDict: Dictionary = Dictionary.getDefaultResourceInstance()

// Returns single-word synonyms
fun String.synonyms(synonymDepth: Int = 3): Set<String> =
  defaultDict.lookupAllIndexWords(this).indexWordArray.map {
    it.senses.map { sense ->
      (getSynonymTree(sense, synonymDepth).toList() +
        listOf(getDirectHyponyms(sense), getDirectHypernyms(sense)))
        .flatten().map { it.synset.words }
        .flatten().mapNotNull { it.lemma }
    }.flatten() + it.lemma
  }.flatten().filter { " " !in it }.toSet()