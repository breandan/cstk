package edu.mcgill.cstk.experiments.repair

import edu.mcgill.cstk.utils.lexAsKotlin

/*
./gradlew kotlinSyntaxRepair
 */

fun main() {
  """StringUtils.splitByCharacterTypeCamelCase(token).joinToString("") { old ->"""
      .lexAsKotlin().joinToString(" ").let { println(it) }
  """        listOf(getDirectHyponyms(sense), getDirectHypernyms(sense))) &&& """
    .lexAsKotlin().joinToString(" ").let { println(it) }
}