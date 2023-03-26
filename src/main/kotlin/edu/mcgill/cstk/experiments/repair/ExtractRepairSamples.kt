package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.hasBalancedBrackets
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.utils.*
import java.io.File

/*
./gradlew extractRepairSamples
 */

fun main() {
  var i = 0
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
//    .readLines().takeWhile { if (it == "  },") i++ < 20000 else true }
//    .joinToString("\n") + "\n}}"

  val goodCode = Klaxon().parse<Map<String, Map<String, Any>>>(json)

  goodCode!!.values.map { cs -> cs["code_string"].toString() }.asSequence()
    .flatMap { it.lines() }
    .filter { " = " in it }
    .filter { "\"" !in it && "'" !in it }
    .filter { '(' in it && '[' in it }
    .filter { selectionCriteria(it) }
    .map { it.trim() }
    .filter { it.length < 160 }
    .filter { !("$it\n").isValidPython() && "lambda" !in it }
    .filter { it.last() !in listOf(':', ',', '(', '[') }
    .toList()
    .forEach { print(it); it.isValidPython { println(":: ${it}") } }

  // https://gist.github.com/breandan/07688f41441591e311e18e504c45609c
}

private fun selectionCriteria(it: String) =
  it.isANontrivialStatementWithBalancedBrackets(1, statementCriteria = { true })