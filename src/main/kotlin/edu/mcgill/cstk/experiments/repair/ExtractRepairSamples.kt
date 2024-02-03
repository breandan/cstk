package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.hasBalancedBrackets
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.regex.Pattern

/*
./gradlew extractRepairSamples
 */

fun main() {
  var i = 0
  val json = File("/src/main/resources/datasets/python/bifi/data/orig_bad_code/orig.bad.json").readText()
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

fun readBIFIContents(
  good: Boolean = true,
  kind: String = if (good) "good" else "bad",
  filename: String = "src/main/resources/datasets/python/bifi/data/orig_${kind}_code/orig.${kind}.json",
  filenameCC: String = "/scratch/b/bengioy/breandan/bifi/data/orig_${kind}_code/orig.${kind}.cc.json",
  file: File = File(filename).let { if (it.exists()) it else File(filenameCC) }
): Sequence<String> =
  file.readLines().asSequence()
    .filter { it.trimStart().startsWith("\"code_string\": \"") }
    .mapNotNull {
      val json = "{$it}"
      val parsedObject = Klaxon().parseJsonObject(json.reader())
      parsedObject.string("code_string")
    }