package edu.mcgill.cstk.experiments.repair

import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.utils.*
import java.io.File

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
  file: File = File(filename)
): Sequence<String> =
  file.readLines().asSequence()
    .filter { it.trimStart().startsWith("\"code_string\": \"") }
    .mapNotNull {
      val json = "{$it}"
      val parsedObject = Klaxon().parseJsonObject(json.reader())
      parsedObject.string("code_string")
    }

/* Download and decompress the py150 dataset:
   mkdir -p src/main/resources/datasets/python/py150 &&
   cd src/main/resources/datasets/python/py150 &&
   wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz &&
   tar -xvf py150_files.tar.gz &&
   tar -xvf data.tar.gz &&
   rm *.tar.gz
*/
fun readPY150Contents(
  prefix: String = "src/main/resources/datasets/python/py150",
  paths: List<String> = File("$prefix/python100k_train.txt").readLines().shuffled()
) =
  paths.asSequence().map { File("$prefix/$it").readText() }