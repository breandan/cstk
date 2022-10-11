package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesize
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.complete
import java.io.File

/*
./gradlew bifiEval
 */

fun main() {
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)
  val modelScores: Scores = (MODELS + tidyparse).associateWith { (0 to 0) }

  parsed!!.values.asSequence()
    .map { it["code_string"].toString().let { it to it.parseError() } }
    .filter { (code, err) -> !code.hasBalancedBrackets() && err.containsBracketIssue() }
    .runningFold(modelScores) { scores, (code, err) ->
      (MODELS + tidyparse).associateWith { model ->
        val repairs = code.dispatchTo(model, cfg)
        // TODO: check whether Python parser actually accepts uncoarsened repair
        scores[model]!!.let { (n, d) -> // numerator / denominator
          if (repairs.isEmpty()) (n + 1) to (d + 1) else n to (d + 1)
        }
      }
    }.forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

val tidyparse = Model("tidyparse")
val cfg = """S -> w | ( ) | [ ] | < > | { } | ( S ) | [ S ] | < S > | { S } | S S""".parseCFG()

fun String.dispatchTo(model: Model, grammar: CFG?): List<String> =
  when (model) {
    tidyparse -> repair(this, grammar!!, String::coarsen, String::uncoarsen, synthesizer = { a, b -> cfg.synthesize(a, b) })
    else -> { if(MSK in this) listOf(model.complete(replace(MSK, model.mask))) else emptyList() }
  }

fun String.containsBracketIssue(): Boolean = listOf("parenth").any { it in this }

fun String.parseError(): String =
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream.bufferedReader().readText()
    .substringBefore('(')

fun String.parses() = parseError().isEmpty()