package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesize
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*
import java.io.File

/*
./gradlew bifiEval
 */

fun main() {
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)
  val modelScores: Scores = (MODELS + tidyparse).associateWith { (0 to 0) }

  parsed!!.values.asSequence()
    .map { it["code_string"].toString().let { it to it.parseOutput() } }
    .filter { (code, err) -> !code.hasBalancedBrackets() && err.containsBracketIssue() }
    .filter { it.first.lines().all { it.length < 180 } }
    .runningFold(modelScores) { scores, (code, originalError) ->
      (MODELS + tidyparse).associateWith { model ->
        val repair: List<String> = code.dispatchTo(model, cfg)
        scores[model]!!.let { (n, d) -> // numerator / denominator
          val parseOutput = repair.firstOrNull()?.parseOutput()
          if (model == tidyparse) printDiagnosis(originalError, code, parseOutput, repair)
          if (parseOutput?.isEmpty() == true) (n + 1) to (d + 1) else n to (d + 1)
        }
      }
    }.forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

private fun printDiagnosis(
  originalError: String,
  code: String,
  parseOutput: String?,
  repair: List<String>
) {
  println("""
Original error: $originalError

${code.lines().joinToString("\n") { "   $it" }}

${if(parseOutput?.isEmpty() == true || parseOutput == null) "Good Repair" else "Bad Repair, new error: $parseOutput"}:

${code.lines().zip(repair.firstOrNull()?.lines() ?: listOf("N/A"))
  .joinToString("\n") { (a, b) -> if (a == b) "   $b" else "** $b" }}

"""
  )
}

val tidyparse = Model("tidyparse")
val cfg = """S -> w | ( ) | [ ] | { } | ( S ) | [ S ] | { S } | S S"""
  .parseCFG().apply { blocked.addAll(setOf("w")) }

fun String.dispatchTo(model: Model, grammar: CFG?): List<String> =
  when (model) {
    tidyparse -> repair(this, grammar!!,
      String::coarsen, String::uncoarsen,
      synthesizer = { a -> synthesize(a) },
      blockers =  setOf("w", "<S>")
    )
    else -> { if (MSK in this) listOf(model.complete(replace(MSK, model.mask))) else emptyList() }
  }

fun String.containsBracketIssue(): Boolean = listOf("parenth").any { it in this }

fun String.parseOutput(): String =
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream
    .bufferedReader().readText().lines().first()

// TODO: Handle parentheses and move onto a new category of BIFI benchmark
//
// - Experiment should run and compare with BIFI
// - Devote 100% of energy to experiment pipeline
// - Look into why so many errors
//
// Fix parentheses errors and push to handle other categories
// Need to demonstrate usefulness on BIFI benchmark
// Review criteria: (Novelty, Significance, Usefulness) = Practice, (Soundness, Completeness) = Theory
// Benefits of using our technique vs others (e.g. lower cost)
// Finish evaluation, do not devote much time to other things.
// Draft needs to be very polished

// Example, introduction, experiments and results skeleton should be in paper draft
// - Method and related work (easy)
// - Introduction and motivating example (hard, spend more time)
    // How does it work? How does it compare to other methods?
    // Convince readers that it is not made-up problem