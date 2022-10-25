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

// Natural errors, unlocalized, can we get it to parse?

fun main() {
  val models = setOf(tidyparse)// + MODELS
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)
  val modelScores: Scores = models.associateWith { (0 to 0) }

  parsed!!.values.shuffled().asSequence()
    .map { it["code_string"].toString().let { it to it.parseOutput() } }
    .filter { (code, err) -> !code.hasBalancedBrackets() && err.containsBracketIssue() }
    .runningFold(modelScores) { scores, (code, originalError) ->
      models.associateWith { model ->
        val repair: List<String> = code.dispatchTo(model, cfg)
        scores[model]!!.let { (n, d) -> // numerator / denominator
          val parseOutput = repair.firstOrNull()?.parseOutput()
          if (model == tidyparse) diagnoseNaturalErrorUnlocalizedRepair(originalError, code, parseOutput, repair)
          if (parseOutput?.isEmpty() == true) (n + 1) to (d + 1) else n to (d + 1)
        }
      }
    }.forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

// "Premature optimization is the root of all evil." -Dijkstra

val tidyparse = Model("tidyparse")
val cfg =
  """S -> w | ( ) | [ ] | { } | ( S ) | [ S ] | { S } | S S"""
  .parseCFG().apply { blocked.addAll(setOf("w")) }

fun String.dispatchTo(model: Model, grammar: CFG?): List<String> =
  when (model) {
    tidyparse -> repair(this, grammar!!,
      String::coarsen, String::uncoarsen,
      synthesizer = { a -> grammar.asCSL.synthesize(*a.toTypedArray()) },
    )
    else -> { if (MSK in this) listOf(model.complete(replace(MSK, model.mask))) else emptyList() }
  }

fun String.containsBracketIssue(): Boolean = listOf("match").any { it in this }

fun String.parseOutput(): String =
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream
    .bufferedReader().readText().lines().first()

private fun diagnoseNaturalErrorUnlocalizedRepair(
  originalError: String,
  code: String,
  parseOutput: String?,
  repair: List<String>
) {
  println("""
Original error: $originalError

${code.lines().joinToString("\n") { "   $it" }}

${if(parseOutput?.isEmpty() == true && repair.isNotEmpty()) "Good Repair" else "Bad Repair: $parseOutput"}:

${code.lines().zip(repair.firstOrNull()?.lines() ?: listOf("(>>>No repair!<<<)"))
  .joinToString("\n") { (a, b) -> if (a == b) "   $b" else "** $b" }}

"""
  )
}

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