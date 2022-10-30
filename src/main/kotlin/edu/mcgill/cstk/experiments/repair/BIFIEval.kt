package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.*
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.disk.Model
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/*
./gradlew bifiEval
 */

// Natural errors, unlocalized, can we get it to parse?

fun main() {
  val models = setOf(tidyparse)// + MODELS
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)
//  val modelScores: Scores = models.associateWith { (0 to 0) }
  val numerator = AtomicInteger(0)
  val denominator = AtomicInteger(0)

  MAX_TOKENS = 20
  MAX_SAMPLE = 100
  TIMEOUT_MS = 10000

  parsed!!.values.shuffled().parallelStream().forEach {
    val code = it["code_string"].toString()
    if(!code.hasBalancedBrackets()) {
      println(code)
      val repair = code.dispatchTo(tidyparse, cfg).firstOrNull()
      if (repair!= null) {
        val parseOutput = repair.parseOutput()
        if (parseOutput.isEmpty()) {
          numerator.incrementAndGet()
          denominator.incrementAndGet()
        } else {
          denominator.incrementAndGet()
        }

        diagnoseNaturalErrorUnlocalizedRepair("?", code, parseOutput, repair)
        println("Tidyparse (valid/total): ${numerator.get()}/${denominator.get()}")
      }
    }
  }
}

// "Premature optimization is the root of all evil." -Dijkstra

val tidyparse = Model("tidyparse")
val cfg =
  """S -> w | n | ( ) | [ ] | { } | ( S ) | [ S ] | { S } | S S"""
    .parseCFG().apply { blocked.addAll(setOf("w")) }

val cfg1 =
  """S -> w | n | ( ) | [ ] | { } | ( S ) | [ S ] | { S } | S S"""
    .parseCFG().apply { blocked.addAll(setOf("w")) }

val cfg2 =
  """S -> w | n | ( ) | [ ] | { } | ( S ) | [ S ] | { S } | S S"""
    .parseCFG().apply { blocked.addAll(setOf("w")) }


fun String.dispatchTo(model: Model, grammar: CFG?): List<String> =
  when (model) {
    tidyparse -> repair(this, grammar!!,
      String::coarsen, String::uncoarsen,
//      synthesizer = { a -> synthesize(a) },
      synthesizer = { a -> a.joinToString(" ").solve(this) }
    )
    else -> { if (MSK in this) listOf(model.complete(replace(MSK, model.mask))) else emptyList() }
  }

fun String.containsBracketIssue(): Boolean = listOf("match", "unbalanced").any { it in this }

fun String.parseOutput(): String =
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream
    .bufferedReader().readText().lines().first()

private fun diagnoseNaturalErrorUnlocalizedRepair(
  originalError: String,
  code: String,
  parseOutput: String?,
  repair: String?
) {
  println("""
Original error: $originalError

${code.lines().joinToString("\n") { "   $it" }}

${if(parseOutput?.isEmpty() == true) "Good Repair" else "Bad Repair: $parseOutput"}:

${code.lines().zip(repair?.lines() ?: listOf("(>>>No repair!<<<)"))
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