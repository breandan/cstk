package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.types.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.*
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.disk.Model
import edu.mcgill.cstk.utils.*
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.*

/*
./gradlew bifiEval
 */

// Natural errors, unlocalized, can we get it to parse?

@OptIn(ExperimentalTime::class)
fun main() {
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)
//  val modelScores: Scores = models.associateWith { (0 to 0) }
  val proposed = AtomicInteger(0)
  val accepted = AtomicInteger(0)
  val total = AtomicInteger(0)

  MAX_TOKENS = 100
  MAX_SAMPLE = 100
  TIMEOUT_MS = 30000

  parsed!!.values.shuffled()
    .map { cs -> cs["code_string"].toString().let { cs["msg"].toString() to it to it.coarsen() } }
    .filter { it.third.length < MAX_TOKENS && !it.second.hasBalancedBrackets() }
    // Sort by length so that parallel sketches all take roughly the same time
    .sortedBy { it.third.length }.parallelStream()
    .forEach { (errMsg, code, coarsened) ->
      val t = TimeSource.Monotonic.markNow()
      var totalValidSamples = 0
      val repair = code.dispatchTo(tidyparse, cfg)
        .also { totalValidSamples = it.size.also {
          if(0 < it) proposed.incrementAndGet()
        }
        }.firstOrNull() ?: "NO_REPAIR_PROPOSAL!"

      val parseOutput = repair.parseOutput()
      if (parseOutput.isNotEmpty()) total.incrementAndGet()
      else listOf(total, accepted).forEach { it.incrementAndGet() }
      println("Drew $totalValidSamples samples before timeout")
      println("Synthesized repair in: ${t.elapsedNow().inWholeMilliseconds}ms")
      println("Tidyparse (proposed/total): ${proposed.get()}/${total.get()}")
      println("Tidyparse (accepted/proposed): ${accepted.get()}/${proposed.get()}")
      diffNaturalErrorUnlocalizedRepair(errMsg, code, parseOutput, repair)
    }
}

// "Premature optimization is the root of all evil." -Dijkstra

val tidyparse = Model("tidyparse")
val cfg =
  """S -> w | ( ) | [ ] | { } | ( S ) | [ S ] | { S } | S S"""
    .parseCFG().apply { blocked.addAll(setOf("w")) }

fun String.coarsenAsPython(): String =
  tokenizeAsPython().joinToString(" ") {
    when {
      it.isBracket() -> it
      else -> "w"
    }
  }

fun String.dispatchTo(model: Model, grammar: CFG?): List<String> =
  when (model) {
    tidyparse -> repair(this, grammar!!,
      String::coarsen, String::uncoarsen,
//      synthesizer = { a -> synthesize(a) },
      synthesizer = { a -> a.solve(this) }
    ) { isValidPython() }

    else -> { if (MSK in this) listOf(model.complete(replace(MSK, model.mask))) else emptyList() }
  }

fun String.containsBracketIssue(): Boolean = listOf("match", "unbalanced").any { it in this }

fun String.parseOutput(): String =
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream
    .bufferedReader().readText().lines().first()

private fun diffNaturalErrorUnlocalizedRepair(
  originalError: String,
  code: String,
  parseOutput: String,
  repair: String?
) {
  println("""
Original error: $originalError

${if(repair == null) "(>>>No repair!<<<)"
  else prettyDiff(code, repair, maxLen = 77, rightHeading = "repair").ifEmpty { "...\n" }}
Repair was ${if(parseOutput.isEmpty()) "ACCEPTED" else "REJECTED"} by Python parser!
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