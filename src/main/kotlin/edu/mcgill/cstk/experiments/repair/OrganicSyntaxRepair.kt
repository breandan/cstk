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
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.*

/*
./gradlew organicSyntaxRepair
 */

// Natural errors, unlocalized, can we get it to parse?

@OptIn(ExperimentalTime::class)
fun main() {
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)

  val strbins: MutableMap<Int, MutableList<CodeSnippet>> = mutableMapOf()

  val minBinSize = 50
  parsed!!.values.shuffled().map { cs ->
    cs["code_string"].toString()
      .let { CodeSnippet(it, it.coarsen(), cs["msg"].toString()) }
  }.filter { it.coarsened.let { it.length in 22..69 } }
    .map { it.also { strbins.getOrPut(it.coarsened.length.bin10()) { mutableListOf() }.add(it) } }
    // Ensure each length category has at least n representatives
    .takeWhile { strbins.size < 5 || strbins.any { it.value.size < minBinSize } }
    .toList()

  MAX_TOKENS = 100
  MAX_SAMPLE = 200
  var pfxs = mutableListOf<String>()
  for (i in listOf(10000, 30000, 60000)) {
    TIMEOUT_MS = i.also { println("REEVALUATING TIMEOUT: $it ms") }
    val lenbins = ConcurrentHashMap<Int, Π3A<AtomicInteger>>()

    strbins.values.map { it.shuffled().take(minBinSize) }.flatten()
      .sortedBy { it.tokens.size }.parallelStream()
      .forEach { (code, coarsened, errMsg, groundTruth) ->
        val (proposed, accepted, total) =
          lenbins.getOrPut(coarsened.length.bin10()) {
              AtomicInteger(0) to
              AtomicInteger(0) to
              AtomicInteger(0)
          }

        val t = TimeSource.Monotonic.markNow()
        var totalValidSamples = 0
        val repair = repair(code, cfg,
          String::coarsen, String::uncoarsen,
          //      synthesizer = { a -> synthesize(a) },
          synthesizer = { a -> a.solve(this) },
          score = { defaultModel.score(it) },
        ).also { totalValidSamples = it.size.also { if (0 < it) proposed.incrementAndGet() } }
          .firstOrNull() ?: NO_REPAIR

        val parseOutput = repair.parsePythonOutput()
        if (parseOutput.isNotEmpty()) total.incrementAndGet()
        else listOf(total, accepted).forEach { it.incrementAndGet() }
        println("Drew $totalValidSamples samples before timeout")
        println("Synthesized repair in: ${t.elapsedNow().inWholeMilliseconds}ms")
        println("Tidyparse (proposed/total): ${proposed.get()}/${total.get()}")
        println("Tidyparse (accepted/proposed): ${accepted.get()}/${proposed.get()}")
        println("len,  10_s,   30_s,   60_s")
        println(lenbins.summarize(pfxs))
        diffNaturalErrorUnlocalizedRepair(errMsg, code, parseOutput, repair)
      }
    pfxs = lenbins.summarize(pfxs).lines().toMutableList()
  }
}

data class CodeSnippet(
  val originalCode: String,
  val coarsened: String,
  val errorMsg: String,
  val groundTruth: String? = null
) {
  val tokens = coarsened.split(" ")
}

const val NO_REPAIR = "NO_REPAIR_PROPOSAL!"

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
      synthesizer = { a -> a.solve(this) },
      filter = { isValidPython() }
    )
    else -> { if (MSK in this) listOf(model.complete(replace(MSK, model.mask))) else emptyList() }
  }

fun String.parsePythonOutput(): String =
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream
    .bufferedReader().readText().lines().first()

fun diffNaturalErrorUnlocalizedRepair(
  originalError: String,
  code: String,
  parseOutput: String,
  repair: String
) {
  println("""
Original error: $originalError

${prettyDiff(code, repair, rightHeading = "repair").ifEmpty { "...\n" }}
${if (repair == NO_REPAIR) "" else "Parser ${if (parseOutput.isEmpty()) "ACCEPTED repair!" else "REJECTED repair because: $parseOutput"}"}
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