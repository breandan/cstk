package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.hasBalancedBrackets
import ai.hypergraph.kaliningraph.parsing.*
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.disk.*
import java.io.File

/*
./gradlew bifiEval
 */

fun main() {
  val cfg = """S -> w | ( ) | [ ] | < > | { } | ( S ) | [ S ] | < S > | { S } | S S""".parseCFG()
  val json = File("bifi/data/orig_bad_code/orig.bad.json").readText()
  val parsed = Klaxon().parse<Map<String, Map<String, Any>>>(json)
  val tidyparse = Model("tidyparse")
  val modelScores: Map<Model, Pair<Int, Int>> =
    (MODELS + tidyparse).associateWith { (0 to 0) }

  parsed!!.values.asSequence()
    .map { it["code_string"].toString().let { it to it.parseError() } }
    .filter { (code, err) -> !code.hasBalancedBrackets() && err.containsBracketIssue() }
    .runningFold(modelScores) { scores, (code,err) ->
      (MODELS + tidyparse).associateWith { model ->
        if (model == tidyparse) {
          val coarsened = code.coarsen()
          val tokens = tokenize(coarsened)
          val (parseForest, stubs) = cfg.parseWithStubs(coarsened)
          val exclude = stubs.allIndicesInsideParseableRegions()
          val repairWorks = coarsened.findRepairs(
            cfg,
            exclude,
            fishyLocations = listOf(tokens.size),
            maxResults = 1
          ).isNotEmpty()

          // TODO: check whether Python parser actually accepts uncoarsened repair

          scores[tidyparse]!!.let { (n, d) -> // numerator / denominator
//            if (completion.hasBalancedBrackets())
            if (repairWorks) (n + 1) to (d + 1) else n to (d + 1)
          }
        } else {
          0 to 0
        }
      }
    }.forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

fun String.containsBracketIssue() =
  listOf("parenth").any { it in this}

fun String.parseError()=
  ProcessBuilder("python", "parser.py", this)
    .start().also { it.waitFor() }.inputStream.bufferedReader().readText()
    .substringBefore('(')

fun String.parses() = parseError().isEmpty()