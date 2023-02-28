package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.*
import kotlin.time.*

/**
 * Synthetic errors in natural data with unlocalized repair
 *
 * In this experiment, we sample nontrivial single-line statements with balanced
 * bracket from MiniGithub, delete a random bracket without telling the location
 * to the model, and ask it to predict the repair. If the ground truth is in the
 * repair set, it gets a 1 else 0.
 */

/*
./gradlew syntheticSyntaxRepair
 */

@OptIn(ExperimentalTime::class)
fun main() {
  val models = setOf(tidyparse)

//  val proposed = AtomicInteger(0)
//  val accepted = AtomicInteger(0)
//  val total = AtomicInteger(0)

  val strbins: MutableMap<Int, MutableList<CodeSnippet>> = mutableMapOf()

  val minBinSize = 50
  DATA_DIR.also { println("Evaluating synthetic syntax repair using $models on $it...") }
    .allFilesRecursively().allMethods()
    .map { it.first.lineSequence() }.flatten()
    .map { it.trim() }
    .filter(String::isANontrivialStatementWithBalancedBrackets)
    .filter { it.coarsen().let { it.length in 23..69 && cfg.parse(it) != null } }
    .map {
      val prompt = it.constructPrompt().replace(MSK, "")
      val coarsened = prompt.coarsen().also { println("Coarsened: $it") }
      println("Bin progress: " + strbins.entries.sortedBy { it.key }.joinToString(", "){ "${it.key} (${it.value.size})" })
      CodeSnippet(
        originalCode = prompt,
        coarsened = coarsened,
        errorMsg = "",
        groundTruth = it
      ).also { strbins.getOrPut(it.coarsened.length.bin10()) { mutableListOf() }.add(it) }
    }
    // Ensure each length category has at least n representatives
    .takeWhile { strbins.size < 5 || strbins.any { it.value.size < minBinSize } }
    .toList()

    MAX_TOKENS = 100
    MAX_SAMPLE = 100
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
            synthesizer = { a -> a.solve(this) }
          ).also {
            totalValidSamples =
              it.size.also { if (0 < it) proposed.incrementAndGet() }
          }.firstOrNull() ?: NO_REPAIR

          val parseOutput = repair.javac()
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

fun Int.bin10() = ((floor((this + 1).toDouble() / 10.0) * 10).toInt())

fun MutableMap<Int, Π3A<AtomicInteger>>.summarize(pfxs: MutableList<String>) =
  entries.sortedBy { it.key }.mapIndexed { i, (a, b) ->
    (if(pfxs.isEmpty()) "$a" else pfxs[i]) + ", " + (b.second.toDouble() / b.third.toDouble()).toString().take(5)
  }.joinToString("\n")