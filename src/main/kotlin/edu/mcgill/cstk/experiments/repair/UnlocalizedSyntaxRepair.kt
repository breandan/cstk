package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import astminer.parse.antlr.java.JavaParser
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.*
import java.util.concurrent.atomic.AtomicInteger
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
./gradlew unlocalizedSyntaxRepair
 */

@OptIn(ExperimentalTime::class)
fun main() {
  val models = setOf(tidyparse)
  val modelScores: Scores = models.associateWith { (0 to 0) }

  val proposed = AtomicInteger(0)
  val accepted = AtomicInteger(0)
  val total = AtomicInteger(0)

  MAX_TOKENS = 100
  MAX_SAMPLE = 100
  TIMEOUT_MS = 30000

  DATA_DIR.also { println("Evaluating syntax repair using $models on $it...") }
    .allFilesRecursively().allMethods()
    .map { it.first.lineSequence() }.flatten()
    .map { it.trim() }
    .filter(String::isANontrivialStatementWithBalancedBrackets)
    .filter { cfg.parse(it.coarsen()) != null }
    .mapNotNull {
      val prompt = it.constructPrompt().replace(MSK, "")
      val coarsened = prompt.coarsen().also { println("Coarsened: $it") }
      CodeSnippet(
        originalCode = prompt,
        coarsened = coarsened,
        errorMsg = "",
        groundTruth = it
      )
    }
    .take(100)
    .toList()
    .sortedBy { it.tokens.size }.parallelStream()
    .forEach { (code, coarsened, errMsg, groundTruth) ->
      val t = TimeSource.Monotonic.markNow()
      var totalValidSamples = 0
      val repair = code.dispatchTo(tidyparse, cfg).also {
        totalValidSamples = it.size.also { if (0 < it) proposed.incrementAndGet() }
      }.firstOrNull() ?: NO_REPAIR

      val parseOutput = repair.javac()
      if (parseOutput.isNotEmpty()) total.incrementAndGet()
      else listOf(total, accepted).forEach { it.incrementAndGet() }
      println("Drew $totalValidSamples samples before timeout")
      println("Synthesized repair in: ${t.elapsedNow().inWholeMilliseconds}ms")
      println("Tidyparse (proposed/total): ${proposed.get()}/${total.get()}")
      println("Tidyparse (accepted/proposed): ${accepted.get()}/${proposed.get()}")
      diffNaturalErrorUnlocalizedRepair(errMsg, code, parseOutput, repair)
    }
}