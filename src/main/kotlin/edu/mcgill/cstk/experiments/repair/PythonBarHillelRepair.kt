package edu.mcgill.cstk.experiments.repair

import NUM_CORES
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import edu.mcgill.cstk.utils.lexToStrTypesAsPython
import java.io.File
import kotlin.random.Random
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/*
./gradlew pythonBarHillelRepair
 */
fun main() {
  // Perfect recall on first 20 repairs takes ~7 minutes on a 2019 MacBook Pro
  var errorRate = 0
  var (recall, total) = 0 to 0
  val sampleTimeByLevDist = mutableMapOf(1 to 0.0, 2 to 0.0, 3 to 0.0)
  val allTimeByLevDist = mutableMapOf(1 to 0.0, 2 to 0.0, 3 to 0.0)
  val samplesBeforeMatchByLevDist = mutableMapOf(1 to 0.0, 2 to 0.0, 3 to 0.0)
  val s2pg = vanillaS2PCFG
  val currentTime = System.currentTimeMillis()
  val positiveHeader = "length, lev_dist, sample_ms, total_ms, total_samples, lev_ball_arcs, productions, edit1, edit2, edit3\n"
  val positive = try { File("bar_hillel_results_positive_$currentTime.csv").also { it.appendText(positiveHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/bar_hillel_results_positive_$currentTime.csv").also { it.appendText(positiveHeader) } }
  val negativeHeader = "length, lev_dist, samples, productions, edit1, edit2, edit3\n"
  val negative = try { File("bar_hillel_results_negative_$currentTime.csv").also { it.appendText(negativeHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/bar_hillel_results_negative_$currentTime.csv").also { it.appendText(negativeHeader) } }
  println("Running Bar-Hillel repair on Python snippets with $NUM_CORES cores")

  invalidLexedPythonStatements.lines().zip(validLexedPythonStatements.lines())
    .shuffled(Random(1)).forEach { (invalid, valid) ->
      val allTime = TimeSource.Monotonic.markNow()
      val toRepair = "$invalid NEWLINE".tokenizeByWhitespace()
      val humanRepair = "$valid NEWLINE".tokenizeByWhitespace()
      val target = humanRepair.joinToString(" ")
      val levAlign = levenshteinAlign(toRepair, humanRepair)
      val levDist = levAlign.patchSize()

      val levBall = makeLevFSA(toRepair, levDist)
      val humanRepairANSI = levenshteinAlign(toRepair, humanRepair).paintANSIColors()
      val intGram = try { s2pg.jvmIntersectLevFSA(levBall) }
        catch (e: Exception) {
          println("Encountered error (${e.message}): $humanRepairANSI")
          println("Recall: $recall / $total, errors: ${++errorRate}\n")
          return@forEach
        }

      total++
      assert(humanRepair in s2pg.language)
      assert(levBall.recognizes(humanRepair))
      assert(humanRepair in intGram.language)
      println("Ground truth repair: $humanRepairANSI")
      val clock = TimeSource.Monotonic.markNow()
      var samplesBeforeMatch = 0
      var matchFound = false
      val timeout = 120.seconds
      val results = mutableListOf<Σᐩ>()
      run untilDone@{
        intGram.sampleDirectlyWR(stoppingCriterion = { clock.elapsedNow() < timeout })
          .distinct().forEach {
            results.add(it)
            samplesBeforeMatch++
            if (it == target) {
              matchFound = true
              return@untilDone
  //            } else {
  //              val ascii = levenshteinAlign(toRepair, it.tokenizeByWhitespace()).paintANSIColors()
  //              println("Found valid repair (${clock.elapsedNow()}): $ascii")
            }
          }
      }

      if (!matchFound) {
        println("Drew $samplesBeforeMatch samples in $timeout, ${intGram.size} prods, length-$levDist human repair not found")
        negative.appendText("${toRepair.size}, $levDist, $samplesBeforeMatch, " +
          "${levBall.Q.size}, ${intGram.size}, ${levAlign.summarize()}\n")
      } else {
        val elapsed = clock.elapsedNow().inWholeMilliseconds
        val allElapsed = allTime.elapsedNow().inWholeMilliseconds
//        val rankedResults = results.map { it to P_BIFI.score(it.mapToBIFIFmt()) }.sortedBy { it.second }.map { it.first }
        println("Found human repair (${clock.elapsedNow()}): $humanRepairANSI")
        println("Found length-$levDist repair in $elapsed ms, $allElapsed ms, $samplesBeforeMatch samples, ${intGram.size} prods")//, rank: ${rankedResults.indexOf(target) + 1} / ${rankedResults.size}")
        println("Recall / samples : ${++recall} / $total, errors: $errorRate")
        sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
        println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / recall }}")
        allTimeByLevDist[levDist] = allTimeByLevDist[levDist]!! + allElapsed
        println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / recall }}")
        samplesBeforeMatchByLevDist[levDist] = samplesBeforeMatchByLevDist[levDist]!! + samplesBeforeMatch
        println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / recall }}")
        positive.appendText("${toRepair.size}, $levDist, $elapsed, $allElapsed, " +
          "$samplesBeforeMatch, ${levBall.Q.size}, ${intGram.size}, ${levAlign.summarize()}\n")
      }

      println()
    }
}

fun Σᐩ.mapToBIFIFmt() =
  tokenizeByWhitespace().dropLast(1).joinToString(" ", "BOS", "EOS").tokenizeByWhitespace()
