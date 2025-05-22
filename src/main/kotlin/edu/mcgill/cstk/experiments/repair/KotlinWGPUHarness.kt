package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import com.sun.net.httpserver.*
import edu.mcgill.cstk.utils.lastGitMessage
import java.awt.Desktop
import java.io.File
import java.net.*
import java.nio.charset.StandardCharsets
import java.util.concurrent.*
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/**
cd ../tidyparse && ./gradlew bundleHeadless && cd ../cstk && ./gradlew -q wgpuBarHillelRepair
*/
fun main() {
  startWGPUServer()

//  val list = sizeAndDistBalancedRepairsUnminimized.map { it.π1 }

//  measureTimeMillis {
//    for (snippet in list) println(send(snippet).lines().take(10).joinToString("\n") + "\n")
//  }.also { println("Total time: ${it / 1000.0}s") }

  evaluateWGPURepairOnStackOverflow()

  stopWGPUServer()
}

private fun evaluateWGPURepairOnStackOverflow() {
  MIN_TOKENS = 3
  MAX_TOKENS = 80
  MAX_RADIUS = 5

  val dataset = sizeAndDistBalancedRepairsUnminimized//corruptedBIFIGoodCode//sizeAndDistBalancedRepairsUnminimized.toList()
  // timeoutCases // corruptedBIFIGoodCode // balancedSmallRepairsUnminimized.toList() // naturallySmallRepairs //pairwiseUniformAll
  val allRate = LBHMetrics()
  val levRates = mutableMapOf<Int, LBHMetrics>()
  val sampleTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val allTimeByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()

  println("Running Matrix Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  println("Sampling timeout: $TIMEOUT_MS ms, max tokens: $MAX_TOKENS, " +
      "max radius: $MAX_RADIUS, max unique: $MAX_UNIQUE, CFG threshold: $CFG_THRESH")
  dataset.first().π2.let { P_BIFI_PY150.score(it.tokenizeByWhitespace()) }

  val latestCommitMessage = lastGitMessage().replace(Regex("[^A-Za-z0-9]"), "_")
    .let { if ("fatal: not a git repository" !in it) it else System.currentTimeMillis().toString() }
//    .replace(" ", "_").replace("/", "_")
  val positiveHeader = "length, lev_dist, sample_ms, total_ms, total_samples, rank\n"
  val negativeHeader = "length, lev_dist, samples\n"
  val title = "wgpu_bar_hillel"
  val positive = try { File("data/${title}_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/${title}_results_positive_$latestCommitMessage.csv").also { it.appendText(positiveHeader) } }
    .also { println("Writing positive CSV to: ${it.absolutePath}") }
  val negative = try { File("data/${title}_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
  catch (e: Exception) { File("/scratch/b/bengioy/breandan/${title}_results_negative_$latestCommitMessage.csv").also { it.appendText(negativeHeader) } }
    .also { println("Writing negative CSV to: ${it.absolutePath}") }
  println()

  val P_1ByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val P_AllByLevDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
  val timeout = (TIMEOUT_MS / 1000).seconds

  dataset.forEach { (brokeStr, fixedStr) ->
    val allTime = TimeSource.Monotonic.markNow()
    val brokeToks = brokeStr.tokenizeByWhitespace()
    val fixedToks = fixedStr.tokenizeByWhitespace()
    val levAlign = levenshteinAlign(brokeToks, fixedToks)

    val humanRepairANSI = levAlign.paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    allRate.total++; levRates.getOrPut(levDist) { LBHMetrics() }.total++

    fun failed(msg: Σᐩ?, st: Σᐩ) {
      println("Encountered error $msg ${allTime.elapsedNow()}):\n$humanRepairANSI\n$st")
      allRate.error++; levRates.getOrPut(levDist) { LBHMetrics() }.error++
      println(allRate.toString())
      negative.appendText("${brokeToks.size}, $levDist, 0\n")

      println()
      println("Precision@1\n===========")
      println(P_1ByLevDist.summarizeLenAndDist())
      println("Precision@All\n=============")
      println(P_AllByLevDist.summarizeLenAndDist())
      println()
    }

    try {
      val clock = TimeSource.Monotonic.markNow()
      val rankedResults = send(brokeStr).lines()
      val elapsed = clock.elapsedNow().inWholeMilliseconds

      val indexOfTarget = rankedResults.indexOf(fixedStr.addNewLineIfMissing()).also {
        if (it == 0) P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
        if (it != -1) P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.top1++
      }

      rankedResults.firstOrNull()?.tokenizeByWhitespace()
        ?.let { println("Top1 scoring repair: ${levenshteinAlign(brokeToks, it).paintANSIColors()}") }

      if (indexOfTarget < 0) {
        println("Drew ${rankedResults.size} samples in ${clock.elapsedNow()}/$timeout length-$levDist human repair not found")
        negative.appendText("${brokeToks.size}, $levDist, ${rankedResults.size}, ${levAlign.summarize()}\n")
      } else {
        val allElapsed = allTime.elapsedNow().inWholeMilliseconds

        allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
        indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
        println("Found length-$levDist repair in $elapsed ms, $allElapsed ms, $indexOfTarget rank")
        allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
        //      sampleTimeByLevDist[levDist] = sampleTimeByLevDist[levDist]!! + elapsed
        sampleTimeByLevDist[levDist] = (sampleTimeByLevDist[levDist] ?: 0.0) + elapsed
        println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / allRate.recall }}")
        allTimeByLevDist[levDist] = (allTimeByLevDist[levDist] ?: 0.0) + allElapsed
        println("Full timings (ms): ${allTimeByLevDist.mapValues { it.value / allRate.recall }}")
        samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + rankedResults.size
        println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / allRate.recall }}")
        positive.appendText("${brokeToks.size}, $levDist, $elapsed, $allElapsed, $indexOfTarget, ${levAlign.summarize()}\n")
      }

      println()
      println("Precision@1\n===========")
      println(P_1ByLevDist.summarizeLenAndDist())
      println("Precision@All\n=============")
      println(P_AllByLevDist.summarizeLenAndDist())
      println()
    }
    catch (e: Exception) { failed(e.message, e.stackTraceToString()) }
    catch (e: Error) { failed(e.message, e.stackTraceToString()) }
  }
}

private const val PORT = 8000
private val page by lazy { File(System.getProperty("user.home"), "tidyparse.html").readText() }
private val streams = LinkedBlockingQueue<HttpExchange>()
@Volatile private var resultWaiter: CompletableFuture<String>? = null
private lateinit var server: HttpServer

private fun HttpExchange.send(code: Int, mime: String, body: ByteArray) {
  responseHeaders.add("Content-Type", mime)
  sendResponseHeaders(code, body.size.toLong())
  responseBody.use { it.write(body) }
}

fun startWGPUServer() {
  if (::server.isInitialized) return
  server = HttpServer.create(InetSocketAddress(PORT), 0).apply {
    createContext("/") { it.send(200, "text/html", page.toByteArray()) }
    createContext("/stream") { ex -> streams.put(ex) }
    createContext("/result") { ex ->
      val txt = ex.requestBody.readAllBytes().toString(StandardCharsets.UTF_8).trim()
      resultWaiter?.complete(txt)
      ex.sendResponseHeaders(204, -1)
    }
    executor = null
    start()
  }

  Desktop.getDesktop().browse(URI("http://localhost:$PORT/"))
}

fun send(query: String, timeoutSec: Long = 30): String {
  val ex = streams.poll(timeoutSec, TimeUnit.SECONDS) ?: error("browser did not open /stream in time")
  resultWaiter = CompletableFuture()
  ex.responseHeaders.add("Content-Type", "text/event-stream")
  ex.sendResponseHeaders(200, 0)
  ex.responseBody.use { os -> os.write("retry: 0\ndata: $query\n\n".toByteArray()) }
  return resultWaiter!!.get(timeoutSec, TimeUnit.SECONDS)
}

fun stopWGPUServer() = if (::server.isInitialized) server.stop(0) else Unit