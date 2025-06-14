package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import com.sun.net.httpserver.*
import edu.mcgill.cstk.utils.lastGitMessage
import java.awt.Desktop
import java.io.File
import java.net.*
import java.net.http.*
import java.nio.charset.StandardCharsets
import java.time.Duration
import java.util.concurrent.*
import kotlin.math.absoluteValue
import kotlin.time.*
import kotlin.time.Duration.Companion.seconds

/**
cd ../tidyparse && ./gradlew bundleHeadless && cd ../cstk && ./gradlew -q wgpuBarHillelRepair 2>&1 | tee scripts/confounders.txt (optional)

./gradlew -q wgpuBarHillelRepair 2>&1 | tee scripts/so_ts.txt

// Evaluate neural reranker
scp breandan@narval.computecanada.ca:/home/breandan/projects/def-jinguo/breandan/cstk/scripts/num_reranker.pt scripts/
cd scripts && export PYTORCH_ENABLE_MPS_FALLBACK=1 && python reranker_serve.py
cd ../tidyparse && ./gradlew bundleHeadless && cd ../cstk && ./gradlew -q wgpuBarHillelRepair 2>&1 | tee cbifi_eval1.txt
*/
fun main() {
//  writeTrainingSetWithGPU()

//  writeCharBIFIToDisk()
  startWGPUServer()

  measureTime { evaluateWGPURepairOnStackOverflow() }.also { println("Finished evaluation in $it") }

  stopWGPUServer()

//  makeTSFromStackOverflow()
}

fun makeTSFromStackOverflow() {
  startWGPUServer()
  sizeAndDistBalancedRepairsUnminimized.mapNotNull { (brokeStr, fixedStr) ->
    val results = sendGPU(brokeStr).lines()
    if (fixedStr in results) listOf(brokeStr, fixedStr) + (results - fixedStr) else null
  }.map { it.map { it.charify() } }.forEach { println(it.joinToString("\n") + "\n") }
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
  val samplesBeforeMatchByLevDist = (1..MAX_RADIUS).associateWith { 0.0 }.toMutableMap()

  println("Running Matrix Bar-Hillel repair on Python snippets with $NUM_CORES cores")
  println("Sampling timeout: $TIMEOUT_MS ms, max tokens: $MAX_TOKENS, max radius: $MAX_RADIUS, max unique: $MAX_UNIQUE, CFG threshold: $CFG_THRESH")
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
    val levDist = levAlign.patchSize()

    val humanRepairANSI = levAlign.paintANSIColors()
    println("Source: ${brokeToks.joinToString(" ")}")
    println("Repair: $humanRepairANSI")

    val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL
    P_1ByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++
    P_AllByLevDist.getOrPut(lenBucket to levDist) { S2PMetrics() }.total++

    allRate.total++; levRates.getOrPut(levDist) { LBHMetrics() }.total++

    fun failed(msg: Σᐩ?, st: Σᐩ) {
      println("Encountered error $msg ${allTime.elapsedNow()}):\n$humanRepairANSI\n$st")
//      allRate.error++; levRates.getOrPut(levDist) { LBHMetrics() }.error++
//      println(allRate.toString())
//      negative.appendText("${brokeToks.size}, $levDist, 0\n")
//
//      println()
//      println("Precision@1\n===========")
//      println(P_1ByLevDist.summarizeLenAndDist())
//      println("Precision@All\n=============")
//      println(P_AllByLevDist.summarizeLenAndDist())
//      println()
    }

    try {
      val clock = TimeSource.Monotonic.markNow()
//      val rankedResults = initiateSerialRepair()
      val query = brokeToks.dropLast(1).joinToString(" ")
//      val cpuResults = sendCPU(brokeToks.dropLast(1).joinToString(" "))
      val docs = sendGPU(query)//parallelPythonRepair(brokeStr).joinToString("\n")
      val lines = docs.lines().map { it.addNewLineIfMissing() }
      println("Received ${lines.size} total docs in ${clock.elapsedNow()}")

      val idx = lines.indexOf(fixedStr)//; println("Idx: $idx")
      if (idx == -1 || idx > 1_000) throw Exception("Fixed string not found in GPU results")

      val rankedResults = lines.map { it.addNewLineIfMissing() }
//      val rankedResults = rerankGPU(query, lines.take(1_000).joinToString("\n"))
//    val rankedResults  rerankGPU(brokeToks.dropLast(1).joinToString(" "))
//        .also { println("Received ${it.size} ranked samples in ${clock.elapsedNow()}") }

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
        allRate.recall++; levRates.getOrPut(levDist) { LBHMetrics() }.recall++
        indexOfTarget.also { if (it == 0) { allRate.top1++; levRates.getOrPut(levDist) { LBHMetrics() }.top1++ } }
        println("Found length-$levDist repair in $elapsed ms, $indexOfTarget/${rankedResults.size} rank")
        allRate.run { println("Lev(*): $allRate") }; println(levRates.summarize())
        sampleTimeByLevDist[levDist] = (sampleTimeByLevDist[levDist] ?: 0.0) + elapsed
        println("Draw timings (ms): ${sampleTimeByLevDist.mapValues { it.value / (levRates[it.key]?.recall ?: 0) }}")
        samplesBeforeMatchByLevDist[levDist] = (samplesBeforeMatchByLevDist[levDist] ?: 0.0) + rankedResults.size
        println("Avg samples drawn: ${samplesBeforeMatchByLevDist.mapValues { it.value / (levRates[it.key]?.recall ?: 0) }}")
        positive.appendText("${brokeToks.size}, $levDist, $elapsed, $indexOfTarget, ${levAlign.summarize()}\n")
      }

      println()
      println("Precision@1\n===========")
      println(P_1ByLevDist.summarizeLenAndDist())
      println("Precision@All\n=============")
      println(P_AllByLevDist.summarizeLenAndDist())
      println()
    }
    catch (e: Exception) { e.printStackTrace(); failed(e.message, e.stackTraceToString()) }
    catch (e: Error) { e.printStackTrace(); failed(e.message, e.stackTraceToString()) }
  }
}

private const val PORT = 8000
private val page by lazy { File(System.getProperty("user.home"), "tidyparse.html").readText() }
private val streams = LinkedBlockingQueue<HttpExchange>()
@Volatile private var resultWaiter: CompletableFuture<String>? = null
private lateinit var server: HttpServer
private lateinit var trainingService: HttpServer

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

fun startTrainingService() {
  val seq = iterBIFIContents().map {
    val str = it.mapToUnquotedPythonTokens() + " NEWLINE"
    if (str in vanillaS2PCFG.language) str else null
  }.asSequence().filterNotNull().iterator()

  if (::trainingService.isInitialized) return
  trainingService = HttpServer.create(InetSocketAddress(PORT + 1), 0).apply {
    createContext("/fetch") { exchange ->
      var body: String
      while (true) {
        val query = seq.next()
        val reprs = sendGPU(query)
        if (reprs.isEmpty()) continue
        val qenc = query.charify()
        val renc = reprs.lines().map { it.charify() }.shuffled()
        body = renc.first() + "\n" + qenc + "\n" + renc.drop(1).joinToString("\n")
        break
      }
      exchange.sendResponseHeaders(200, body.length.toLong())
      exchange.responseBody.use { it.write(body.toByteArray()) }
    }
    executor = null
    start()
  }

  println("BIFI iterator service started at http://localhost:${PORT + 1}/fetch")
}

fun sendCPU(query: String): String =
  initiateSerialRepair(query.tokenizeByWhitespace(), vanillaS2PCFG).toSet()
    .map { it to P_BIFI_PY150.score(it.tokenizeByWhitespace()) }
    .sortedBy { it.second }.map { it.first }.take(65535).joinToString("\n")

fun TimeSource.Monotonic.ValueTimeMark.hasTimeLeft() = elapsedNow().inWholeMilliseconds < TIMEOUT_MS

fun writeValidationSet() {
  LED_BUFFER = 2
  P_BIFI_PY150.score(listOf("hello", "world"))
  sizeAndDistBalancedRepairsUnminimized.chunked(100) {
      it.parallelStream()
        .map { (b, f) -> b.addNewLineIfMissing() to f.addNewLineIfMissing() }
        .forEach { (broke, fixed) ->
          try {
            val query = broke
            val reprs = sendCPU(query)
            if (reprs.isNotBlank()) {
              val qenc = query.charify()
              val fenc = fixed.charify()
              val renc = reprs.lines().map { it.charify() } - fenc
              println(qenc + "\n" + fenc + "\n" + renc.joinToString("\n") + "\n")
            }
          } catch (_: Exception) {}
        }
    }.toList()
}

// Training set for reranker.py
fun writeTrainingSetWithMulticoreCPU() {
  LED_BUFFER = 3
  P_BIFI_PY150.score(listOf("hello", "world"))
  streamBIFIContents().map {
      val str = it.mapToUnquotedPythonTokens().addNewLineIfMissing()
      if (str in vanillaS2PCFG.language) str else null
    }.filter { it != null }.forEach {
      try {
        val query = it!!
        val reprs = sendCPU(query)
        if (reprs.isNotBlank()) {
          val qenc = query.charify()
          val renc = reprs.lines().map { it.charify() }.shuffled()
          println(renc.first() + "\n" + qenc + "\n" + renc.drop(1).joinToString("\n") + "\n")
        }
      } catch (_: Exception) {}
  }
}

// Training set for reranker.py
fun writeTrainingSetWithGPU() {
  startWGPUServer()

  iterBIFIContents().map {
    val str = it.mapToUnquotedPythonTokens().addNewLineIfMissing()
    if (str in vanillaS2PCFG.language) str else null
  }.filterNotNull().forEach {
      val query = it
      val reprs = sendGPU(query)
      if (reprs.isNotBlank()) {
        val qenc = query.charify()
        val renc = reprs.lines().map { it.charify() }.shuffled()
        println(renc.first() + "\n" + qenc + "\n" + renc.drop(1).joinToString("\n") + "\n")
      }
  }
}

fun sendGPU(query: String, timeoutSec: Long = 30) = try {
  val ex = streams.poll(timeoutSec, TimeUnit.SECONDS) ?: error("browser did not open /stream in time")
  resultWaiter = CompletableFuture()
  ex.responseHeaders.add("Content-Type", "text/event-stream")
  ex.sendResponseHeaders(200, 0)
  ex.responseBody.use { os -> os.write("retry: 0\ndata: $query\n\n".toByteArray()) }
  resultWaiter!!.get(timeoutSec, TimeUnit.SECONDS)
} catch (e: Exception) { "" }

fun String.charify() = "|${encodeToMakemore()}}"
fun String.uncharify() = removePrefix("|").removeSuffix("}").decodeFromMakemore()

fun rerankGPU(query: String, docs: String = sendGPU(query), url: String = "http://localhost:8082/rerank"): List<String> {
  if (docs.isEmpty()) return emptyList()
  val client = HttpClient.newBuilder().build()

  val query = query.charify()
  val docs = docs.lines().filter { it.isNotBlank() }
    .map { it.addNewLineIfMissing() }
    .map { it.charify() }
  val reqBody = "$query\n${docs.joinToString("\n")}"

  val request = HttpRequest.newBuilder().uri(URI.create(url))
    .timeout(Duration.ofMinutes(2))
    .POST(HttpRequest.BodyPublishers.ofString(reqBody)).build()

  val responseBody = try {
    client.send(request, HttpResponse.BodyHandlers.ofString()).body()
  } catch (e: HttpTimeoutException) {
    println("Error: Request to reranker timed out. ${e.message}")
    return emptyList()
  }

  return responseBody.lines().map { it.uncharify() }
}

fun stopWGPUServer() = if (::server.isInitialized) server.stop(0) else Unit
fun stopTrainingService() = if (::trainingService.isInitialized) trainingService.stop(0) else Unit

/*
w/ wGPU LBH, Markov Chain and MAX_LED=5

Lev(*): Top-1/rec/pos/total: 326 / 2677 / 7041 / 7044, errors: 3, P@1: 0.04628052243043725, P@All: 0.3800397501419648
Lev(1): Top-1/rec/pos/total: 258 / 2073 / 2169 / 2172, errors: 3, P@1: 0.11878453038674033, P@All: 0.9544198895027625
Lev(2): Top-1/rec/pos/total: 33 / 403 / 1909 / 1909, errors: 0, P@1: 0.017286537454164485, P@All: 0.21110529072812992
Lev(3): Top-1/rec/pos/total: 12 / 121 / 1064 / 1064, errors: 0, P@1: 0.011278195488721804, P@All: 0.1137218045112782
Lev(4): Top-1/rec/pos/total: 18 / 65 / 1123 / 1123, errors: 0, P@1: 0.016028495102404273, P@All: 0.0578806767586821
Lev(5): Top-1/rec/pos/total: 5 / 15 / 776 / 776, errors: 0, P@1: 0.006443298969072165, P@All: 0.019329896907216496
Draw timings (ms): {1=1068.1139335076577, 2=177.1258871871498, 3=45.29174449010086, 4=26.103100485618228, 5=8.396339185655584}
Full timings (ms): {1=1068.2607396339185, 2=177.14979454613373, 3=45.29958909226746, 4=26.10870377288009, 5=8.396712738139708}
Avg samples drawn: {1=107.48487112439298, 2=43.35674262233844, 3=10.301456854688084, 4=5.065745237205827, 5=1.0590212924915952}

Precision@1
===========
|σ|∈[0, 10): Top-1/total: 47 / 450 ≈ 0.10444444444444445
|σ|∈[10, 20): Top-1/total: 50 / 1094 ≈ 0.04570383912248629
|σ|∈[20, 30): Top-1/total: 42 / 1091 ≈ 0.0384967919340055
|σ|∈[30, 40): Top-1/total: 46 / 1102 ≈ 0.041742286751361164
|σ|∈[40, 50): Top-1/total: 39 / 1021 ≈ 0.03819784524975514
|σ|∈[50, 60): Top-1/total: 47 / 921 ≈ 0.051031487513572206
|σ|∈[60, 70): Top-1/total: 36 / 769 ≈ 0.04681404421326398
|σ|∈[70, 80): Top-1/total: 19 / 596 ≈ 0.031879194630872486
Δ(1)= Top-1/total: 258 / 2172 ≈ 0.11878453038674033
Δ(2)= Top-1/total: 33 / 1909 ≈ 0.017286537454164485
Δ(3)= Top-1/total: 12 / 1064 ≈ 0.011278195488721804
Δ(4)= Top-1/total: 18 / 1123 ≈ 0.016028495102404273
Δ(5)= Top-1/total: 5 / 776 ≈ 0.006443298969072165
(|σ|∈[0, 10), Δ=1): Top-1/total: 34 / 184 ≈ 0.18478260869565216
(|σ|∈[0, 10), Δ=2): Top-1/total: 6 / 133 ≈ 0.045112781954887216
(|σ|∈[0, 10), Δ=3): Top-1/total: 2 / 58 ≈ 0.034482758620689655
(|σ|∈[0, 10), Δ=4): Top-1/total: 5 / 42 ≈ 0.11904761904761904
(|σ|∈[0, 10), Δ=5): Top-1/total: 0 / 33 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 37 / 298 ≈ 0.12416107382550336
(|σ|∈[10, 20), Δ=2): Top-1/total: 6 / 296 ≈ 0.02027027027027027
(|σ|∈[10, 20), Δ=3): Top-1/total: 3 / 174 ≈ 0.017241379310344827
(|σ|∈[10, 20), Δ=4): Top-1/total: 3 / 197 ≈ 0.015228426395939087
(|σ|∈[10, 20), Δ=5): Top-1/total: 1 / 129 ≈ 0.007751937984496124
(|σ|∈[20, 30), Δ=1): Top-1/total: 33 / 292 ≈ 0.11301369863013698
(|σ|∈[20, 30), Δ=2): Top-1/total: 5 / 291 ≈ 0.01718213058419244
(|σ|∈[20, 30), Δ=3): Top-1/total: 2 / 182 ≈ 0.01098901098901099
(|σ|∈[20, 30), Δ=4): Top-1/total: 2 / 195 ≈ 0.010256410256410256
(|σ|∈[20, 30), Δ=5): Top-1/total: 0 / 131 ≈ 0.0
(|σ|∈[30, 40), Δ=1): Top-1/total: 37 / 296 ≈ 0.125
(|σ|∈[30, 40), Δ=2): Top-1/total: 7 / 290 ≈ 0.02413793103448276
(|σ|∈[30, 40), Δ=3): Top-1/total: 0 / 192 ≈ 0.0
(|σ|∈[30, 40), Δ=4): Top-1/total: 2 / 185 ≈ 0.010810810810810811
(|σ|∈[30, 40), Δ=5): Top-1/total: 0 / 139 ≈ 0.0
(|σ|∈[40, 50), Δ=1): Top-1/total: 31 / 290 ≈ 0.10689655172413794
(|σ|∈[40, 50), Δ=2): Top-1/total: 3 / 292 ≈ 0.010273972602739725
(|σ|∈[40, 50), Δ=3): Top-1/total: 4 / 138 ≈ 0.028985507246376812
(|σ|∈[40, 50), Δ=4): Top-1/total: 1 / 168 ≈ 0.005952380952380952
(|σ|∈[40, 50), Δ=5): Top-1/total: 0 / 133 ≈ 0.0
(|σ|∈[50, 60), Δ=1): Top-1/total: 37 / 290 ≈ 0.12758620689655173
(|σ|∈[50, 60), Δ=2): Top-1/total: 5 / 265 ≈ 0.018867924528301886
(|σ|∈[50, 60), Δ=3): Top-1/total: 0 / 132 ≈ 0.0
(|σ|∈[50, 60), Δ=4): Top-1/total: 2 / 145 ≈ 0.013793103448275862
(|σ|∈[50, 60), Δ=5): Top-1/total: 3 / 89 ≈ 0.033707865168539325
(|σ|∈[60, 70), Δ=1): Top-1/total: 34 / 278 ≈ 0.1223021582733813
(|σ|∈[60, 70), Δ=2): Top-1/total: 0 / 197 ≈ 0.0
(|σ|∈[60, 70), Δ=3): Top-1/total: 1 / 110 ≈ 0.00909090909090909
(|σ|∈[60, 70), Δ=4): Top-1/total: 1 / 112 ≈ 0.008928571428571428
(|σ|∈[60, 70), Δ=5): Top-1/total: 0 / 72 ≈ 0.0
(|σ|∈[70, 80), Δ=1): Top-1/total: 15 / 244 ≈ 0.06147540983606557
(|σ|∈[70, 80), Δ=2): Top-1/total: 1 / 145 ≈ 0.006896551724137931
(|σ|∈[70, 80), Δ=3): Top-1/total: 0 / 78 ≈ 0.0
(|σ|∈[70, 80), Δ=4): Top-1/total: 2 / 79 ≈ 0.02531645569620253
(|σ|∈[70, 80), Δ=5): Top-1/total: 1 / 50 ≈ 0.02

Precision@All
=============
|σ|∈[0, 10): Top-1/total: 202 / 450 ≈ 0.4488888888888889
|σ|∈[10, 20): Top-1/total: 386 / 1094 ≈ 0.35283363802559414
|σ|∈[20, 30): Top-1/total: 379 / 1091 ≈ 0.3473877176901925
|σ|∈[30, 40): Top-1/total: 402 / 1102 ≈ 0.3647912885662432
|σ|∈[40, 50): Top-1/total: 334 / 1021 ≈ 0.32713026444662097
|σ|∈[50, 60): Top-1/total: 391 / 921 ≈ 0.4245385450597177
|σ|∈[60, 70): Top-1/total: 332 / 769 ≈ 0.4317295188556567
|σ|∈[70, 80): Top-1/total: 251 / 596 ≈ 0.4211409395973154
Δ(1)= Top-1/total: 2073 / 2172 ≈ 0.9544198895027625
Δ(2)= Top-1/total: 403 / 1909 ≈ 0.21110529072812992
Δ(3)= Top-1/total: 121 / 1064 ≈ 0.1137218045112782
Δ(4)= Top-1/total: 65 / 1123 ≈ 0.0578806767586821
Δ(5)= Top-1/total: 15 / 776 ≈ 0.019329896907216496
(|σ|∈[0, 10), Δ=1): Top-1/total: 153 / 184 ≈ 0.8315217391304348
(|σ|∈[0, 10), Δ=2): Top-1/total: 31 / 133 ≈ 0.23308270676691728
(|σ|∈[0, 10), Δ=3): Top-1/total: 7 / 58 ≈ 0.1206896551724138
(|σ|∈[0, 10), Δ=4): Top-1/total: 11 / 42 ≈ 0.2619047619047619
(|σ|∈[0, 10), Δ=5): Top-1/total: 0 / 33 ≈ 0.0
(|σ|∈[10, 20), Δ=1): Top-1/total: 294 / 298 ≈ 0.9865771812080537
(|σ|∈[10, 20), Δ=2): Top-1/total: 51 / 296 ≈ 0.17229729729729729
(|σ|∈[10, 20), Δ=3): Top-1/total: 22 / 174 ≈ 0.12643678160919541
(|σ|∈[10, 20), Δ=4): Top-1/total: 14 / 197 ≈ 0.07106598984771574
(|σ|∈[10, 20), Δ=5): Top-1/total: 5 / 129 ≈ 0.03875968992248062
(|σ|∈[20, 30), Δ=1): Top-1/total: 291 / 292 ≈ 0.9965753424657534
(|σ|∈[20, 30), Δ=2): Top-1/total: 58 / 291 ≈ 0.19931271477663232
(|σ|∈[20, 30), Δ=3): Top-1/total: 24 / 182 ≈ 0.13186813186813187
(|σ|∈[20, 30), Δ=4): Top-1/total: 6 / 195 ≈ 0.03076923076923077
(|σ|∈[20, 30), Δ=5): Top-1/total: 0 / 131 ≈ 0.0
(|σ|∈[30, 40), Δ=1): Top-1/total: 295 / 296 ≈ 0.9966216216216216
(|σ|∈[30, 40), Δ=2): Top-1/total: 75 / 290 ≈ 0.25862068965517243
(|σ|∈[30, 40), Δ=3): Top-1/total: 23 / 192 ≈ 0.11979166666666667
(|σ|∈[30, 40), Δ=4): Top-1/total: 9 / 185 ≈ 0.04864864864864865
(|σ|∈[30, 40), Δ=5): Top-1/total: 0 / 139 ≈ 0.0
(|σ|∈[40, 50), Δ=1): Top-1/total: 262 / 290 ≈ 0.903448275862069
(|σ|∈[40, 50), Δ=2): Top-1/total: 51 / 292 ≈ 0.17465753424657535
(|σ|∈[40, 50), Δ=3): Top-1/total: 13 / 138 ≈ 0.09420289855072464
(|σ|∈[40, 50), Δ=4): Top-1/total: 7 / 168 ≈ 0.041666666666666664
(|σ|∈[40, 50), Δ=5): Top-1/total: 1 / 133 ≈ 0.007518796992481203
(|σ|∈[50, 60), Δ=1): Top-1/total: 288 / 290 ≈ 0.993103448275862
(|σ|∈[50, 60), Δ=2): Top-1/total: 73 / 265 ≈ 0.27547169811320754
(|σ|∈[50, 60), Δ=3): Top-1/total: 16 / 132 ≈ 0.12121212121212122
(|σ|∈[50, 60), Δ=4): Top-1/total: 8 / 145 ≈ 0.05517241379310345
(|σ|∈[50, 60), Δ=5): Top-1/total: 6 / 89 ≈ 0.06741573033707865
(|σ|∈[60, 70), Δ=1): Top-1/total: 274 / 278 ≈ 0.9856115107913669
(|σ|∈[60, 70), Δ=2): Top-1/total: 39 / 197 ≈ 0.19796954314720813
(|σ|∈[60, 70), Δ=3): Top-1/total: 12 / 110 ≈ 0.10909090909090909
(|σ|∈[60, 70), Δ=4): Top-1/total: 6 / 112 ≈ 0.05357142857142857
(|σ|∈[60, 70), Δ=5): Top-1/total: 1 / 72 ≈ 0.013888888888888888
(|σ|∈[70, 80), Δ=1): Top-1/total: 216 / 244 ≈ 0.8852459016393442
(|σ|∈[70, 80), Δ=2): Top-1/total: 25 / 145 ≈ 0.1724137931034483
(|σ|∈[70, 80), Δ=3): Top-1/total: 4 / 78 ≈ 0.05128205128205128
(|σ|∈[70, 80), Δ=4): Top-1/total: 4 / 79 ≈ 0.05063291139240506
(|σ|∈[70, 80), Δ=5): Top-1/total: 2 / 50 ≈ 0.04
*/