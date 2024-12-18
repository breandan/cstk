package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.automata.BAutomaton
import ai.hypergraph.kaliningraph.automata.FSATrajectory
import ai.hypergraph.kaliningraph.parsing.terminals
import ai.hypergraph.kaliningraph.parsing.Σᐩ
import ai.hypergraph.kaliningraph.repair.vanillaS2PCFG
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.to
import java.net.URL
import java.net.URLEncoder
import java.util.PriorityQueue
import java.util.concurrent.PriorityBlockingQueue
import kotlin.collections.plus
import kotlin.time.Duration
import kotlin.time.TimeSource

// Character-level LLM, transforms Python lexical tokens to ASCII characters
object MakeMore {
  object PyTokMap {
    val tm: Map<Σᐩ, Char> = vanillaS2PCFG.terminals.mapIndexed { i, it -> it to (33 + i).toChar() }.toMap()
    val mt: Map<Char, Σᐩ> = vanillaS2PCFG.terminals.mapIndexed { i, it -> (33 + i).toChar() to it }.toMap()
  }

  val MAKEMORE_URL = "http://localhost:8000/makemore/"

  fun callExternal(s: String): String =
    PyTokMap.mt[URL(MAKEMORE_URL + URLEncoder.encode(s, "utf-8")).readText().first()]!!

  fun encode(str: String) = str.tokenizeByWhitespace().map { PyTokMap.tm[it] }.joinToString("")
  fun nextTokens(str: String): List<Σᐩ> =
    URL("http://localhost:8000/makemore?next=" +
      URLEncoder.encode(str.tokenizeByWhitespace().map { PyTokMap.tm[it] }.joinToString(""), "utf-8")
    ).readText().tokenizeByWhitespace().map { PyTokMap.mt[it.first()]!! }

  fun checkSamples() {
    """
  "#"W"VwX!"#"="W"X!"#x!K"L"T!R"WwX!S"#"="W"VwX!
  "#YZ!K"L"WvXT!Rv!S!
  "#Y[wTwVwTwVwTw\V[wTwVwTwVwTw\V[wTwVwTwVwTwVwTw\Z!
  C"W"V${'$'}"XT!R"#"="WX="WwwXo"="W"#wV"#zX!"="WX!"#"="W"V"V"X!8"W"V"X="="W"V"X!S!
  K"L"T!R"#W"Y"Zo"YvZo"YvZo"X!S!
  "#pv!"#YZ!"#Y"="W"XZ!"#"="WYXK"L"WXZ!B"b"^"W"K"L"G"L"X!"T!R"="W"VwX!S!
  K"L"T!RK"L"T!R"#"Y"="WwXr"WYvZX!K"L"W"="WvXXT!R"#w!"="W"X!"="WYwZX!SSS!
  D"W"XT!RC"W"XT!R"W"V"X="WX!"WwX!SD"T!R"#v!SS"#"WwX!
  ;"<"V"!"#YwVwZ!"#"WX!K"L"T!R"#"="W"X!"="W"X!"="W"W"="XX!SK"L"="T!R"Y"V"Z'"="X!S!
    """.lines().map { it.trim().map { PyTokMap.mt[it] }.joinToString(" ") }.forEach { println(it) }
  }

  fun prepTrainingSet() {
    val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
    val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()

    contents.lines().asSequence().windowed(4, 4).map { it[0] to it[1] to it[2] to it[3] }
      .forEach { try { println((it.π2.tokenizeByWhitespace() + "NEWLINE")
        .map { PyTokMap.tm[it]!! }.joinToString("")) } catch (e: Exception) {} }
  }


  // Steers a random walk using the last n-1 transitions from the Markov Chain
  fun decodeDFA(
    bAutomaton: BAutomaton,
    // BAutomata uses a Unicode alphabet, and the Markov Chain recognizes a
    // string-based alphabet, so we need a way to translate between the two
    dec: Map<Char, Σᐩ>, // Maps unicode characters back to strings
    callback: (Σᐩ) -> Unit = {},
    timeout: Duration = Duration.INFINITE,
    beamWidth: Long = 1_000_000L, // Maximum number of trajectories to keep at each step
  ): List<Σᐩ> {
    val startTime = TimeSource.Monotonic.markNow()
    val fullTrajectories = PriorityBlockingQueue<FSATrajectory>(10000) // Max-heap for full trajectories
    val beam = PriorityQueue<FSATrajectory>() // Beam for partial trajectories

    beam.add(FSATrajectory(List(0) { null }, bAutomaton.initialState, 0.0))

    while (
      fullTrajectories.size < beamWidth &&
      beam.isNotEmpty() &&
      startTime.elapsedNow() < timeout
    ) {
      val nextBeam = beam.parallelStream().flatMap { partTraj ->
        val lastToks = partTraj.traj.reversed().filterNotNull()
        val t = partTraj.lastState.transitions.flatMap { next -> (next.min..next.max).map { tok -> dec[tok] to next } }.toMap()
        val nextTokens = try { if (t.size == 1) t.keys else nextTokens(lastToks.joinToString(" ")).filter { it in t } }
        catch (_: Exception) { listOf() }.ifEmpty { t.keys }

        var scores = nextTokens.mapIndexed { i, _ -> nextTokens.size.toDouble() - i }
        val sum = scores.sum()
        scores = scores.map { it.toDouble() / sum }

        nextTokens.mapIndexed { i, n -> partTraj.append(n, t[n]!!.dest, scores[i]) }
          .flatMap { traj ->
            if (traj.isComplete) {
              fullTrajectories.add(traj)
              callback(traj.toString())
              if (traj.lastState.transitions.isNotEmpty()) listOf(traj) else emptyList()
            } else { listOf(traj) }
          }.stream()
      }.sorted().limit(beamWidth).toList()

      beam.clear()
      beam.addAll(nextBeam)
    }

    val deduped = fullTrajectories.distinct().map { it.toString() }.toList()

    println("Took ${startTime.elapsedNow()} to decode ${deduped.size} trajectories, with ${beam.size} in queue")
    return deduped
  }
}