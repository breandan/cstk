package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.automata.BAutomaton
import ai.hypergraph.kaliningraph.automata.FSATrajectory
import ai.hypergraph.kaliningraph.parsing.language
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
    val size = tm.size
  }

  val MAKEMORE_URL = "http://localhost:8000/makemore/"

  fun callExternal(s: String): String =
    PyTokMap.mt[URL(MAKEMORE_URL + URLEncoder.encode(s, "utf-8")).readText().first()]!!

  fun complete(str: String): String {
    val next = URL("http://localhost:8000/makemore?next=" + URLEncoder.encode(str, "utf-8"))
      .readText()
      .tokenizeByWhitespace()
      .firstOrNull()

    return if (next == "}") str + next else complete(str + next)
  }

  fun encode(str: String) = str.tokenizeByWhitespace().map { PyTokMap.tm[it] }.joinToString("")
  fun nextTokens(str: String): List<Σᐩ> =
    URL("http://localhost:8000/makemore?next=" +
      URLEncoder.encode(str.tokenizeByWhitespace().map { PyTokMap.tm[it] }.joinToString(""), "utf-8")
    ).readText().tokenizeByWhitespace().map { PyTokMap.mt[it.first()]!! }

  fun checkSamples() {
    """
C"W"XT!R"="#"WX!"="WX!"="#"WX!"="#"="="!"="#"="="!"="#"="="!S!
C"WXT!R"="WwVwV"#wX!"="WwV"#yX!"="WwVwX!;"="="<"!B"W"="Xbv!BwL"="!S!
C"W"XT!R"#"W"X!"#Y"Y"T"o"ZK"L"WvVpvVvXZ!S!
C"W"V"#xXT!Rw!"="#"="YwZ!"="WwVzX!"="WwVyX!G"T!R"="WwV"="WwXX!S"="WwX!"="WYQ"T"YvZV"="V"="V"="V"="V"YvZV"XZ!S!
D"T!Rw!C"W"XT!R"="#w!"="#w!"="#w!"="#w!"="#w!"="#w!"="#w!"="#w!SC"W"XT!R"="#w!"="#w!"="#w!"="#w!SS!
C"W"XT!Rw!"#"="WwX!B"W"Xbv!S!
C"W"#wV"#wV"#xXT!Rw!"#"="="W"="="W"XVwX!"#"W"V"X!"W"V"X="W"V"X!S!
C"W"V"XT!Rw!"#w="W"="X!"W"X!"="W"X!G_"="="W"XT!R"="W"X!SP"WX?"T!R"#"="WX!"W"X!"#"W"="K"L"X!G"T!R"#"="W"X!SS"="WX!S!
C"W"XT!R"#"Y"W"XTZ!JW"W"W"XXf"XT!R"V"#"V"!S8"!S!
    """.trimIndent().lines().map { it.trim().map { PyTokMap.mt[it] }.joinToString(" ") }
      .forEach { println(it); println(it in vanillaS2PCFG.language) }
  }

  fun checkCorruptedSamples() {
    """
      "V"V"#z!K"L"W"="W"XXT!R"#"="W"s"Vw="W"XX!"#"="="W"V"X!"="W"X!"="WX!S! 5 "V"V"#zT!K"L"W"="W"="XXT!R"#"="W"s"Vw="W"X"#"="="W"V"X"="W"X!S!
      C"W"XT!R"V"#"="W"X!G"bwT!R"="W"X!"="W"X!SIT!R"="W"X!SSZ! 6 C"W"XT!"V"#"="W"X!G"bwT!G"bwT!R"="W"X!"="WvX!SI!
      ;"<"!C"WXT!R"#v!"#"WX!"#"W"X!K"V"L"W"XT!R"'"WvV"X="WX!S8"="WX!S! 5 ;"<"!"WXT!R"#"!"WX!"#"W"X!K"V"L"W"XT!R"'"WvV"X="WX!S8"="WX!S!
      C"WXT!RK"L"="T!R"#"!"#"W"V"X!S"#"WQ"T"="X!"#"WX!S! 5 C"WXT!RK"L"="T!R"#"!"#""W"V"X!"#"WQ""T"="X!"#"WX!SS!
      D"W"XT!RC"W"XT!R"W"X!"="#v!SS"#"W"WXX!"#"W"X! 1 D"W"XT!RC"W"XT!R"W"X!"="#v!SS"#"W"WXX!"#"W"X!
      OT!R"#"="W"="W"#"XV"#"="X!SN"="T!R"#"="="W"#"V"#"="X!S! 1 OT!R"#"="W"="W"#""X"#"="XX!SN"="T!R"#"="="W"#"vV"#"="!S!
      C"W"V"XT!R"#"WYW"V"XK"L"ZK"L"="X!K"L"W"V"XT!R"#"W"V"#"W"XXW"X!8"!SS! 5 "W"V"XT!R"#"WYW"V"XK"L"ZK"L"="WXXT!RK"L""="W"XT!R"#"W"V"#"W"XXW"X!S8"!SS!
      "YwZ#w!"#Y"K"L"G"="W"XZ!"#x!K"L"T!RG"="W"XY"Zb"T!R"="W"X!SS! 2 "YwZ#w!"#Y"K"L"G"="W"XZ!"#x!K"L"T!RG"="W"XY"Zb"T!R"="W"X!SS!
      "#"W"="V"="V"X="W"="V"="VwX! 1 ""#"W"="V"="V"X="W"="V"="VwX!
      ;="<"!<"!"#"W[WwVwV[wTw\VwT[wTw\VwT[wTwVwTw\VwT[wTwVwTw\\V"#[wT[wTwV\VwT[wTwVwTw\\V[wT[wTy\VwTw\X! 1 <"="<"!<"!"#"W[WwVwV\V[wTw\VwT[wTw\VwT[wTwVwTw\VwT[wTwVwTw\\V"#[wT[wTw\VwT[wTwVwTw\\!V[wT[wTwVwTw\\ZXV!
    """.trimIndent().lines().forEach {
      it.trim().split(" ").let { (a, b, c) ->
        val fixed = a.map { PyTokMap.mt[it] }.joinToString(" ")
        val broke = c.map { PyTokMap.mt[it] }.joinToString(" ")
        val valid = "${fixed in vanillaS2PCFG.language} ${broke !in vanillaS2PCFG.language}"
        val eq = "$b = ${levenshtein(a.map { it }.joinToString(" "), c.map { it }.joinToString(" "))}"
        println("$fixed\n$broke\n$valid\n$eq")
      }
    }
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
    beamWidth: Long = 100L, // Maximum number of trajectories to keep at each step
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
      val nextBeam = try {
        beam.parallelStream().flatMap { partTraj ->
          if (startTime.elapsedNow() > timeout) throw Exception("Timeout!")
          val lastToks = partTraj.traj.reversed().filterNotNull()
          val txs = partTraj.lastState.transitions.flatMap { next -> (next.min..next.max).map { tok -> dec[tok] to next } }.toMap()
          val nextTokens = try { if (txs.size == 1) txs.keys else nextTokens(lastToks.joinToString(" ")).filter { it in txs } }
          catch (_: Exception) { listOf() }.ifEmpty { txs.keys }

          var scores = nextTokens.mapIndexed { i, _ -> nextTokens.size.toDouble() - i }
          val sum = scores.sum()
          scores = scores.map { it.toDouble() / sum }

          nextTokens.mapIndexed { i, n -> partTraj.append(n, txs[n]!!.dest, scores[i]) }
            .flatMap { traj ->
              if (traj.isComplete) {
                fullTrajectories.add(traj)
                callback(traj.toString())
                if (traj.lastState.transitions.isNotEmpty()) listOf(traj) else emptyList()
              } else { listOf(traj) }
            }.stream()
        }.sorted().limit(beamWidth).toList()
      } catch (_: Exception) { emptyList<FSATrajectory>() }

      beam.clear()
      beam.addAll(nextBeam)
    }

    val deduped = fullTrajectories.distinct().map { it.toString() }.toList()

    println("Took ${startTime.elapsedNow()} to decode ${deduped.size} trajectories, with ${beam.size} in queue")
    return deduped
  }
}