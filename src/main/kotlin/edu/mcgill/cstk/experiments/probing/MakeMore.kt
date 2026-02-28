package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.automata.BAutomaton
import ai.hypergraph.kaliningraph.automata.FSATrajectory
import ai.hypergraph.kaliningraph.automata.options
import ai.hypergraph.kaliningraph.automata.toDFA
import ai.hypergraph.kaliningraph.parsing.language
import ai.hypergraph.kaliningraph.parsing.terminals
import ai.hypergraph.kaliningraph.parsing.Σᐩ
import ai.hypergraph.kaliningraph.repair.MAX_DFA_IN
import ai.hypergraph.kaliningraph.repair.s2pg
import ai.hypergraph.kaliningraph.repair.vanillaS2PCFG
import ai.hypergraph.kaliningraph.repair.vanillaS2PCFGWE
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.to
import ai.hypergraph.kaliningraph.types.Π2A
import edu.mcgill.cstk.experiments.repair.sizeAndDistBalancedRepairsUnminimized
import java.io.File
import java.net.URL
import java.net.URLEncoder
import java.util.PriorityQueue
import java.util.concurrent.PriorityBlockingQueue
import kotlin.collections.plus
import kotlin.time.Duration
import kotlin.time.TimeSource
import kotlin.time.measureTimedValue

// Character-level LLM, transforms Python lexical tokens to ASCII characters
object MakeMore {
  // BIFI dataset vocab, n.b. does not include entire vanillaS2PCFG alphabet due to rare tokens, e.g., <>
  val vocab = " !\"#\$%'()+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdfhklmnopqrstuvwxyz|}~"
  val terms = vocab.map { decode("$it") }.toSet()

  object PyTokMap {
    val tm: Map<Σᐩ, Char> = s2pg.terminals.mapIndexed { i, it -> it to (33 + i).toChar() }.toMap()
    val mt: Map<Char, Σᐩ> = s2pg.terminals.mapIndexed { i, it -> (33 + i).toChar() to it }.toMap()
    val size = tm.size
  }

  val MAKEMORE_URL = "http://localhost:8000/makemore?next="

  fun complete(str: String): String {
    val logprobs = nextTokensAndScores(str)
//    logprobs.forEach { (a, b) -> println("$a $b") }
    val next = logprobs.first().first

//    println(str.map { PyTokMap.mt[it] ?: it }.joinToString(" "))
    return if (next == "}") str + next else complete(str + next)
  }

  fun encode(str: String) = str.tokenizeByWhitespace().map { PyTokMap.tm[it] }.joinToString("")
  fun decode(str: String) = str.map { PyTokMap.mt[it] }.joinToString(" ")

  private fun req(str: String) = URL(MAKEMORE_URL + URLEncoder.encode(str, "utf-8")).readText()

  fun nextTokens(str: String): List<Σᐩ> =
    req(str).lines().map {
      val t = it.tokenizeByWhitespace()
      if (t.size < 2) " " else t.first()
    }

  fun nextTokensAndScores(str: String): List<Pair<Σᐩ, Double>> =
    req(str).lines().filter { it.split(" ").size == 2 }.mapNotNull {
      val (a, b) = it.split(" ").let { (a, b) -> PyTokMap.mt[a.first()] to b.toDouble() }
      if (a == null) null else a to b
    }

  fun nextTokensReadable(str: String): List<Σᐩ> = nextTokens(str).map { PyTokMap.mt[it.first()]!! }

  fun predDist(str: String) = nextTokens("$str ").first { it.first().isDigit() && it != "0" }.toInt()

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

  fun checkEditRadiusPred(broken: String): Int =
    nextTokens("|" + encode(broken) + " ").first { it.first().isDigit() }.toInt()

  fun checkPairwiseSamples() {
    """
      C"W"V"V"#xXT!Rw!G_"T!R8xVx!S"#"WX!"="W"V"X!"#"="W"X!"="W"="X!8w="WYwVwVw="ws"Y"ZK"L"ZX!S! 2 C"W"V"V"#xXT!Rw!G_"T!R8xVx!S"#"WX!"="W"V"X!"#"="W"X!"="W"="X!8w="WYwVwVws"Vws"Y"ZK"L"ZX!S!
      C"W"XT!R"="="WyX!"="W"="="WXX!R="="WzX!"="W"="="WXX!"="W"="="WXX!S! 1 C"W"XT!R"="="WyX!"="W"="="WXX!"="="WzX!"="W"="="WXX!"="W"="="WXX!S!
      C"WXT!RG_"="="WwXT!R8z!S"#"WX!8"!SS 1 C"WXT!RG_"="="WwXT!R8z!S"#"WX!8"!S!
      C"W"XT!Rw!8""!"#vX!S! 2 C"W"XT!Rw!8"W"#"V"#vX!S!
      C"W"V"V"V"V"XT!Rw!G"c"="]_"="]"`v]"c"="]"`vT!R8x!SIT!R8"="WW"V"V"V"V"#wG"="b"="IwXV"X!S! 1 C"W"V"V"V"V"XT!Rw!G"c"="]_"="]"`v]"c"="]"`vT!R8x!SIT!R8"="WW"V"V"V"V"#wG"="b"="IwXV"X!SS!
      C"W"V"XT!R"#v!J"a"T!RG"b"T!R"'v!"(v!SIT!RG"b"T!R8!SH"`"T!R"#"!SIT!R"#"!6!S8"!SSJ"W"p"XcvT!RG_"s"fxT!R""!"("!SSS! 1 C"W"V"XT!R"#v!J"a"T!RG"b"T!R"'v!"(v!SIT!RG"b"T!R8!SH"`"T!R"#"!SIT!R"#"!6!S8"!SSJ"W"p"XcvT!RG_"s"fxT!R"#"!"("!SSS!
      D"W"XT!R"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!n#v!"#v!R#v!"#v!"#v!"#v!S! 2 D"W"XT!R"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!"#v!S!
      ;"<"T"#"#v!S#v!! 3 ;"<"!"#"#v!"#v!!
      C"W"V"V"#YZV"#zXT5R;"="="="<"V"aG"^"T!R"V"#"!S8"="W"V"V"X!S! 2 C"W"V"V"#YZV"#zXT!R;"="="="<"V"!G"^"T!R"V"#"!S8"="W"V"V"X!S!
    """.trimIndent().lines().forEach {
      it.trim().split(" ").let { (a, b, c) ->
        val fixed = c.map { PyTokMap.mt[it] }.joinToString(" ")
        val broke = a.map { PyTokMap.mt[it] }.joinToString(" ")
        val valid = "${fixed in vanillaS2PCFG.language} ${broke !in vanillaS2PCFG.language}"
        val eq = "$b = ${levenshtein(a.map { it }.joinToString(" "), c.map { it }.joinToString(" "))}"
        println("$fixed\n$broke\n$valid\n$eq")
      }
    }
  }

  fun previewSamples() {
    File("synt_fixes.txt").readLines().filter { it.length < 30 }
      .take(100_000).shuffled().toList().take(10)
      .also { it.forEach { println(it) } }
      .map { it.tokenizeByWhitespace() }
      .map { decode(it[0]) to " ${it[1]} " to decode(it[2]) }
      .forEach { println((it.first + it.second + it.third).replace("null", "")) }
  }

  fun prepTrainingSet() {
    val filename = "datasets/python/stack_overflow/naturally_small_repairs_unminimized_base64.txt"
    val contents = object {}.javaClass.classLoader.getResource(filename)!!.readText()

    contents.lines().asSequence().windowed(4, 4).map { it[0] to it[1] to it[2] to it[3] }
      .forEach { try { println((it.π2.tokenizeByWhitespace() + "NEWLINE")
        .map { PyTokMap.tm[it]!! }.joinToString("")) } catch (e: Exception) {} }
  }

  fun measureRankOfTrueNextTokenWithSyntaxConstraints() { TODO() }

  fun measureRankOfTrueNextTokenWithLBHConstraints() {
    val s2pg = vanillaS2PCFG
    val parikhMap = s2pg.parikhMap
    val termDict = TermDict(s2pg.terminals)

    var crankTot = 0.0
    var urankTot = 0.0
    var instances = 0

    sizeAndDistBalancedRepairsUnminimized.forEach { (broke, fixed) ->
      val toRepair = broke.tokenizeByWhitespace()
      val humanRepair = fixed.tokenizeByWhitespace()
      val trueLevDist = levenshtein(broke, fixed)
      val encoded = encode(broke)
      val predLevDist = predDist("|$encoded")

      val intGram = try {
        val monoEditBounds = vanillaS2PCFGWE.maxParsableFragmentB(toRepair, pad = trueLevDist)
//    val multiEditBounds = vanillaS2PCFGWE.findMinimalMultiEditBounds(toRepair, monoEditBounds, levDist)
        val fsa = makeLevFSA(toRepair, trueLevDist, monoEditBounds)

        if (!fsa.recognizes(fixed))
          throw Exception("Human repair is unrecognizable!")
        else println("LEV-FSA recognizes human repair")

        s2pg.jvmIntersectLevFSAP(fsa = fsa, parikhMap = parikhMap)
          .also { intGram -> intGram.ifEmpty { println("Intersection grammar was empty!"); null } }
      } catch (e: Exception) { null } catch (e: Error) { null }

      try {
        if (intGram == null) throw Exception("Exception while building grammar!")
        else if (MAX_DFA_IN < intGram.size) throw Exception("Int grammar was still too large!")
        else if (humanRepair !in intGram.language) {
          println("Human repair recognized by original CFG: " + (humanRepair in vanillaS2PCFG.language))
          throw Exception("Human repair is unrecognizable by LEV ∩ CFG!")
        } else println("Human repair is recognized by LEV ∩ CFG!")
      } catch (e: Exception) { return@forEach }

      val pTree = measureTimedValue { intGram.toPTree(origCFG = s2pg) }
        .also { println("Constructed PTree in ${it.duration}") }.value

      val dfa = pTree.toDFA(minimize = true)!!

      val (uTokIndices, cTokIndices) = decodeDFAWithGroundTruthSteering(
        origStr = "|$encoded $trueLevDist ",
        trueRepair = encode(fixed),
        bAutomaton = dfa,
        dec = termDict,
      )

      urankTot += uTokIndices.let { it.sum().toDouble() / it.size }
      crankTot += cTokIndices.let { it.sum().toDouble() / it.size }

      println("BROKE_SEQ: $broke")
      println("FIXED_SEQ: $fixed")
      println("PRED_DIST: $predLevDist (ACT_DIST=$trueLevDist)")
      println("URANK_IDX: ${uTokIndices.joinToString(" ")}")
      println("CRANK_IDX: ${cTokIndices.joinToString(" ")}")
      println("AVGs: UNC=${urankTot / ++instances}, CST=${crankTot / instances}")
    }
  }

  fun decodeDFAWithGroundTruthSteering( // What was the rank of the true next token under LBH constraints?
    origStr: String,
    trueRepair: String,
    bAutomaton: BAutomaton,
    dec: Map<Char, Σᐩ>, // Maps unicode characters back to strings
  ): Π2A<List<Int>> {
    val unconstrainedIndices = mutableListOf<Int>()
    val constrainedIndices = mutableListOf<Int>()
    var options = bAutomaton.initialState.options(dec)

    trueRepair.map { "$it" }.fold(origStr) { acc, tok ->
      val nextToks = nextTokens(acc)
      unconstrainedIndices += nextToks.indexOf(tok)
      constrainedIndices +=
        if (unconstrainedIndices.last() == 0) 0 // Unconstrained 0-index implies the constrained index will be 0
        else nextToks.filter { PyTokMap.mt[it.first()]?.let { it in options } == true }.indexOf(tok)

      options = options[PyTokMap.mt[tok.first()]]!!.options(dec)
      acc + tok
    }

    return unconstrainedIndices to constrainedIndices
  }

  // Steers a random walk using the last n-1 transitions from the Markov Chain
  fun decodeDFA(
    origStr: String,
    bAutomaton: BAutomaton,
    // BAutomata uses a Unicode alphabet, and the Markov Chain recognizes a
    // string-based alphabet, so we need a way to translate between the two
    dec: Map<Char, Σᐩ>, // Maps unicode characters back to strings
    callback: (Σᐩ) -> Unit = {},
    timeout: Duration = Duration.INFINITE,
    beamWidth: Int = 200, // Maximum number of trajectories to keep at each step
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
          val lastToks = partTraj.traj.reversed().filterNotNull().joinToString(" ")
          val txs = partTraj.lastState.options(dec)
          val query = "$origStr${encode(lastToks)}"

          val nextTokensAndScores = try {
            if (txs.size == 1) listOf(txs.keys.first() to 1.0) // Short-circuit if only one option is available
            else nextTokensAndScores(query).filter { it.first in txs } // Otherwise, call LLM to fetch logProbs
          } catch (ex: Exception) { println("$ex / $query"); listOf() }
            // Fallback to uniform distribution over alphabet if error or empty transitions
            .ifEmpty { txs.keys.map { it to 1.0 / txs.keys.size }.filter { it.first in terms } }

          nextTokensAndScores.map { (t, s) -> partTraj.append(t, txs[t]!!, s) }
            .flatMap { traj ->
              if (traj.isComplete) {
                fullTrajectories.add(traj)
                callback(traj.toString())
                if (traj.lastState.transitions.isNotEmpty()) listOf(traj) else emptyList()
              } else { listOf(traj) }
            }.stream()
        }.sorted().limit(beamWidth.toLong()).toList()
      } catch (_: Exception) { emptyList<FSATrajectory>() }

      beam.clear()
      beam.addAll(nextBeam)
    }

    val deduped = fullTrajectories.distinct().map { it.toString() }.toList()

    println("Took ${startTime.elapsedNow()} to decode ${deduped.size} trajectories, with ${beam.size} in queue")
    return deduped
  }
}