package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.terminals
import ai.hypergraph.kaliningraph.repair.vanillaS2PCFG
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.to
import java.net.URL
import java.net.URLEncoder

object MakeMore {
  object PyTokMap {
    val tm = vanillaS2PCFG.terminals.mapIndexed { i, it -> it to (33 + i).toChar() }.toMap()
    val mt = vanillaS2PCFG.terminals.mapIndexed { i, it -> (33 + i).toChar() to it }.toMap()
  }

  val MAKEMORE_URL = "http://localhost:8000/makemore/"

  fun callExternal(s: String): String =
    PyTokMap.mt[URL(MAKEMORE_URL + URLEncoder.encode(s, "utf-8")).readText().first()]!!

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
}