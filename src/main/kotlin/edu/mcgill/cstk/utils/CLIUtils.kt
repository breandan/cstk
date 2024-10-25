package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.parsing.Σᐩ
import java.io.*


// Output stream that rejects all lines starting with "Parser error:" or "Lex error:"
class FilteredOutputStream(out: OutputStream) : PrintStream(out) {
  override fun println(x: String?) {
    if (x == null) return
    if (x.toString().let {
//      it.startsWith("logging: ") ||
        it.startsWith("Parser error:") ||
          it.startsWith("Lexer error:")
      }) return
    super.println(x)
  }
}

fun Σᐩ.execute() =
  ProcessBuilder(split(' ')).start().waitFor()

fun Σᐩ.execAndCapture() =
  ProcessBuilder(split(' ')).start().inputStream.bufferedReader().readText()

fun lastGitMessage() = "git log -1 --pretty=%s".execAndCapture().lines().first()