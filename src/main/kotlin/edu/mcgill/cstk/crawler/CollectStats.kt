package edu.mcgill.cstk.crawler

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.defaultTokenizer
import edu.mcgill.cstk.nlp.allMethods

fun main() {
  println("total lines, total tokens, avg line len, len comments, len code")
  DATA_DIR.allFilesRecursively()
    .allMethods()
    .forEach {  (method, uri) ->
      try {
        val string = method.toString()
         val lines = string.lines()
        println(
          "" +
            lines.size + "," +
            defaultTokenizer.tokenize(string).size +
            lines.size + "," +
            lines.map { defaultTokenizer.tokenize(it).size }.average()
              .toInt() + "," +
            defaultTokenizer.tokenize(method.docComment ?: "").size + "," +
            defaultTokenizer.tokenize(method.body?.toString() ?: "").size
        )
      }catch(exception: Exception) {}
    }
}