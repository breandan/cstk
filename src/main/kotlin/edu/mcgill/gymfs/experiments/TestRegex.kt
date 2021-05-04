package edu.mcgill.gymfs.experiments

fun main() {
  println(synthesizeRegex("asdf", "testasdf"))
}

fun synthesizeRegex(vararg strings: String) =
  ProcessBuilder("./grex", *strings).start()
    .inputStream.reader(Charsets.UTF_8)
    .use { Regex(it.readText()) }