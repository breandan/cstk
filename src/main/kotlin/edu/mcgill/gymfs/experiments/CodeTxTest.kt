package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*

fun main() {
  val codeSnippet = """
    static void main(String[] args) {
     Scanner in = new Scanner(System.in);
     int amount = in.nextInt();
     for (int i = 1; i<=10; i++) {
       int sum = amount * i;
       println(amount + " x " + i + " = " + sum);
       test(1, 2, 3, 4, 5)
     }
    }
  """.trimIndent()

  // Syntax-destroying mutations
  println("====SYNTAX MUTATION========")
  println(codeSnippet.mutateSyntax())
  println("====SHUFFLE LINES MUTATION=====")
  println(codeSnippet.shuffleLines())

  // Semantics-preserving mutations
  println("====RENAMING MUTATION======")
  println(codeSnippet.renameTokens())
  println("====SWAPPING LINES WITH NO DATAFLOW DEPS======")
  println(codeSnippet.swapMultilineNoDeps())
  println("====ADDING DEAD CODE======")
  println(codeSnippet.addDeadCode())

  // Semantics-altering mutations
  println("====SWAPPING ARGUMENTS=====")
  println(codeSnippet.permuteArgumentOrder())
  println("====FUZZING LOOP BOUNDS====")
  println(codeSnippet.fuzzLoopBoundaries())
  println("====SWAP +/- MUTATION=====")
  println(codeSnippet.swapPlusMinus())
}

fun String.mutateSyntax() =
  map {
    if (!it.isWhitespace() && Math.random() < 0.3)
      ('!'..'~').random()
    else it
  }.joinToString("")

fun String.shuffleLines() = lines().shuffled().joinToString("\n")

fun String.swapPlusMinus() =
  map { if (it == '+') '-' else it }.joinToString("")

fun String.same() = this

fun String.renameTokens(): String {
  val toReplace = mostFrequentIdentifier()
  val synonym = synonymize(toReplace) // Can be a fixed token, e.g. "tt"
  return if (toReplace.isBlank() || synonym.isBlank()) this
  else replace(toReplace, synonym)
}

fun String.mostFrequentIdentifier(): String =
  split(Regex("[^\\w']+")).filter {
    it.length > 4 && it !in reservedWords &&
      it.all(Char::isJavaIdentifierPart) && it.first().isLowerCase()
  }.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: ""

fun String.permuteArgumentOrder(): String =
  replace(Regex("\\((.*,.*)\\)")) { match ->
    match.groupValues[1].split(",").shuffled().joinToString(",", "(", ")")
  }

fun String.fuzzLoopBoundaries(): String =
  replace(Regex("(for|while)(.*)([0-9]+)(.*)")) { match ->
    match.groupValues.let { it[1] + it[2] +
      (it[3].toInt() + (1..3).random()) + it[4] }
  }

fun String.swapMultilineNoDeps(): String =
  lines().chunked(2).map { lines ->
    if (lines.size != 2) return@map lines
    val (a, b) = lines.first() to lines.last()
    // Same indentation
    if (a.trim().length - a.length != b.trim().length - b.length)
      return@map listOf(a, b)

    // Only swap if no dataflow deps are present
    val hasIdsInCommon = a.split(Regex("[^A-Za-z]")).toSet()
      .intersect(b.split(Regex("[^A-Za-z]")))
      .any { it.isNotEmpty() && it.all(Char::isJavaIdentifierPart) }

    if (hasIdsInCommon) listOf(a, b) else listOf(b, a)
  }.flatten().joinToString("\n")

fun String.addDeadCode(): String =
  lines().joinToString("\n") {
    if (Math.random() < 0.3) "$it; int deadCode = 2;" else it
  }