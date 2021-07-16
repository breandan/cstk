package edu.mcgill.gymfs.experiments

import org.apache.commons.lang3.StringUtils
import kotlin.text.RegexOption.MULTILINE

fun main() {
  val codeSnippet = """
    static void main(String[] args) {
     Scanner in = new Scanner(System.in);
     int N = in.nextInt();
     for (int i = 1; i<=10; i++) {
       int sum = N * i;
       println(N + " x " + i + " = " + sum);
       test(1, 2, 3, 4, 5)
     }
    }
  """.trimIndent()

  println("====SYNTAX MUTATION========")
  println(codeSnippet.mutateSyntax())
  println("====STRUCTURE MUTATION=====")
  println(codeSnippet.mutateStructure())
  println("====SEMANTICS MUTATION=====")
  println(codeSnippet.mutateSemantics())

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
}

fun String.mutateSyntax() =
  map {
    if (!it.isWhitespace() && Math.random() < 0.3)
      ('!'..'~').random() else it
  }.joinToString("")

fun String.mutateStructure() = lines().shuffled().joinToString("\n")

fun String.mutateSemantics() =
  map { if (it == '+') '-' else it }.joinToString("")

fun String.renameTokens(): String {
  val toReplace = split(Regex("[^\\w']+"))
    .filter { it.length > 5 && it.all { it.isJavaIdentifierPart() } }
    .groupingBy { it }.eachCount().maxByOrNull { it.value }!!.key
  return replace(toReplace, "X")
}

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
  lines().zipWithNext().map { (a, b) ->
    //Same indentation
    if(a.trim().length - a.length != b.trim().length - b.length)// Same indent
      return@map a to b

    // Only swap if no dataflow deps are present
    val noIdsInCommon = a.split(Regex("[^A-Za-z]")).toSet()
      .intersect(b.split(Regex("[^A-Za-z]")))
      .filter { it.all { it.isJavaIdentifierPart() } }
      .isEmpty()

    if (noIdsInCommon) b to a else a to b
  }.unzip().first.joinToString("\n")

fun String.addDeadCode(): String =
  lines().joinToString("\n") {
    if (Math.random() < 0.3) "$it; int deadCode = 2;" else it
  }