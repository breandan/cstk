package edu.mcgill.gymfs.experiments

import org.apache.commons.lang3.StringUtils

fun main() {
  val codeSnippet = """
    static void main(String[] args) {
     Scanner in = new Scanner(System.in);
     int N = in.nextInt();
     for (int i = 1; i<=10; i++) {
       int sum = N * i;
       println(N + " x " + i + " = " + sum);
     }
    }
  """.trimIndent()

  println("====SYNTAX MUTATION========")
  println(codeSnippet.mutateSyntax())
  println("====STRUCTURE MUTATION=====")
  println(codeSnippet.mutateStructure())
  println("====SEMANTICS MUTATION=====")
  println(codeSnippet.mutateSemantics())
  println("====RENAMING MUTATION======")
  println(codeSnippet.renameTokens())
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
