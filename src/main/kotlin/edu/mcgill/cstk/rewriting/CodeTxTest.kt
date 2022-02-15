package edu.mcgill.cstk.rewriting

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.probing.*
import edu.mcgill.cstk.experiments.synonymize
import edu.mcgill.cstk.nlp.*
import kotlin.math.min

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
  println("====ADDING NONESSENTIAL CODE======")
  println(codeSnippet.addExtraLogging())

  // Semantics-altering mutations
  println("====SWAPPING ARGUMENTS=====")
  println(codeSnippet.permuteArgumentOrder())
  println("====FUZZING LOOP BOUNDS====")
  println(codeSnippet.fuzzLoopBoundaries())
  println("====SWAP +/- MUTATION=====")
  println(codeSnippet.swapPlusMinus())

  TEST_DIR.allFilesRecursively().allMethods()
    .map { it.first.toString() to it.second }
    .take(1000)
    .map { (method, origin) ->
      val variant = method.renameTokens()
      if (variant == method) null else method to variant
    }.toList().mapNotNull { it }.forEach { (original, variant) ->
      if (original != variant) printSideBySide(original, variant)
    }
}

val reservedWords = setOf(
  // Java
  "abstract", "assert", "boolean", "break", "byte", "case",
  "catch", "char", "class", "const", "continue", "default",
  "double", "do", "else", "enum", "extends", "false",
  "final", "finally", "float", "for", "goto", "if",
  "implements", "import", "instanceof", "int", "interface", "long",
  "native", "new", "null", "package", "private", "protected",
  "public", "return", "short", "static", "strictfp", "super",
  "switch", "synchronized", "this", "throw", "throws", "transient",
  "true", "try", "void", "volatile", "while",

  // Kotlin
  "as", "is", "as", "break", "class", "continue", "do", "else", "false", "for",
  "fun", "if", "in", "null", "object", "package", "return", "super", "this",
  "throw", "true", "try", "typealias", "typeof", "val",
  "var", "when", "while", "by", "delegates",
  "catch", "constructor", "delegate", "dynamic", "field", "file", "finally",
  "get", "import", "init", "param", "property",
  "receiver", "set", "is", "setparam", "value", "where", "actual", "abstract",
  "annotation", "companion", "const", "crossinline",
  "data", "enum", "expect", "external", "final", "infix", "inline", "inner",
  "internal", "lateinit", "noinline", "open", "operator",
  "out", "override", "private", "protected", "public", "reified", "sealed",
  "suspend", "tailrec", "vararg", "field", "it",

  // Data types
  "byte", "short", "int", "long", "float", "double", "boolean", "char",
  "Byte", "Short", "Int", "Long", "Float", "Double", "Boolean", "Char",

  // Common SDK names
  "System", "out", "println"
)

fun String.mutateSyntax() =
  map {
    if (!it.isWhitespace() && Math.random() < 0.3)
      ('!'..'~').random()
    else it
  }.joinToString("")

fun String.shuffleLines() = lines().shuffled(DEFAULT_RAND).joinToString("\n")

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
  replace(Regex("\\((.*, .*)\\)")) { match ->
    match.groupValues[1].split(", ").shuffled(DEFAULT_RAND).joinToString(", ", "(", ")")
  }

fun String.fuzzLoopBoundaries(): String =
  replace(Regex("(for|while)(.*)(\\d+)(.*)")) { match ->
    match.groupValues.let { it[1] + it[2] +
      (it[3].toInt() + (1..3).random()) + it[4] }
  }

// Swaps adjacent lines with same indentation and no dataflow deps
fun String.swapMultilineNoDeps(): String =
  lines().chunked(2).map { lines ->
    if (lines.size != 2) return@map lines
    val (a, b) = lines.first() to lines.last()
    // Same indentation
    if (a.trim().length - a.length != b.trim().length - b.length)
      return@map listOf(a, b)

    // Only swap if no dataflow deps are present
    val hasIdsInCommon = a.split(Regex("[^A-Za-z]")).toSet()
      .intersect(b.split(Regex("[^A-Za-z]")).toSet())
      .any { it.isNotEmpty() && it.all(Char::isJavaIdentifierPart) }

    if (hasIdsInCommon) listOf(a, b) else listOf(b, a)
  }.flatten().joinToString("\n")

const val FILL = "<FILL>"
fun Model.fillFirstDoc(snippet: String): String? =
  snippet.lines().first { docCriteria(it) }.let { firstDoc ->
    try {
      snippet.lines().joinToString("\n") {
        if (it == firstDoc)
          it.substringBefore("//") + "// $FILL"
        else it
      }.let {
        completeDocumentation(it,
          min(20, defaultTokenizer.tokenize(firstDoc.substringAfter("//")).size)
        )
      }
    } catch (exception: Exception) { null }
  }

tailrec fun Model.completeDocumentation(
  snippet: String,
  length: Int = 20,
  nextToken: String? = makeQuery(snippet.replaceFirst(FILL, MSK))
    // Ensure at least one natural language character per token
    .firstOrNull { it.any(Char::isLetterOrDigit) }
): String? =
  if (length == 1 || nextToken == null) snippet.replace(FILL, "")
  else completeDocumentation(
    snippet = nextToken.let { snippet.replaceFirst(FILL, it + FILL) },
    length = length - 1
  )

tailrec fun String.fillOneByOne(): String =
  if (FILL !in this) this
  else replaceFirst(FILL, defaultModel.makeQuery(replaceFirst(FILL, MSK)).first()).fillOneByOne()

fun String.addExtraLogging(): String =
  (listOf("") + lines() + "").windowed(3, 1).joinToString("\n") { window ->
    val (lastLine, thisLine, nextLine) = Triple(window[0], window[1], window[2])

    val loggingFrequency = 0.3 // Lower to decrease logging frequency
    if (loggingFrequency < DEFAULT_RAND.nextDouble()) return@joinToString thisLine

    val matchLastLine = Regex("\\s+.*?;").matchEntire(lastLine)
    val matchThisLine = Regex("\\s+.*?;").matchEntire(thisLine)
    if (
    // Only print inside nested blocks of statements
      matchLastLine != null && matchThisLine != null &&
      // Space out print statements
      "print" !in lastLine + thisLine + nextLine
    ) {
      val toIndent = thisLine.takeWhile { !it.isJavaIdentifierPart() }
//      val toPrint = matchLastLine.groupValues[1]
      "${toIndent}System.out.println(\"debug\");\n$thisLine"
    } else thisLine
  }.fillOneByOne()

fun String.rewriteLoop(): String =
  TODO("Convert loop with conditional to infinite loop with breakout")

fun String.invertConditional(): String =
  TODO("Reverse branching statement with !")

//fun String.addExtraComments(commentFrequency: Double = 0.2): String =
//  lines().fold("" to "") { (lastLine, prevLines), thisLine ->
//    thisLine to "$prevLines\n" +
//      if (
//        lastLine.trim().endsWith(";") &&
//        thisLine.first().isWhitespace() &&
//        thisLine.trimStart().first().isJavaIdentifierPart() &&
//        Random.nextDouble() < commentFrequency
//      ) {
//        val indent = thisLine.takeWhile { !it.isJavaIdentifierPart() }
//        "${indent}println($FILL);\n$thisLine"
//      } else thisLine
//  }.second