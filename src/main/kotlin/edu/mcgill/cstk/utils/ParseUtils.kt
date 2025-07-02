package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import me.vovak.antlr.parser.*
import org.antlr.v4.runtime.*
import org.jetbrains.kotlin.lexer.KotlinLexer
import org.jetbrains.kotlin.spec.grammar.tools.*
import java.util.Base64
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.text.replace

val errorListener =
  object: BaseErrorListener() {
    override fun syntaxError(
      recognizer: Recognizer<*, *>?,
      offendingSymbol: Any?,
      line: Int,
      charPositionInLine: Int,
      msg: Σᐩ?,
      e: RecognitionException?
    ) { throw Exception("$msg") }
  }

// Reports the index of the token that caused the error
val indexReportingListener =
  object: BaseErrorListener() {
    override fun syntaxError(
      recognizer: Recognizer<*, *>?,
      offendingSymbol: Any?,
      line: Int,
      charPositionInLine: Int,
      msg: Σᐩ?,
      e: RecognitionException?
    ) {
      (offendingSymbol as? Token)
        ?.let { throw Exception(it.tokenIndex.toString()) }
        ?: throw Exception("")
    }
  }

class ErrorListener : BaseErrorListener() {
  var hasErrors: Boolean = false
    private set

  override fun syntaxError(
    recognizer: Recognizer<*, *>?,
    offendingSymbol: Any?,
    line: Int,
    charPositionInLine: Int,
    msg: String?,
    e: RecognitionException?
  ) { hasErrors = true }
}

// Exhaustive tokenization includes whitespaces
fun Σᐩ.tokenizeAsPython(exhaustive: Boolean = false): List<Σᐩ> =
  if (!exhaustive) lexAsPython().allTokens.map { it.text }
  else tokenizeAsPython(false).fold(listOf(this)) { runningTokens, t ->
    val toSplit = runningTokens.last()
    runningTokens.dropLast(1) +
      if (t.all { it.isWhitespace() }) listOf(toSplit.takeWhile { it.isWhitespace() }, toSplit.dropWhile { it.isWhitespace() })
      else if (toSplit.startsWith(t)) listOf(t, toSplit.removePrefix(t))
      else if (t in toSplit) toSplit.substringBefore(t).let { listOf(it, t, toSplit.removePrefix(it).removePrefix(t)) }
      else if (toSplit.isEmpty()) listOf(t)
      else throw Exception("Could not find token $t in ${toSplit.map { it.code }}").also { println("\n\n$this\n\n") }
  }

fun List<Int>.getIndexOfFirstPythonError(): Int {
  val tokenSource = ListTokenSource(map { CommonToken(it) })
  val tokens = CommonTokenStream(tokenSource)
  return try {
    Python3Parser(tokens)
      .apply { removeErrorListeners(); addErrorListener(indexReportingListener) }
      .file_input()
    -1
  } catch (e: Exception) { e.message?.toIntOrNull() ?: -1 }
}

@JvmName("isValidPyLStr")
fun List<Σᐩ>.isValidPython(): Boolean =
  if (isNotEmpty()) map { pythonVocabBindex.getUnsafe(it) ?: it.toInt() }
    .let { if (it.last() != 39) it + 39 else it }
    .isValidPython() else false

@JvmName("isValidPyLInt")
fun List<Int>.isValidPython(): Boolean {
  if (isEmpty()) return false
  val withNewline = let { if (it.last() != 39) it + 39 else it }
  val tokenSource = ListTokenSource(withNewline.map { CommonToken(it) })
  val tokens = CommonTokenStream(tokenSource)
  val listener = ErrorListener()

  return try {
    Python3Parser(tokens)
      .apply { removeErrorListeners(); addErrorListener(listener) }
      .file_input()

    !listener.hasErrors
  } catch (e: Exception) { false }
}

val PYMAP = Python3Lexer.VOCABULARY.let {  v ->
  (0..v.maxTokenType).associateBy { v.getDisplayName(it) }
}

fun Σᐩ.toPythonIntType(trimmed: Σᐩ = trim()) =
  when (trimmed) {
    "BOS" -> Int.MIN_VALUE
    "EOS" -> Int.MAX_VALUE
    "INDENT" -> 98
    "DEDENT" -> 99
    "" -> -1
    else -> PYMAP[trimmed] ?: PYMAP["'$trimmed'"] ?: trimmed.toInt()
  }

fun Σᐩ.lexToIntTypesAsPython(
  lexer: Lexer = Python3Lexer(CharStreams.fromString(this))
) = lexer.allTokens.map { it.type }

val pythonVocabBindex: Bindex<Σᐩ> =
  Python3Lexer(CharStreams.fromString("")).vocabulary.let { vocab ->
    (0..vocab.maxTokenType).associateWith { vocab.getDisplayName(it) }
  }.let { Bindex(it) }//.also { println(it.toString()) }

fun Σᐩ.lexToStrTypesAsPython(
  lexer: Lexer = Python3Lexer(CharStreams.fromString(this)),
  vocabulary: Vocabulary = lexer.vocabulary
) = lexer.allTokens.map { vocabulary.getDisplayName(it.type) }

fun Σᐩ.lexToPythonTokens(
  lexer: Lexer = Python3Lexer(CharStreams.fromString(this)),
  vocabulary: Vocabulary = lexer.vocabulary
) = lexer.allTokens.toList()

fun Σᐩ.lexAsPython(): Python3Lexer =
  Python3Lexer(CharStreams.fromStream(byteInputStream()))

fun Σᐩ.lexAsJava(): Java8Lexer =
  Java8Lexer(CharStreams.fromStream(byteInputStream()))

fun Int.toPyRuleName(): String =
  if (this == Int.MIN_VALUE) "BOS"
  else if (this == Int.MAX_VALUE) "EOS"
  else Python3Lexer.VOCABULARY.getDisplayName(this).let {
    when (it) {
      "98" -> "INDENT"
      "99" -> "DEDENT"
      else -> it
    }
  }

fun Int.toPyRuleNameUnquoted(): String =
  toPyRuleName().let {
    if (it.startsWith("'") && it.endsWith("'") && 1 < it.length)
      it.substring(1, it.length - 1)
    else it
  }

//val KOTLIN_LEXER = KotlinLexer()
//fun String.lexAsKotlin(): List<String> =
//  tokenizeKotlinCode(this).map { it.text }
fun Σᐩ.lexAsKotlin(KOTLIN_LEXER: KotlinLexer = KotlinLexer()): List<Σᐩ> {
  KOTLIN_LEXER.start(this)
  val tokens = mutableListOf<Σᐩ>()
  while (KOTLIN_LEXER.tokenType != null) {
    try {
      tokens.add(KOTLIN_LEXER.tokenText)
      KOTLIN_LEXER.advance()
    } catch (_: Exception) { }
  }

  return tokens.filter { it.isNotBlank() }
}

fun Σᐩ.javac(): Σᐩ =
  try {
    val context = """
      class Hello {
          public static void main (String args[])
          {
              $this
          }
      }
    """.trimIndent()
    Java8Parser(CommonTokenStream(context.lexAsJava().apply { removeErrorListeners(); addErrorListener(errorListener) }))
      .apply { removeErrorListeners(); addErrorListener(errorListener) }
      .compilationUnit()
    ""
  } catch (e: Exception) { e.message!! }

fun Σᐩ.isValidJava() = javac().isEmpty()

fun Σᐩ.isValidPython(onErrorAction: (Σᐩ?) -> Unit = {}): Boolean =
  try {
    Python3Parser(
      CommonTokenStream((this + "\n")
      .lexAsPython().apply { removeErrorListeners(); addErrorListener(errorListener) })
    )
      .apply { removeErrorListeners(); addErrorListener(errorListener) }
      .file_input()
    true
  } catch (e: Exception) {
    onErrorAction(e.message)
    false
  }

fun String.mapToPythonCode(): String {
  val tokens = this.split(" ")
  val output = StringBuilder()
  var indentLevel = 0
  var lineStart = true
  var freshName = 0

  for (token in tokens) {
    when (token) {
      "INDENT" -> indentLevel++ // Increase indentation
      "DEDENT" -> indentLevel-- // Decrease indentation
      "NEWLINE" -> { output.append("\n"); lineStart = true }
      else -> {
        if (lineStart) {
          output.append("\t".repeat(indentLevel.coerceAtLeast(0)))
          lineStart = false
        } else { output.append(" ") }
        val mappedToken = when (token) {
          "NAME" -> "NAME_${freshName++}"
          "NUMBER" -> "1"
          "STRING" -> "\"\""
          else -> token
        }
        output.append(mappedToken)
      }
    }
  }
  return output.toString()
}

fun getOutput(code: String): String = try {
    val src = code.mapToPythonCode()
//  println("Checking:\n$src")
    val encoded = Base64.getEncoder().encodeToString(src.toByteArray(Charsets.UTF_8))

    val pythonCode = """
            import sys, base64, textwrap
            encoded = sys.argv[1]
            _src = base64.b64decode(encoded).decode('utf-8')
            _src = textwrap.dedent(_src)
            compile(_src, 'test_compile.py', 'exec')
        """.trimIndent()

    val processBuilder = ProcessBuilder("python", "-c", pythonCode, encoded)
    .redirectErrorStream(true)
    val process = processBuilder.start()
    val output = StringBuilder()
    BufferedReader(InputStreamReader(process.inputStream))
      .use { reader -> reader.forEachLine { output.append(it).append("\n") } }
    if (process.waitFor() == 0) "" else output.toString().trim()
  } catch (e: Exception) { e.message ?: e.localizedMessage }

fun Σᐩ.isInterpretablePython(): Boolean =
// Checks whether IO contains the string "SyntaxError"
  !ProcessBuilder("pylyzer", "-c", this).start()
    .apply { waitFor() }.errorStream.bufferedReader().readText()
    .contains("SyntaxError")

fun Σᐩ.getPythonErrorMessage(): Σᐩ =
  try {
    Python3Parser(
      CommonTokenStream((this + "\n")
        .lexAsPython().apply { removeErrorListeners(); addErrorListener(errorListener) })
    )
      .apply { removeErrorListeners(); addErrorListener(errorListener) }
      .file_input()
    ""
  } catch (e: Exception) { e.message!! }

fun Σᐩ.isSyntacticallyValidKotlin(): Boolean =
  try { parseKotlinCode(tokenizeKotlinCode(this)).let { true } }
  catch (_: Throwable) { false }

fun Σᐩ.coarsen(): Σᐩ =
  defaultTokenizer().joinToString(" ") {
    when {
      it.isBracket() -> it
      it == edu.mcgill.cstk.experiments.repair.MSK -> "_"
      else -> "w"
    }
  }

fun Σᐩ.uncoarsen(prompt: Σᐩ): Σᐩ {
  val words = prompt.defaultTokenizer().filter { it !in COMMON_BRACKETS }.toMutableList()
  return tokenizeByWhitespace().joinToString("") { token ->
    when {
      token.isBracket() -> token
      words.isEmpty() -> { //System.err.println("IDK what happened:\nSynthesized:  $this");
        "" }
      token == "w" -> words.removeAt(0)
      else -> throw Exception("Unknown token: $token")
    }
  } + words.joinToString("")
}

val pythonKeywords = setOf(
  "False", "None", "True", "and", "as", "assert",
  "async", "await", "break", "class", "continue",
  "def", "del", "elif", "else", "except", "finally",
  "for", "from", "global", "if", "import", "in", "is",
  "lambda", "nonlocal", "not", "or", "pass", "raise",
  "return", "try", "while", "with", "yield"
)

fun Σᐩ.coarsenAsPython(): Σᐩ =
  tokenizeAsPython().joinToString(" ") {
    when {
      it.isBracket() -> it
      it.none { it.isLetterOrDigit() } -> it
      it in pythonKeywords -> it
      else -> "w"
    }
  }

fun Σᐩ.uncoarsenAsPython(prompt: Σᐩ): Σᐩ {
  val words = prompt.tokenizeByWhitespace()
    .filter { it !in pythonKeywords && it.any { it.isLetterOrDigit() }}.toMutableList()
  val uncoarsed = tokenizeByWhitespace().joinToString(" ") { token ->
    when {
      token.isBracket() -> token
      token.none { it.isLetterOrDigit() } -> token
      token == "w" -> words.removeFirst()
      token in pythonKeywords -> token
      else -> throw Exception("Unknown token: $token")
    }
  } + words.joinToString(" ", " ")

//  println("After uncoarsening: $uncoarsed")
  return uncoarsed
}

val officialKotlinKeywords = setOf(
  "as", "as?", "break", "class", "continue", "do", "else", "false", "for", "fun", "if", "in",
  "!in", "interface", "is", "!is", "null", "object", "package", "return", "super", "this",
  "throw", "true", "try", "typealias", "val", "var", "when", "while", "by", "catch", "constructor",
  "delegate", "dynamic", "field", "file", "finally", "get", "import", "init", "param", "property",
  "receiver", "set", "setparam", "where", "actual", "abstract", "annotation", "companion",
  "const", "crossinline", "data", "enum", "expect", "external", "final", "infix", "inline",
  "inner", "internal", "lateinit", "noinline", "open", "operator", "out", "override", "private",
  "protected", "public", "reified", "sealed", "suspend", "tailrec", "vararg", "field", "it"
)

fun Σᐩ.coarsenAsKotlin(lex: Boolean = true): Σᐩ =
  (if(lex) lexAsKotlin() else tokenizeByWhitespace()).joinToString(" ") {
    when {
      it.isBracket() -> it
      it.none { it.isLetterOrDigit() } -> it
      it in officialKotlinKeywords -> it
      it.first().isUpperCase() -> "W"
      else -> "w"
    }
  }

fun Σᐩ.uncoarsenAsKotlin(prompt: Σᐩ): Σᐩ {
  val words = prompt.tokenizeByWhitespace()
    .filter { it !in officialKotlinKeywords && it.any { it.isLetterOrDigit() } }.toMutableList()
  val uncoarsed = tokenizeByWhitespace().joinToString(" ") { token ->
    when {
      token.isBracket() -> token
      token.none { it.isLetterOrDigit() } -> token
      token.equals("w", ignoreCase = true) -> words.removeFirst()
      token in officialKotlinKeywords -> token
      else -> throw Exception("Unknown token: $token")
    }
  } + words.joinToString(" ", " ")

//  println("After uncoarsening: $uncoarsed")
  return uncoarsed
}

