package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.types.cc
import com.github.difflib.text.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.probing.embeddingServer
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import me.vovak.antlr.parser.*
import net.sf.extjwnl.data.PointerUtils.*
import net.sf.extjwnl.dictionary.Dictionary
import org.antlr.v4.runtime.*
import org.apache.commons.lang3.StringUtils
import spoon.Launcher
import java.io.File
import java.net.*
import java.nio.file.*
import kotlin.io.path.toPath
import kotlin.math.max
import kotlin.random.Random

val errorListener =
  object: BaseErrorListener() {
    override fun syntaxError(
      recognizer: Recognizer<*, *>?,
      offendingSymbol: Any?,
      line: Int,
      charPositionInLine: Int,
      msg: String?,
      e: RecognitionException?
    ) { throw Exception("$msg") }
  }

fun String.javac() =
  try {
    val context ="""
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

fun String.isValidJava() = javac().isEmpty()

fun String.isValidPython(onErrorAction: (String?) -> Unit = {}) =
  try {
    Python3Parser(CommonTokenStream((this + "\n")
      .lexAsPython().apply { removeErrorListeners(); addErrorListener(errorListener) }))
      .apply { removeErrorListeners(); addErrorListener(errorListener) }
      .file_input()
    true
  } catch (e: Exception) {
    onErrorAction(e.message)
    false
  }

fun String.tokenizeAsPython(): List<String> =
  lexAsPython().allTokens.map { it.text }
fun String.lexAsPython() =
  Python3Lexer(CharStreams.fromStream(byteInputStream()))

fun String.lexAsJava() =
  Java8Lexer(CharStreams.fromStream(byteInputStream()))

fun String.execute() =
  ProcessBuilder( split(" ") ).start().waitFor()

fun synonymize(token: String): String =
  StringUtils.splitByCharacterTypeCamelCase(token).joinToString("") { old ->
    old.synonyms().filter { it !in RESERVED_TOKENS }.ifEmpty { setOf(old) }
      .random().let { new ->
        if (old.first().isLowerCase()) new
        else "" + new[0].uppercaseChar() + new.drop(1)
      }
  }

val defaultDict: Dictionary = Dictionary.getDefaultResourceInstance()

// Returns single-word synonyms
fun String.synonyms(synonymDepth: Int = 3): Set<String> =
  defaultDict.lookupAllIndexWords(this).indexWordArray.map {
    it.senses.map { sense ->
      (getSynonymTree(sense, synonymDepth).toList() +
        listOf(getDirectHyponyms(sense), getDirectHypernyms(sense)))
        .flatten().map { it.synset.words }
        .flatten().mapNotNull { it.lemma }
    }.flatten() + it.lemma
  }.flatten().filter { " " !in it }.toSet()

// Query in context
data class QIC(
  val query: String,
  val path: Path,
  val context: String,
  val offset: Int
)

// Searches through all files in the path for the query
fun URI.slowGrep(query: String, glob: String = "*"): Sequence<QIC> =
  allFilesRecursively().map { it.toPath() }
    .mapNotNull { path ->
      path.read()?.let { contents ->
        contents.extractConcordances(".*$query.*")
          .map { (cxt, idx) -> QIC(query, path, cxt, idx) }
      }
    }.flatten()

// Returns a list of all code fragments in all paths and their locations
fun Sequence<URI>.allCodeFragments(): Sequence<Pair<Concordance, String>> =
  map { path ->
    path.allLines()
      .mapIndexed { lineNum, line -> lineNum to line }
      .filter { (_, l) -> l.isNotBlank() && l.any(Char::isLetterOrDigit) }
      .map { (ln, l) -> Concordance(path, ln) to l.trim() }
//    .chunked(5).map { it.joinToString("\n") }
  }.flatten()

val controlFlowKeywords = setOf(
  "if", "else", "while", "case", "for", "switch",
  "do", "continue", "break", "&&", "||", "?", ":", "catch",
  "finally", "throw", "throws", "default", "return"
)
val funKeywords =
  setOf("public ", "private ", "void ", "static ", "fun ", "/**")
val notFunKeywords = setOf("class")
val openParens = setOf('(', '{', '[')
val closeParens = setOf(')', '}', ']')

// Slices files into method-level chunks using a Dyck-1 language
fun Sequence<URI>.allMethods(
  parser: (String) -> List<String> =
    { file -> Launcher.parseClass(file).methods.map { it.toString() } }
): Sequence<Pair<String, URI>> =
  mapNotNull { path ->
    path.contents()?.let {
      try {
        parser(it).map { it to path }
      } catch (exception: Exception) { null }
    }
  }.flatten()

fun String.splitMethods(): List<String> =
  lineSequence().fold(-1 to listOf<String>()) { (dyckSum, methods), line ->
    if (dyckSum < 0 && funKeywords.any { it in line } && notFunKeywords.none { it in line } && "(" in line) {
      line.countBalancedBrackets() to methods + line
    } else if (dyckSum == 0) {
      if (line.isBlank()) -1 to methods else 0 to methods.put(line)
    } else if (dyckSum > 0) {
      dyckSum + line.countBalancedBrackets() to methods.put(line)
    } else {
      -1 to methods
    }
  }.second.map { it.trimIndent() }.filter { "(" in it && "{" in it }

fun List<String>.put(line: String) = dropLast(1) + (last() + "\n" + line)

fun String.countBalancedBrackets(): Int = countBracketsAndMaxDepth().first

fun String.countBracketsAndMaxDepth() =
  fold(0 to 0) { (s, depth), c ->
    when (c) {
      in openParens -> (s + 1) to max(s, depth)
      in closeParens -> (s - 1) to depth
      else -> (s to depth)
    }
  }

fun URI.contents(): String? =
    when (scheme) {
      TGZ_SCHEME -> vfsManager.readText(this)
      FILE_SCHEME -> if (extension() in FILE_EXTs) File(this).readText() else null
      else -> null
    }

fun URI.allLines(): Sequence<String> =
  contents()?.lineSequence() ?: emptySequence()

fun Path.read(start: Int = 0, end: Int = -1): String? =
  try {
    Files.readString(this)
  } catch (e: Exception) {
    null
  }?.let { it.substring(start, if (end < 0) it.length else end) }

// Returns all substrings matching the query and their immediate context
fun String.extractConcordances(query: String): Sequence<Pair<String, Int>> =
  Regex(query).findAll(this).map {
    val range = 0..length
    val (matchStart, matchEnd) =
      it.range.first.coerceIn(range) cc (it.range.last + 1).coerceIn(range)
    substring(matchStart, matchEnd) to matchStart
  }

fun previewResult(query: String, loc: Concordance) =
  "[?=$query] ${loc.getContext(0).preview(query)}\t($loc)"

fun String.preview(query: String, window: Int = 10) =
  extractConcordances(query).map { (q, b) ->
    val range = 0..length
    substring((b - window).coerceIn(range), b) + "[?]" +
      substring(b + q.length, (b + q.length + window).coerceIn(range))
  }.joinToString("…", "…", "…") { it.trim() }

//https://github.com/huggingface/transformers/issues/1950#issuecomment-558770861
// Short sequence embedding: line-level
fun vectorize(query: String) = matrixize(query).first()

// Long sequence embedding: method level
fun matrixize(query: String): Array<DoubleArray> =
  defaultModel.makeQuery(query).first().lines()
  .map { it.trim().replace("[", "").replace("]", "") }
  .map { it.split(" ").filter(String::isNotEmpty).map(String::toDouble) }
  .map { it.toDoubleArray() }.toTypedArray()

fun Model.score(
  query: String,
  tokens: List<String> = query.splitByNonWordChars()
) =
  tokens.indices.map { tokens[it] to tokens.toMutableList().apply { this[it] = mask } }
    .map { (t, q) -> makeQueryAndScore(q.joinToString(""), listOf(t)).first() }

// Fill-in-the-middle training/inference: https://arxiv.org/pdf/2207.14255.pdf
fun Model.fillEveryHole(query: String) =
  query.split(mask).let { parts ->
    (1 until parts.size).fold(parts.first() to parts.drop(1)) { acc, _ ->
      val suffix = acc.second.joinToString("")
      complete(acc.first + mask + suffix)
        .dropLast(suffix.length) + acc.second.first() to acc.second.drop(1)
    }.first
  }

// Expands MSK autoregressively until maxTokens reached or stopChar encountered
tailrec fun Model.complete(
  query: String = mask,
  fullCompletion: String = query.replace(mask, makeQuery(query).first()),
  maxTokens: Int = 1,
): String =
  if (maxTokens <= 1) fullCompletion
  else complete(
    query = fullCompletion + mask,
    maxTokens = maxTokens - 1
  )

tailrec fun Model.completeUntilStopChar(
  query: String = mask,
  fullCompletion: String =
    // Take first probable stop sequence
    query.replace(mask, makeQuery(query).let { it.firstOrNull { ";" in it } ?: it.first() }),
  maxTokens: Int = 1,
  isStopChar: (Char) -> Boolean = { it in setOf('\n', ';') }
): String =
  if (maxTokens <= 1 || fullCompletion.any { isStopChar(it) }) fullCompletion
  else completeUntilStopChar(
    query = fullCompletion + mask,
    maxTokens = maxTokens - 1
  )

fun Model.countTokens(query: String) =
  "$SERVER_URL${name}?tokenize=${URLEncoder.encode(query, "utf-8")}"
    .let { URL(it).readText().count { it == ',' } }

fun Model.makeQuery(query: String = "", hints: Collection<String> = listOf()): List<String> =
  makeQueryAndScore(query, hints).map { it.first }

const val maxRetries = 5

fun <T> retry(times: Int = maxRetries, block: () -> T): T? =
  (0..times).asSequence()
    .map { try { block() } catch (ex: Exception) { Thread.sleep(Random.nextLong(100, 2000)); null } }
    .firstNotNullOfOrNull { it }

fun Model.score(query: String): Float =
  ("$SERVER_URL${name}?score=${URLEncoder.encode(query, "utf-8")}")
    .let { url ->
      retry(10) { URL(url).readText().toFloat() }
        ?: Float.NaN.also { println("Error: $url") }
    }

/** Queries a model for its predictions. See [embeddingServer]. */
fun Model.makeQueryAndScore(query: String = "", hints: Collection<String> = listOf()): List<Pair<String, Double>> =
  ("$SERVER_URL${name}?query=${URLEncoder.encode(query, "utf-8")}" +
    // http://localhost:8000/microsoft/graphcodebert-base?query=System.%3Cmask%3E.println()&hint=test&hint=err&hint=out
    hints.joinToString("") { "&hint=" + URLEncoder.encode(it, "utf-8") })
    .let { url ->
      retry {
        URL(url).readText().lines()
          .map { it.substringBeforeLast(",") to it.substringAfterLast(",").toDouble() }
      } ?: listOf("" to 0.0).also { println("Error: $url") }
    }

fun List<String>.sortedByDist(query: String, metric: MetricStringDistance) =
  sortedBy { metric.distance(it, query) }

fun printLaTeXSummary(
  summary: String,
  original: String,
  synthetic: String,
  variant: String,
  discrepancy: String
) =
"""
%---------

\pagebreak
\section{Example}
\subsection{Summary}

\begin{lstlisting}[language=java]
$summary
\end{lstlisting}

\subsection{Original}
\begin{lstlisting}[language=java]
${diffString(original, synthetic).first}
\end{lstlisting}
\subsection{Synthetic}

\begin{lstlisting}[language=java]
${diffString(original, synthetic).second}
\end{lstlisting}

\subsection{Variant}

\begin{lstlisting}[language=java]
${diffString(original, variant).second}
\end{lstlisting}

\subsection{Comment}

TODO.

\subsection{Discrepancy}

\begin{lstlisting}[language=java]
$discrepancy
\end{lstlisting}

%--------
""".trimIndent().also { println(it) }

fun diffString(old: String, new: String) =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .ignoreWhiteSpaces(true)
    .inlineDiffByWord(true)
    .newTag { l -> if(l) "(*@\\hlred{" else "}@*)" }
    .oldTag { l -> if(l) "(*@\\hlred{" else "}@*)" }
    .lineNormalizer { it }
    .build()
    .generateDiffRows(old.lines(), new.lines())
    .fold("" to "") { (o, n), it ->
      "$o\n${it.oldLine}" to "$n\n${it.newLine}"
    }

fun prettyDiffs(
  list: List<String>,
  headings: List<String>,
  maxLen: Int = 180, maxLines: Int = 200,
) =
  (list.windowed(2).zip(headings.windowed(2))).let { diffs ->
    diffs.map { prettyDiffHorizontal(it.first[0], it.first[1], it.second[0], it.second[1], maxLen, maxLines) }
      .let { it.joinToString(List(it.maxOf { it.lines().maxOf { it.length } }) { "=" }.joinToString("", "", "\n")) }
  } + "\n\n"

const val ANSI_RESET = "\u001B[0m"
const val ANSI_BLACK = "\u001B[30m"
const val ANSI_RED = "\u001B[31m"
const val ANSI_GREEN = "\u001B[32m"
const val ANSI_YELLOW = "\u001B[33m"
const val ANSI_BLUE = "\u001B[34m"
const val ANSI_PURPLE = "\u001B[35m"
const val ANSI_CYAN = "\u001B[36m"
const val ANSI_WHITE = "\u001B[37m"

const val ANSI_BLACK_BACKGROUND = "\u001B[40m"
const val ANSI_RED_BACKGROUND = "\u001B[41m"
const val ANSI_GREEN_BACKGROUND = "\u001B[42m"
const val ANSI_ORANGE_BACKGROUND = "\u001B[43m"
const val ANSI_YELLOW_BACKGROUND = "\u001B[43m"
const val ANSI_BLUE_BACKGROUND = "\u001B[44m"
const val ANSI_PURPLE_BACKGROUND = "\u001B[45m"
const val ANSI_CYAN_BACKGROUND = "\u001B[46m"
const val ANSI_WHITE_BACKGROUND = "\u001B[47m"

fun String.visibleLen() =
  replace(ANSI_RED_BACKGROUND,"")
    .replace(ANSI_GREEN_BACKGROUND,"")
    .replace(ANSI_RESET,"").length

// Just print the new line with ASCII colors but no border
fun prettyDiffNoFrills(original: String, new: String) =
DiffRowGenerator.create()
  .showInlineDiffs(true)
  .inlineDiffByWord(true)
  .newTag { l -> if(l) "<begin>" else "<end>" }
  .oldTag { _ -> "" }
  .build()
  .generateDiffRows(original.split(" "), new.split(" ")).joinToString(" ") {
    when (it.tag) {
      DiffRow.Tag.INSERT -> it.newLine.replace("<begin>", ANSI_GREEN_BACKGROUND).replace("<end>", ANSI_RESET)
      DiffRow.Tag.CHANGE -> it.newLine.replace("<begin>", ANSI_YELLOW_BACKGROUND).replace("<end>", ANSI_RESET)
      DiffRow.Tag.DELETE -> it.newLine//.replace("<begin>", "$ANSI_RED_BACKGROUND _" ).replace("<end>", ANSI_RESET)
      else -> it.newLine.replace("<begin>", "").replace("<end>", "")
    }
  }

fun prettyDiffHorizontal(
  left: String, right: String,
  leftHeading: String = "original",
  rightHeading: String = "new",
  maxLen: Int = 180, maxLines: Int = 200,
): String {
  val sb = StringBuilder()
  val leftLines = left.lines()
  val rightLines = right.lines()
  if (leftLines.all { it.length < maxLen } && leftLines.size < maxLines) {
    val rows = DiffRowGenerator.create()
      .showInlineDiffs(true)
      .ignoreWhiteSpaces(true)
      .inlineDiffByWord(true)
      .lineNormalizer { it }
      .oldTag { l -> if(l) ANSI_RED_BACKGROUND else ANSI_RESET }
      .newTag { l -> if(l) ANSI_GREEN_BACKGROUND else ANSI_RESET }
      .build()
      .generateDiffRows(leftLines, rightLines)

    val padLeft = rows.maxOf { it.oldLine.visibleLen() } + 3
    val padRight = rows.maxOf { it.newLine.visibleLen() } + 3

    val tlsep = "┌".padEnd(padLeft, '─')
    val trsep = "┬".padEnd(padRight, '─')
    sb.appendLine("$tlsep$trsep┐")
    sb.appendLine(
      "│ $leftHeading".padEnd(padLeft, ' ') +
        "│ $rightHeading".padEnd(padRight, ' ') + "│"
    )

    val lsep = "├".padEnd(padLeft, '─')
    val rsep = "┼".padEnd(padRight, '─')

    fun String.adjust(len: Int) = padEnd(len + length - visibleLen() - 3, ' ')
    sb.appendLine("$lsep$rsep┤")
    rows.forEach { row ->
      sb.appendLine("│ ${row.oldLine.adjust(padLeft)} │ ${row.newLine.adjust(padRight)} │")
    }

    val blsep = "└".padEnd(padLeft, '─')
    val brsep = "┴".padEnd(padRight, '─')
    sb.appendLine("$blsep$brsep┘")
  }
  return sb.toString()
}

fun String.splitByNonWordChars() = split(Regex("((?<=\\W)|(?=\\W))"))