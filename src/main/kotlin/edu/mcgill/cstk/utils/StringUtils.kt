package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.COMMON_BRACKETS
import ai.hypergraph.kaliningraph.types.cc
import com.github.difflib.text.*
import com.github.difflib.text.DiffRow.Tag.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.probing.embeddingServer
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import me.vovak.antlr.parser.*
import net.sf.extjwnl.data.PointerUtils.*
import net.sf.extjwnl.dictionary.Dictionary
import org.antlr.v4.runtime.*
import org.apache.commons.lang3.StringUtils
import org.jetbrains.kotlin.lexer.*
import spoon.Launcher
import java.io.File
import java.net.*
import java.nio.file.*
import kotlin.io.path.toPath
import kotlin.math.max
import kotlin.random.Random

fun synonymize(token: Σᐩ): Σᐩ =
  StringUtils.splitByCharacterTypeCamelCase(token).joinToString("") { old ->
    old.synonyms().filter { it !in RESERVED_TOKENS }.ifEmpty { setOf(old) }
      .random().let { new ->
        if (old.first().isLowerCase()) new
        else "" + new[0].uppercaseChar() + new.drop(1)
      }
  }

val defaultDict: Dictionary = Dictionary.getDefaultResourceInstance()

// Returns single-word synonyms
fun Σᐩ.synonyms(synonymDepth: Int = 3): Set<Σᐩ> =
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
  val query: Σᐩ,
  val path: Path,
  val context: Σᐩ,
  val offset: Int
)

// Searches through all files in the path for the query
fun URI.slowGrep(query: Σᐩ, glob: Σᐩ = "*"): Sequence<QIC> =
  allFilesRecursively().map { it.toPath() }
    .mapNotNull { path ->
      path.read()?.let { contents ->
        contents.extractConcordances(".*$query.*")
          .map { (cxt, idx) -> QIC(query, path, cxt, idx) }
      }
    }.flatten()

// Returns a list of all code fragments in all paths and their locations
fun Sequence<URI>.allCodeFragments(): Sequence<Pair<Concordance, Σᐩ>> =
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
  parser: (Σᐩ) -> List<Σᐩ> =
    { file -> Launcher.parseClass(file).methods.map { it.toString() } }
): Sequence<Pair<Σᐩ, URI>> =
  mapNotNull { path ->
    path.contents()?.let {
      try {
        parser(it).map { it to path }
      } catch (exception: Exception) { null }
    }
  }.flatten()

fun Σᐩ.splitMethods(): List<Σᐩ> =
  lineSequence().fold(-1 to listOf<Σᐩ>()) { (dyckSum, methods), line ->
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

fun List<Σᐩ>.put(line: Σᐩ) = dropLast(1) + (last() + "\n" + line)

fun Σᐩ.countBalancedBrackets(): Int = countBracketsAndMaxDepth().first

fun Σᐩ.countBracketsAndMaxDepth() =
  fold(0 to 0) { (s, depth), c ->
    when (c) {
      in openParens -> (s + 1) to max(s, depth)
      in closeParens -> (s - 1) to depth
      else -> (s to depth)
    }
  }

fun URI.contents(): Σᐩ? =
    when (scheme) {
      TGZ_SCHEME -> vfsManager.readText(this)
      FILE_SCHEME -> if (extension() in FILE_EXTs) File(this).readText() else null
      else -> null
    }

fun URI.allLines(): Sequence<Σᐩ> =
  contents()?.lineSequence() ?: emptySequence()

fun Path.read(start: Int = 0, end: Int = -1): Σᐩ? =
  try {
    Files.readString(this)
  } catch (e: Exception) {
    null
  }?.let { it.substring(start, if (end < 0) it.length else end) }

// Returns all substrings matching the query and their immediate context
fun Σᐩ.extractConcordances(query: Σᐩ): Sequence<Pair<Σᐩ, Int>> =
  Regex(query).findAll(this).map {
    val range = 0..length
    val (matchStart, matchEnd) =
      it.range.first.coerceIn(range) cc (it.range.last + 1).coerceIn(range)
    substring(matchStart, matchEnd) to matchStart
  }

fun previewResult(query: Σᐩ, loc: Concordance) =
  "[?=$query] ${loc.getContext(0).preview(query)}\t($loc)"

fun Σᐩ.preview(query: Σᐩ, window: Int = 10) =
  extractConcordances(query).map { (q, b) ->
    val range = 0..length
    substring((b - window).coerceIn(range), b) + "[?]" +
      substring(b + q.length, (b + q.length + window).coerceIn(range))
  }.joinToString("…", "…", "…") { it.trim() }

//https://github.com/huggingface/transformers/issues/1950#issuecomment-558770861
// Short sequence embedding: line-level
fun vectorize(query: Σᐩ) = matrixize(query).first()

// Long sequence embedding: method level
fun matrixize(query: Σᐩ): Array<DoubleArray> =
  defaultModel.makeQuery(query).first().lines()
  .map { it.trim().replace("[", "").replace("]", "") }
  .map { it.split(' ').filter(Σᐩ::isNotEmpty).map(Σᐩ::toDouble) }
  .map { it.toDoubleArray() }.toTypedArray()

fun Model.score(
  query: Σᐩ,
  tokens: List<Σᐩ> = query.splitByNonWordChars()
) =
  tokens.indices.map { tokens[it] to tokens.toMutableList().apply { this[it] = mask } }
    .map { (t, q) -> makeQueryAndScore(q.joinToString(""), listOf(t)).first() }

// Fill-in-the-middle training/inference: https://arxiv.org/pdf/2207.14255.pdf
fun Model.fillEveryHole(query: Σᐩ) =
  query.split(mask).let { parts ->
    (1 until parts.size).fold(parts.first() to parts.drop(1)) { acc, _ ->
      val suffix = acc.second.joinToString("")
      complete(acc.first + mask + suffix)
        .dropLast(suffix.length) + acc.second.first() to acc.second.drop(1)
    }.first
  }

// Expands MSK autoregressively until maxTokens reached or stopChar encountered
tailrec fun Model.complete(
  query: Σᐩ = mask,
  fullCompletion: Σᐩ = query.replace(mask, makeQuery(query).first()),
  maxTokens: Int = 1,
): Σᐩ =
  if (maxTokens <= 1) fullCompletion
  else complete(
    query = fullCompletion + mask,
    maxTokens = maxTokens - 1
  )

tailrec fun Model.completeUntilStopChar(
  query: Σᐩ = mask,
  fullCompletion: Σᐩ =
    // Take first probable stop sequence
    query.replace(mask, makeQuery(query).let { it.firstOrNull { ";" in it } ?: it.first() }),
  maxTokens: Int = 1,
  isStopChar: (Char) -> Boolean = { it in setOf('\n', ';') }
): Σᐩ =
  if (maxTokens <= 1 || fullCompletion.any { isStopChar(it) }) fullCompletion
  else completeUntilStopChar(
    query = fullCompletion + mask,
    maxTokens = maxTokens - 1
  )

fun Model.countTokens(query: Σᐩ) =
  "$SERVER_URL${name}?tokenize=${URLEncoder.encode(query, "utf-8")}"
    .let { URL(it).readText().count { it == ',' } }

fun Model.makeQuery(query: Σᐩ = "", hints: Collection<Σᐩ> = listOf()): List<Σᐩ> =
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
${latexDiffMultilineStrings(original, synthetic).first}
\end{lstlisting}
\subsection{Synthetic}

\begin{lstlisting}[language=java]
${latexDiffMultilineStrings(original, synthetic).second}
\end{lstlisting}

\subsection{Variant}

\begin{lstlisting}[language=java]
${latexDiffMultilineStrings(original, variant).second}
\end{lstlisting}

\subsection{Comment}

TODO.

\subsection{Discrepancy}

\begin{lstlisting}[language=java]
$discrepancy
\end{lstlisting}

%--------
""".trimIndent().also { println(it) }

fun latexDiffMultilineStrings(old: Σᐩ, new: Σᐩ): Pair<String, String> =
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
  list: List<Σᐩ>,
  headings: List<Σᐩ>,
  maxLen: Int = 180, maxLines: Int = 200,
) =
  (list.windowed(2).zip(headings.windowed(2))).let { diffs ->
    diffs.map { prettyDiffHorizontal(it.first[0], it.first[1], it.second[0], it.second[1], maxLen, maxLines) }
      .let { it.joinToString(List(it.maxOf { it.lines().maxOf { it.length } }) { "=" }.joinToString("", "", "\n")) }
  } + "\n\n"

fun Σᐩ.visibleLen() =
  replace(ANSI_RED_BACKGROUND,"")
    .replace(ANSI_GREEN_BACKGROUND,"")
    .replace(ANSI_RESET,"")
    // Replace tabs with 4 spaces
    .replace("\t", "  ")
    .length

fun latexDiffSingleLOC(original: Σᐩ, new: Σᐩ) =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { l -> if(l) "(*@<begin>" else "<end>@*)" }
    .build()
    .generateDiffRows(original.tokenizeByWhitespace(), new.tokenizeByWhitespace())
    .joinToString(" ") {
      when (it.tag) {
        INSERT -> it.newLine.replace("<begin>", "\\hlgreen{").replace("<end>", "}")
        CHANGE -> it.newLine.replace("<begin>", "\\hlorange{").replace("<end>", "}")
        DELETE -> "\\hlred{${List(it.oldLine.length){ " " }.joinToString("")}}"
        else -> it.newLine.replace("<begin>", "").replace("<end>", "")
      }
    }.replace("&lt;", "<").replace("&gt;", ">")

// Just print the new line with ASCII colors but no border
fun prettyDiffNoFrills(original: Σᐩ, new: Σᐩ) =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { l -> if (l) "<begin>" else "<end>" }
    .oldTag { _ -> "" }
    .build()
    .generateDiffRows(original.split(' '), new.split(' ')).joinToString(" ") {
      when (it.tag) {
        INSERT -> it.newLine.replace("<begin>", ANSI_GREEN_BACKGROUND).replace("<end>", ANSI_RESET)
        CHANGE -> it.newLine.replace("<begin>", ANSI_YELLOW_BACKGROUND).replace("<end>", ANSI_RESET)
        DELETE -> "$ANSI_RED_BACKGROUND${List(it.oldLine.length){ " " }.joinToString("")}$ANSI_RESET"
        else -> it.newLine.replace("<begin>", "").replace("<end>", "")
      }
    }.replace("&lt;", "<").replace("&gt;", ">")

// Trim the section before the first change and after the last change, then trim the original and align to the same length
fun prettyDiffNoFrillsTrimAndAlignWithOriginal(original: Σᐩ, new: Σᐩ) =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { l -> if (l) "<begin>" else "<end>" }
    .oldTag { _ -> "" }
    .build()
    .generateDiffRows(original.split(' '), new.split(' ')).let {
      fun String.redo(sub: String) = replace("<begin>", sub).replace("<end>", ANSI_RESET)
      fun String.stripTags() = replace("<begin>", "").replace("<end>", "")
      fun String.strippedLen() = stripTags().length
      it.joinToString(" ") {
        when (it.tag) {
          INSERT -> List(it.newLine.strippedLen()){ " " }.joinToString("")
          CHANGE -> "${ANSI_YELLOW_BACKGROUND}${it.oldLine}${ANSI_RESET}" +
            it.oldLine.padEnd(maxOf(it.oldLine.strippedLen(), it.newLine.strippedLen())).drop(it.oldLine.length)
          DELETE -> "$ANSI_RED_BACKGROUND${it.oldLine}$ANSI_RESET"
          else -> it.oldLine
        }
      }.replace("&lt;", "<").replace("&gt;", ">") + "\n" +
      it.joinToString(" ") {
        when (it.tag) {
          INSERT -> it.newLine.redo(ANSI_GREEN_BACKGROUND)
          CHANGE -> it.newLine.redo(ANSI_YELLOW_BACKGROUND) +
            it.newLine.stripTags().let { nl -> nl.padEnd(maxOf(it.oldLine.strippedLen(), it.newLine.strippedLen())).drop(nl.length) }
          DELETE -> List(it.oldLine.strippedLen()) { " " }.joinToString("")
          else -> it.newLine.stripTags()
        }
      }.replace("&lt;", "<").replace("&gt;", ">")
    }.let {
      val tokenizedLines = it.lines()
      val minFirstChange = tokenizedLines.minOf { s -> s.indexOf("\u001B").let { if (it < 0) s.length else it } }
      val maxLastChange = tokenizedLines.minOf { it.length - (it.lastIndexOf("\u001B") + 5) }
      val prefix = tokenizedLines.first().substring(0, minFirstChange).tokenizeByWhitespace().takeLast(3).joinToString(" ")
      val suffix = tokenizedLines.first().let { it.substring(it.length - maxLastChange)}.tokenizeByWhitespace().take(3).joinToString(" ")
      tokenizedLines.joinToString("\n") { "... $prefix ${it.drop(minFirstChange).dropLast(maxLastChange)}$suffix ..." }
    }

// Print the side-by-side diff with ASCII colors and a border
// https://en.wikipedia.org/wiki/Box-drawing_character
fun prettyDiffHorizontal(
  left: Σᐩ, right: Σᐩ,
  leftHeading: Σᐩ = "original",
  rightHeading: Σᐩ = "new",
  maxLen: Int = 180, maxLines: Int = 200,
): Σᐩ {
  val sb = StringBuilder()
  val leftLines = left.replace("\t", "  ").replace("&lt;", "<").replace("&gt;", ">").lines()
  val rightLines = right.replace("\t", "  ").replace("&lt;", "<").replace("&gt;", ">").lines()
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

    val padLeft = max(leftHeading.visibleLen(), rows.maxOf { it.oldLine.visibleLen() }) + 3
    val padRight = max(rightHeading.visibleLen(), rows.maxOf { it.newLine.visibleLen() }) + 3

    val tlsep = "┌".padEnd(padLeft, '─')
    val trsep = "┬".padEnd(padRight, '─')
    sb.appendLine("$tlsep$trsep┐")
    sb.appendLine(
      "│ $leftHeading".padEnd(padLeft, ' ') +
        "│ $rightHeading".padEnd(padRight, ' ') + "│"
    )

    val lsep = "├".padEnd(padLeft, '─')
    val rsep = "┼".padEnd(padRight, '─')

    fun Σᐩ.adjust(len: Int) = padEnd(len + length - visibleLen() - 3, ' ')
    sb.appendLine("$lsep$rsep┤")
    rows.forEach { row ->
      sb.appendLine("│ ${row.oldLine.adjust(padLeft)} $ANSI_RESET│ ${row.newLine.adjust(padRight)} $ANSI_RESET│")
    }

    val blsep = "└".padEnd(padLeft, '─')
    val brsep = "┴".padEnd(padRight, '─')
    sb.appendLine("$blsep$brsep┘")
  }
  return sb.toString()
}

fun Σᐩ.splitByNonWordChars() = split(Regex("((?<=\\W)|(?=\\W))"))

// Filters for anything visible keyboard characters, excluding whitespace
fun Σᐩ.visibleChars() = filter { it in '!'..'~' }

fun Σᐩ.isANontrivialStatementWithBalancedBrackets(
  depth: Int = 2,
  statementCriteria: Σᐩ.() -> Boolean = { trim().endsWith(';') && hasBalancedBrackets() },
  parensAndDepth: Pair<Int, Int> = countBracketsAndMaxDepth(),
) = statementCriteria() && parensAndDepth.let { (p, d) -> p == 0 && depth < d }

fun Σᐩ.isBracket() = length == 1 && this in COMMON_BRACKETS

val p: (Char) -> Complex = { c ->
  val a = ".LO,KIMJUNHYBGTVFRCDEXSWZAQ".indexOf(c)
  Complex((a % 3) * 4.0, a - a.toDouble() / 3)
}

fun qwertyFingerTravelDist(x: Char, y: Char) =
  (p(x) - p(y)).abs() * 4.76

data class Complex(val real: Double, val imaginary: Double) {
  operator fun minus(other: Complex) =
    Complex(real - other.real, imaginary - other.imaginary)

  fun abs() = Math.sqrt(real * real + imaginary * imaginary)
}