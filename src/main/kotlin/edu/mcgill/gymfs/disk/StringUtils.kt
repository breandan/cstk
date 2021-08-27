package edu.mcgill.gymfs.disk

import com.github.difflib.DiffUtils
import com.github.difflib.text.DiffRowGenerator
import edu.mcgill.gymfs.math.kantorovich
import info.debatty.java.stringsimilarity.Levenshtein
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import net.automatalib.automata.fsa.DFA
import org.apache.commons.lang3.StringUtils
import java.net.*
import java.nio.file.*
import kotlin.io.path.*
import kotlin.text.Charsets.UTF_8


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

val funKeywords = setOf("public ", "private ", "void ", "static ", "fun ")
val openParens = setOf('(', '{', '[')
val closeParens = setOf(')', '}', ']')

// Slices files into method-level chunks using a Dyck-1 language
fun Sequence<URI>.allMethods(): Sequence<String> = map { path ->
  path.allLines().fold(-1 to listOf<String>()) { (dyckSum, methods), line ->
    if (dyckSum < 0 && funKeywords.any { it in line } && "(" in line) {
      line.countBalancedBrackets() to methods + line
    } else if (dyckSum == 0) {
      if(line.isBlank()) -1 to methods else 0 to methods.put(line)
    } else if (dyckSum > 0) {
      dyckSum + line.countBalancedBrackets() to methods.put(line)
    } else {
      -1 to methods
    }
  }.second
}.flatten().map { it.trimIndent() }

fun List<String>.put(line: String) = dropLast(1) + (last() +"\n"+ line)

fun String.countBalancedBrackets(): Int =
  fold(0) { s, c -> if (c in openParens) s + 1 else if (c in closeParens) s - 1 else s }

fun URI.allLines(): Sequence<String> =
  when (scheme) {
    TGZ_SCHEME -> vfsManager.resolveFile(this)
      .content.getString(UTF_8).lineSequence()
    FILE_SCHEME -> toPath().let {
      if (it.extension == FILE_EXT && it.exists())
        it.readText().lineSequence()
      else emptySequence()
    }
    else -> emptySequence()
  }

fun Path.read(start: Int = 0, end: Int = -1): String? =
  try {
    Files.readString(this)
  } catch (e: Exception) {
    null
  }
    ?.let { it.substring(start, if (end < 0) it.length else end) }

// Returns all substrings matching the query and their immediate context
fun String.extractConcordances(query: String): Sequence<Pair<String, Int>> =
  Regex(query).findAll(this).map {
    val range = 0..length
    val (matchStart, matchEnd) =
      it.range.first.coerceIn(range) to (it.range.last + 1).coerceIn(range)
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

fun List<String>.filterByDFA(dfa: DFA<*, Char>) = filter {
  try {
    dfa.accepts(it.toCharArray().toList())
  } catch (exception: Exception) {
    false
  }
}

//https://github.com/huggingface/transformers/issues/1950#issuecomment-558770861
// Short sequence embedding: line-level
fun vectorize(query: String) = matrixize(query).first()

// Long sequence embedding: method level
fun matrixize(query: String): Array<DoubleArray> = makeQuery(query).lines()
  .map { it.trim().replace("[", "").replace("]", "") }
  .map { it.split(" ").filter(String::isNotEmpty).map(String::toDouble) }
  .map { it.toDoubleArray() }.toTypedArray()

// Expands MSK autoregressively until maxTokens reached or stopChar encountered
tailrec fun complete(
  query: String,
  lastToken: String = "",
  fullCompletion: String = lastToken + makeQuery(query),
  maxTokens: Int = 1,
  isStopChar: (Char) -> Boolean = { !it.isJavaIdentifierPart() }
): String =
  if (maxTokens == 1 || lastToken.any { isStopChar(it) }) fullCompletion
  else complete(
    query = query.replace(MSK, lastToken + MSK),
    lastToken = fullCompletion,
    maxTokens = maxTokens - 1
  )

fun getMaskSubstitution(original: String, revised: String) =
  (revised.lines() to original.lines()).let { (revisedLines, originalLines) ->
    if (revisedLines.size != originalLines.size) return@getMaskSubstitution ERR
    val originalLineIndex = originalLines.indexOfFirst { MSK in it }
    val originalLine = originalLines[originalLineIndex]
    val revisedLine = revisedLines[originalLineIndex]
    DiffUtils.diffInline(originalLine, revisedLine)
      .deltas.mapNotNull { delta ->
        delta.source.lines.zip(delta.target.lines)
          .mapNotNull { (source, target) -> if (source == MSK) target else null }
          .firstOrNull()
      }.firstOrNull()
      ?: ERR // Sometimes unable to recover mask b/c 🤗 mangles sequence
//.also { println("ERROR: \n\n"); printSideBySide(original, revised) }
  }

fun makeQuery(query: String = ""): String =
  getMaskSubstitution(query,
    URL(EMBEDDING_SERVER + URLEncoder.encode(query, "utf-8"))
      .readText()
  )

fun List<String>.sortedByDist(query: String, metric: MetricStringDistance) =
  sortedBy { metric.distance(it, query) }

object MetricCSNF: MetricStringDistance {
  /**
   * NF1, NF2 := CSNF(S1 + S2)
   * CSNFΔ(SN1, SN2) := LEVΔ(NF1, NF2)
   */
  override fun distance(s1: String, s2: String) =
    codeSnippetNormalForm(s1 to s2).let { (a, b) -> Levenshtein().distance(a, b) }

  fun codeSnippetNormalForm(pair: Pair<String, String>): Pair<String, String> =
    (StringUtils.splitByCharacterTypeCamelCase(pair.first).toList() to
      StringUtils.splitByCharacterTypeCamelCase(pair.second).toList()).let { (c, d) ->
      val vocab = (c.toSet() + d.toSet()).mapIndexed { i, s -> s to i }.toMap()
      c.map { vocab[it] }.joinToString("") to d.map { vocab[it] }.joinToString("")
    }
}

fun printSideBySide(
  left: String, right: String,
  leftHeading: String = "original",
  rightHeading: String = "new",
  maxLen: Int = 80, maxLines: Int = 20
) {
  val (leftLines, rightLines) = left.lines() to right.lines()
  if (leftLines.all { it.length < maxLen } && leftLines.size < maxLines) {
    val rows = DiffRowGenerator.create()
      .showInlineDiffs(true)
      .ignoreWhiteSpaces(true)
      .inlineDiffByWord(true)
      .lineNormalizer { it }
      .oldTag { _ -> "~" }
      .newTag { _ -> "**" }
      .build()
      .generateDiffRows(leftLines, rightLines)

    println(
      "| $leftHeading".padEnd(maxLen + 3, ' ') +
        "| $rightHeading".padEnd(maxLen + 3, ' ') + "|"
    )
    val sep = "|".padEnd(maxLen + 3, '-')
    println("$sep$sep|")
    rows.forEach { row ->
      println(
        "| " + row.oldLine.padEnd(maxLen, ' ') + " | " +
          row.newLine.padEnd(maxLen, ' ') + " |"
      )
    }
    println("\n")
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
  "Byte", "Short", "Int", "Long", "Float", "Double", "Boolean", "Char"
)

fun main() {
  println(kantorovich(matrixize("test  a ing 123"), matrixize("{}{{}{{{}}}{asdf g")))
  println(kantorovich(matrixize("test  a ing 123"), matrixize("open ing 222")))
}