package edu.mcgill.gymfs.disk

import edu.mcgill.gymfs.math.kantorovich
import info.debatty.java.stringsimilarity.Levenshtein
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import net.automatalib.automata.fsa.DFA
import net.sf.extjwnl.data.PointerUtils
import net.sf.extjwnl.dictionary.Dictionary
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
fun matrixize(query: String): Array<DoubleArray> = query(query).lines()
  .map { it.trim().replace("[", "").replace("]", "") }
  .map { it.split(" ").filter(String::isNotEmpty).map(String::toDouble) }
  .map { it.toDoubleArray() }.toTypedArray()

tailrec fun complete(
  query: String, tokens: Int = 1,
  reply: String = query(query)
): String = if (tokens == 1) reply else complete(query = reply + MSK, tokens = tokens - 1)

fun query(query: String = ""): String =
  URL(EMBEDDING_SERVER + URLEncoder.encode(query, "utf-8")).readText()

val dict = Dictionary.getDefaultResourceInstance();

// Returns single-word synonyms
fun synonyms(word: String): Set<String> =
  dict.lookupAllIndexWords(word).indexWordArray.map {
    it.senses.map { sense ->
      PointerUtils.getDirectHypernyms(sense).map { it.synset.words }
    }
  }.flatten().flatten().flatten()
    .mapNotNull { it.lemma }.filter { !it.contains(" ") }.toSet()

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
  "suspend", "tailrec", "vararg", "field", "it"
)

fun main() {
  println(kantorovich(matrixize("test  a ing 123"), matrixize("{}{{}{{{}}}{asdf g")))
  println(kantorovich(matrixize("test  a ing 123"), matrixize("open ing 222")))
}