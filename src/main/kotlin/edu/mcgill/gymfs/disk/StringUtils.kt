package edu.mcgill.gymfs.disk

import edu.mcgill.gymfs.math.kantorovich
import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import net.automatalib.automata.fsa.DFA
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
  try { Files.readString(this) } catch (e: Exception) { null }
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
fun vectorize(query: String): DoubleArray =
  URL(SERVER_ADDRESS + URLEncoder.encode("$CODEBERT_CLS_TOKEN$query", "utf-8"))
    .readText().lines()
    .map { it.trim().replace("[", "").replace("]", "") }
    .map { it.split(" ").filter(String::isNotEmpty).map(String::toDouble) }
    .first().toDoubleArray()

// Sentence embedding
fun matrixize(query: String): Array<DoubleArray> =
  URL(
    SERVER_ADDRESS +
      URLEncoder.encode("$CODEBERT_BOS_TOKEN$query$CODEBERT_EOS_TOKEN", "utf-8")
  ).readText().lines()
    .map { it.trim().replace("[", "").replace("]", "") }
    .map { it.split(" ").filter(String::isNotEmpty).map(String::toDouble) }
    .map { it.toDoubleArray() }
    .toTypedArray()

fun tokenize(query: String) =
  URL(SERVER_ADDRESS + URLEncoder.encode(query, "utf-8")).readText().split(" ")

fun List<String>.sortedByDist(query: String, metric: MetricStringDistance) =
  sortedBy { metric.distance(it, query) }

fun synthesizeRegex(vararg strings: String) =
  ProcessBuilder("./grex", *strings).start()
    .inputStream.reader(UTF_8)
    .use { Regex(it.readText()) }

fun main() {
  println(kantorovich(matrixize("test  a ing 123"), matrixize("{}{{}{{{}}}{asdf g")))
  println(kantorovich(matrixize("test  a ing 123"), matrixize("open ing 222")))
}