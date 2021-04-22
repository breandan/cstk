package edu.mcgill.gymfs.disk

import info.debatty.java.stringsimilarity.interfaces.MetricStringDistance
import java.net.*
import java.nio.file.*
import kotlin.io.path.*

// Query in context
data class QIC(
  val query: String,
  val path: Path,
  val context: String,
  val offset: Int
)

// Searches through all files in the path for the query
fun Path.slowGrep(query: String, glob: String = "*"): List<QIC> =
  allFilesRecursively(glob).mapNotNull { path ->
    path.read()?.let { contents ->
      contents.extractConcordances(".*$query.*")
        .map { (cxt, idx) -> QIC(query, path, cxt, idx) }
    }
  }.flatten()

// Returns all files in the path matching the extension
fun Path.allFilesRecursively(glob: String = "*.$FILE_EXT"): List<Path> =
  (Files.newDirectoryStream(this).filter { it.toFile().isDirectory } +
    Files.newDirectoryStream(this, glob)).partition { Files.isDirectory(it) }
    .let { (dirs, files) ->
      files + dirs.map { it.allFilesRecursively(glob) }.flatten()
    }

// Returns a list of all code fragments in all paths and their locations
@OptIn(ExperimentalPathApi::class)
fun List<Path>.allCodeFragments(): List<Pair<Location, String>> =
  map { path ->
    path.readText().lines()
      .mapIndexed { lineNum, line -> lineNum to line }
      .filter { (_, l) -> l.isNotBlank() && l.any(Char::isLetterOrDigit) }
      .map { (ln, l) -> Location(path.toUri(), ln) to l.trim() }
//    .chunked(5).map { it.joinToString("\n") }
  }.flatten()

fun Path.read(start: Int = 0, end: Int = -1): String? =
  try { Files.readString(this) } catch (e: Exception) { null }
    ?.let { it.substring(start, if (end < 0) it.length else end) }

// Returns all substrings matching the query and their immediate context
fun String.extractConcordances(query: String): List<Pair<String, Int>> =
  Regex(query).findAll(this).map {
    val range = 0..length
    val (matchStart, matchEnd) =
      it.range.first.coerceIn(range) to (it.range.last + 1).coerceIn(range)
    substring(matchStart, matchEnd) to matchStart
  }.toList()

fun previewResult(query: String, loc: Location) =
  "[?=$query] ${loc.getContext(0).preview(query)}\t($loc)"

fun String.preview(query: String, window: Int = 10) =
  extractConcordances(query).map { (q, b) ->
    val range = 0..length
    substring((b - window).coerceIn(range), b) + "[?]" +
      substring(b + q.length, (b + q.length + window).coerceIn(range))
  }.joinToString("…", "…", "…") { it.trim() }

//https://github.com/huggingface/transformers/issues/1950#issuecomment-558770861
fun vectorize(query: String): DoubleArray =
  URL(SERVER_ADDRESS + URLEncoder.encode("$CODEBERT_CLS_TOKEN$query", "utf-8"))
    .readText().lines()
    .map { it.trim().replace("[", "").replace("]", "") }
    .map { it.split(" ").filter(String::isNotEmpty).map(String::toDouble) }
    .first().toDoubleArray()

fun tokenize(query: String) =
  URL(SERVER_ADDRESS + URLEncoder.encode(query, "utf-8")).readText().split(" ")

fun List<String>.sortedByDist(query: String, metric: MetricStringDistance) =
  sortedBy { metric.distance(it, query) }