package edu.mcgill.gymfs.disk

import info.debatty.java.stringsimilarity.MetricLCS
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

fun Path.slowGrep(query: String, glob: String = "*"): List<QIC> =
  allFilesRecursively(glob).mapNotNull { path ->
    path.read()?.let { contents ->
      contents.extractConcordances(".*$query.*")
        .map { (cxt, idx) -> QIC(query, path, cxt, idx) }
    }
  }.flatten()

fun Path.allFilesRecursively(glob: String = FILE_EXT): List<Path> =
  (Files.newDirectoryStream(this).filter { it.toFile().isDirectory } +
    Files.newDirectoryStream(this, glob))
    .partition { Files.isDirectory(it) }
    .let { (dirs, files) ->
      files + dirs.map { it.allFilesRecursively(glob) }.flatten()
    }

@OptIn(ExperimentalPathApi::class)
fun List<Path>.allCodeFragments() = map { path ->
  path.readText().lines().filter { it.isNotBlank() }
    .mapIndexed { i, it -> Location(path.toUri(), i) to it.trim() }
//    .chunked(5).map { it.joinToString("\n") }
}.flatten()

fun Path.read(start: Int = 0, end: Int = -1) =
  try { Files.readString(this) } catch (e: Exception) { null }
    ?.let { it.substring(start, if (end < 0) it.length else end) }

fun String.extractConcordances(query: String) =
  Regex(query).findAll(this).map {
    val range = 0..length
    val (matchStart, matchEnd) =
      it.range.first.coerceIn(range) to (it.range.last + 1).coerceIn(range)
    substring(matchStart, matchEnd) to matchStart
  }.toList()

fun String.preview(query: String, window: Int = 10) =
  extractConcordances(query).map { (q, b) ->
    val range = 0..length
    substring((b - window).coerceIn(range), b) + "[?]" +
      substring(b + q.length, (b + q.length + window).coerceIn(range))
  }.joinToString("…", "…", "…") { it.trim() }

fun vectorize(query: String): FloatArray =
  URL(SERVER_ADDRESS + URLEncoder.encode(query, "utf-8")).readText().lines()
  .map { it.trim().replace("[", "").replace("]", "") }
  .map { it.split(" ").filter(String::isNotEmpty).map { it.toFloat() } }
  .flatten().toFloatArray().copyOf(512)

fun tokenize(query: String) =
  URL(SERVER_ADDRESS + URLEncoder.encode(query, "utf-8")).readText().split(" ")