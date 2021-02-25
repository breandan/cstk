package edu.mcgill.gymfs

import java.nio.file.*

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

fun Path.allFilesRecursively(glob: String = "*"): List<Path> =
  Files.newDirectoryStream(this, glob).partition { Files.isDirectory(it) }
    .let { (dirs, files) ->
      files + dirs.map { it.allFilesRecursively(glob) }.flatten()
    }

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

fun String.chop(query: String, window: Int = 10) =
  extractConcordances(query).joinToString("…", "…", "…") { (q, b) ->
    val range = 0..length
    substring((b - window).coerceIn(range), b) + "[?]" +
      substring(b + q.length, (b + q.length + window).coerceIn(range))
  }