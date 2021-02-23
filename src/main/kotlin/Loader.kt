import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.google.common.jimfs.*
import java.nio.file.*

fun main(args: Array<String>) = Loader().main(args)

class Loader: CliktCommand() {
  val jfs = Jimfs.newFileSystem(Configuration.forCurrentPlatform())

  private val rootDir by option(
    "--path", help = "Root directory"
  ).default(Paths.get("").toAbsolutePath().toString())

  private val query by option(
    "--query", help = "Query to find"
  ).default("match")

  override fun run() {
    println("Searching $rootDir for $query")
    Files.createDirectories(jfs.getPath(rootDir))
    Path.of(rootDir).allFilesRecursively()
      .forEach { src ->
        val path = src.toAbsolutePath().toString()
        Files.createDirectories(jfs.getPath(path).parent)
        Files.copy(src, jfs.getPath(path))
      }

    val foo = jfs.getPath(rootDir)

    foo.grep(query).forEach { println(it) }
  }
}

fun Path.grep(query: String, glob: String = "*") =
  allFilesRecursively(glob).mapNotNull { path ->
    path.read()?.let { contents ->
      contents.extractConcordances(query).map { (cxt, idx) ->
        QIC(query, path, cxt, idx)
      }
    }
  }.flatten()

fun Path.allFilesRecursively(glob: String = "*"): List<Path> =
  Files.newDirectoryStream(this, glob).partition { Files.isDirectory(it) }
    .let { (dirs, files) ->
      files + dirs.map { it.allFilesRecursively(glob) }.flatten()
    }

// Query in context
data class QIC(
  val query: String,
  val path: Path,
  val context: String,
  val offset: Int
)

fun Path.read(start: Int = 0, end: Int = -1) =
  try {
    Files.readString(this).let {
      it.substring(start, if (end < 0) it.length else end)
    }
  } catch (e: Exception) {
    null
  }

fun String.extractConcordances(query: String? = null) =
  Regex(".*$query.*").findAll(this).map {
    val (matchStart, matchEnd) =
      (it.range.first).coerceAtLeast(0) to
        (it.range.last + 1).coerceAtMost(length)
    substring(matchStart, matchEnd) to matchStart
  }.toList()