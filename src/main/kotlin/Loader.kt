import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.google.common.jimfs.*
import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import java.nio.file.*
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.system.measureTimeMillis

fun main(args: Array<String>) = Loader().main(args)

class Loader: CliktCommand() {
  val rootDir by option(
    "--path", help = "Root directory"
  ).default(Paths.get("").toAbsolutePath().toString())

  val query by option(
    "--query", help = "Query to find"
  ).default("match")

  // An in-memory file system mirroring the contents of rootDir
  val jfs = Jimfs.newFileSystem(Configuration.forCurrentPlatform())
  // Suffix trie multimap for (file, offset) pairs of matching prefixes
  val trie = ConcurrentSuffixTree<Queue<Pair<Path, Int>>>(DefaultCharArrayNodeFactory())

  override fun run() {
    println("Searching $rootDir for $query")
    val jfsRoot = Path.of(rootDir).mirrorHDFS()
    measureTimeMillis { indexFS(jfsRoot) }
      .let { println("Indexing took $it ms") }

    println("\nSearching grep for [?]=[$query]...\n")
    measureTimeMillis { jfsRoot.grep(query)
      .also { it.forEachIndexed { i, it -> println("$i.) " + it.context.chop(query)) } }
      .also { println("Grep found ${it.size} results") } }
      .let { println("Grep took $it ms") }

    println("\nSearching trie for [?]=[$query]...\n")
    measureTimeMillis { trie.getValuesForKeysContaining(query).flatten()
      .also { it.forEachIndexed { i, it -> println("$i.) " + it.getLine().chop(query)) } }
      .also { println("Trie found ${ it.size } results") } }
      .let { println("Trie took $it ms") }
  }

  // Indexes all lines in all files in the path
  private fun indexFS(jfsRoot: Path) {
    jfsRoot.allFilesRecursively().parallelStream().forEach { src ->
      try {
        Files.readAllLines(src).forEachIndexed { lineIndex, line ->
          if (line.length < 500)
            ConcurrentLinkedQueue(listOf(src to lineIndex))
              .let { trie.putIfAbsent(line + 1, it)?.offer(it.first()) }
        }
      } catch (e: Exception) {
//        System.err.println("Unreadable …${src.fileName} due to ${e.message}")
      }
    }
  }

  // Creates a mirror image of the HD path in memory
  private fun Path.mirrorHDFS(): Path {
    val jfsRoot = jfs.getPath(toString()).also { Files.createDirectories(it) }
    allFilesRecursively().parallelStream().forEach { src ->
      val path = src.toAbsolutePath().toString()
      Files.createDirectories(jfs.getPath(path).parent)
      Files.copy(src, jfs.getPath(path))
    }
    return jfsRoot
  }
}

fun Path.grep(query: String, glob: String = "*"): List<QIC> =
  allFilesRecursively(glob).mapNotNull { path ->
    path.read()?.let { contents ->
      contents.extractConcordances(".*$query.*").map { (cxt, idx) ->
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
      substring(b + q.length, (b + q.length + window).coerceIn(range)) }

fun Pair<Path, Int>.getLine() =
  Files.newBufferedReader(first).lineSequence().take(second + 1).last()