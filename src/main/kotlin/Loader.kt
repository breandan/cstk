import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.google.common.jimfs.*
import com.googlecode.concurrenttrees.radix.ConcurrentRadixTree
import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
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
  // Radix trie multimap for (file, offset) pairs of matching prefixes
  val trie = ConcurrentRadixTree<Queue<Pair<Path, Int>>>(DefaultCharArrayNodeFactory())

  override fun run() {
    println("Searching $rootDir for $query")
    val jfsRoot = Path.of(rootDir).mirrorHDFS()
    indexFS(jfsRoot)

    measureTimeMillis { jfsRoot.grep(query)
      .also { println("Grep found ${it.size} results") } }
      .let { println("Grep took $it ms") }

    measureTimeMillis { trie.getValuesForKeysStartingWith(query)
      .also { println("Trie found ${ it.flatten().size } results") } }
      .let { println("Trie took $it ms") }
  }

  // Indexes all words in all files in the path
  private fun indexFS(jfsRoot: Path) {
    jfsRoot.allFilesRecursively().parallelStream().forEach { src ->
      try {
        val content = Files.readString(src)
        Regex("(\\w+)").findAll(content).forEach {
            val result = src to it.range.first
            trie.putIfAbsent(it.value,
              ConcurrentLinkedQueue(listOf(result)))?.offer(result)
        }
      } catch (e: Exception) {}
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