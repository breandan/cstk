import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import java.io.*
import java.nio.file.*
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.zip.*
import kotlin.io.path.*
import kotlin.time.*

fun main(args: Array<String>) = Grepper().main(args)

@OptIn(ExperimentalTime::class)
class Grepper: CliktCommand() {
  val path by option("--path", help = "Root directory")
    .default(Paths.get("").toAbsolutePath().toString())

  val query by option("--query", help = "Query to find").default("match")

  val index by option("--index", help = "Prebuilt index file").default("")

  // Suffix trie multimap for (file, offset) pairs of matching prefixes
  val trie: ConcurrentSuffixTree<Queue<Pair<String, Int>>>
    by lazy { buildOrLoadIndex(File(index), Path.of(path)) }

  override fun run() {
    println("\nSearching index of size ${trie.size()} for [?]=[$query]...\n")

    measureTimedValue { search(query) }.let { (res, time) ->
      res.take(10).forEachIndexed { i, it ->
        println("$i.) " + it.getLine().chop(query))
      }
      println("\nFound ${res.size} results in $time")
    }
  }

  fun Pair<String, Int>.getLine() =
    Files.newBufferedReader(Path.of(first)).lineSequence().take(second + 1).last()

  fun search(query: String) = trie.getValuesForKeysContaining(query).flatten()
}

fun Path.grep(query: String, glob: String = "*"): List<QIC> =
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

// Query in context
data class QIC(
  val query: String,
  val path: Path,
  val context: String,
  val offset: Int
)

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

fun Any?.serialize(path: File) =
  ObjectOutputStream(GZIPOutputStream(FileOutputStream(path))).use { it.writeObject(this) }

fun deserialize(file: File): Any =
  ObjectInputStream(GZIPInputStream(FileInputStream(file))).use { it.readObject() }

// Creates a mirror image of the HD path in memory
@ExperimentalPathApi
private fun Path.mirrorHDFS(imfs: FileSystem): Path {
  val jfsRoot = imfs.getPath(toString()).also { Files.createDirectories(it) }
  allFilesRecursively().parallelStream().filter { it.extension == "kt" }
    .forEach { src ->
      try {
        val path = src.toAbsolutePath().toString()
        Files.createDirectories(imfs.getPath(path).parent)
        Files.copy(src, imfs.getPath(path))
      } catch (e: Exception) {
        System.err.println("Uncopyable …${src.fileName} due to ${e.message}")
      }
    }
  return jfsRoot
}

// Indexes all lines in all files in the path
fun indexPath(rootDir: Path): ConcurrentSuffixTree<Queue<Pair<String, Int>>> =
  ConcurrentSuffixTree<Queue<Pair<String, Int>>>(DefaultCharArrayNodeFactory())
    .also { trie ->
      rootDir.allFilesRecursively().parallelStream().forEach { src ->
        try {
          Files.readAllLines(src).forEachIndexed { lineIndex, line ->
            if (line.length < 500)
              ConcurrentLinkedQueue(listOf(src.toString() to lineIndex))
                .let { trie.putIfAbsent(line + 1, it)?.offer(it.first()) }
          }
        } catch (e: Exception) {
//        System.err.println("Unreadable …${src.fileName} due to ${e.message}")
        }
      }
    }

@OptIn(ExperimentalTime::class)
fun buildOrLoadIndex(
  index: File,
  rootDir: Path
): ConcurrentSuffixTree<Queue<Pair<String, Int>>> =
  if (!index.exists())
    measureTimedValue { indexPath(rootDir.also { println("Indexing $it") }) }
      .let { (trie, time) ->
        val file = if (index.name.isEmpty()) File("gymfs.idx") else index
        trie.also { it.serialize(file); println("Indexed in $time to: $file") }
      }
 else {
    println("Loading index from ${index.absolutePath}")
    deserialize(index) as ConcurrentSuffixTree<Queue<Pair<String, Int>>>
  }