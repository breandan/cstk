import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import java.io.File
import java.nio.file.*
import java.util.*
import kotlin.time.*

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
        println("$i.) " + it.getLine().chop(query) + "\nFrom: $it")
      }
      println("\nFound ${res.size} results in $time")
    }
  }

  fun Pair<String, Int>.getLine() =
    Files.newBufferedReader(Path.of(first)).lineSequence().take(second + 1).last()

  fun search(query: String) = trie.getValuesForKeysContaining(query).flatten()
}

fun main(args: Array<String>) = Grepper().main(args)