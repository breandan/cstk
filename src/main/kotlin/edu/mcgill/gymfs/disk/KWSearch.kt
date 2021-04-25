package edu.mcgill.gymfs.disk

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import edu.mcgill.gymfs.indices.buildOrLoadKWIndex
import java.io.File
import java.net.URI
import java.nio.file.*
import java.util.*
import kotlin.time.*

class TrieSearch: CliktCommand() {
  val path by option("--path", help = "Root directory")
    .default(TEST_DIR.toString())

  val query by option("--query", help = "Query to find").default("match")

  val index by option("--index", help = "Prebuilt index file")
    .default(DEFAULT_KWINDEX_FILENAME)

  // Suffix trie multimap for (file, offset) pairs of matching prefixes
  val trie: ConcurrentSuffixTree<Queue<Location>>
    by lazy { buildOrLoadKWIndex(File(index), URI(path)) }

  fun search(query: String): List<Location> =
    trie.getValuesForKeysContaining(query).flatten()

  @OptIn(ExperimentalTime::class)
  override fun run() {
    println("\nSearching index of size ${trie.size()} for [?]=[$query]…\n")

    measureTimedValue { search(query) }.let { (res, time) ->
      res.take(10).forEachIndexed { i, it ->
        println("$i.) ${previewResult(query, it)}")
        val nextLocations = it.expand(this@TrieSearch)
        println("Next locations:")
        nextLocations.forEachIndexed { index, (query, loc) ->
          println("\t$index.) ${previewResult(query, loc)}")
        }
        println()
      }
      println("\nFound ${res.size} results in $time")
    }
  }
}

fun main(args: Array<String>) = TrieSearch().main(args)
//fun main() =
//  TrieSearch().main(
//    arrayOf(
//      "--query=test", "--index=mini_github.idx",
//      "--path=/home/breandan/IdeaProjects/gym-fs"
//    )
//  )