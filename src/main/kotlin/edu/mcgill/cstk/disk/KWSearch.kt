package edu.mcgill.cstk.disk

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import edu.mcgill.cstk.disk.indices.*
import edu.mcgill.cstk.utils.previewResult
import java.io.File
import java.net.URI
import kotlin.time.*

class TrieSearch: CliktCommand() {
  val path by option("--path", help = "Root directory")
    .default(TEST_DIR.toString())

  val query by option("--query", help = "Query to find").default("match")

  val index by option("--index", help = "Prebuilt index file")
    .default(DEFAULT_KWINDEX_FILENAME)

  // Suffix trie multimap for (file, offset) pairs of matching prefixes
  val trie: KWIndex by lazy { buildOrLoadKWIndex(File(index), URI(path)) }

  @OptIn(ExperimentalTime::class)
  override fun run() {
    println("\nSearching index of size ${trie.size()} for [?]=[$query]…\n")

    measureTimedValue { trie.search(query) }.let { (res, time) ->
      res.take(10).forEachIndexed { i, it ->
        println("$i.) ${previewResult(query, it)}")
        val nextLocations = it.expand(trie)
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