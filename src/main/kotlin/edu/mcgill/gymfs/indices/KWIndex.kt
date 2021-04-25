package edu.mcgill.gymfs.indices

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import edu.mcgill.gymfs.disk.*
import java.io.File
import java.net.URI
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.time.*

fun buildOrLoadKWIndex(indexFile: File, rootDir: URI): KWIndex =
  if (!indexFile.exists())
    rebuildKWIndex(rootDir).apply { serializeTo(indexFile) }
  else indexFile.deserializeFrom()

// Indexes all lines in all files in the path

@OptIn(ExperimentalTime::class)
fun rebuildKWIndex(rootDir: URI): KWIndex =
  measureTimedValue {
    println("Rebuilding keyword index...")
    KWIndex(DefaultCharArrayNodeFactory()).apply {
      rootDir.allFilesRecursively().toList().parallelStream().forEach { src ->
        indexURI(src) { line, location -> indexLine(line, location) }
      }
    }
  }.let { println("Built keyword index in ${it.duration}"); it.value }

typealias KWIndex = ConcurrentSuffixTree<Queue<Location>>

val KWIndex.defaultFilename: String by lazy { "keyword.idx" }

fun KWIndex.indexLine(line: String, location: Location) {
    ConcurrentLinkedQueue(listOf(location))
      .let { putIfAbsent(line.take(500), it)?.offer(it.first()) }
}

fun main() {
  buildOrLoadKWIndex(
    indexFile = File(DEFAULT_KWINDEX_FILENAME),
    rootDir = File("src").toURI()
  )
}