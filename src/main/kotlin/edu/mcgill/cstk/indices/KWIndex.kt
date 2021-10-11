package edu.mcgill.cstk.indices

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import edu.mcgill.cstk.disk.*
import java.io.File
import java.net.URI
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.time.*

fun buildOrLoadKWIndex(
  indexFile: File = File(DEFAULT_KNNINDEX_FILENAME),
  rootDir: URI = TEST_DIR
): KWIndex =
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
        println("Finished indexing $src")
      }
    }
  }.let { println("Built keyword index in ${it.duration}"); it.value }

typealias KWIndex = ConcurrentSuffixTree<Queue<Concordance>>

val KWIndex.defaultFilename: String by lazy { "keyword.idx" }

fun KWIndex.indexLine(line: String, location: Concordance) {
    ConcurrentLinkedQueue(listOf(location)).let {
      line.split(DELIMITER).filter { it.isNotBlank() }
        .forEach { token -> putIfAbsent(token, it)?.offer(it.first()) }
    }
}

fun main() {
  buildOrLoadKWIndex(
    indexFile = File(DEFAULT_KWINDEX_FILENAME),
    rootDir = File("data").toURI()
  )
}