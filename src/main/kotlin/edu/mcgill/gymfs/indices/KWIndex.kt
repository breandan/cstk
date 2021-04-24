package edu.mcgill.gymfs.indices

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import edu.mcgill.gymfs.disk.*
import org.apache.commons.vfs2.VFS
import java.io.File
import java.nio.file.Path
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.io.path.*
import kotlin.time.*

fun buildOrLoadKWIndex(index: File, rootDir: Path): KWIndex =
  if (!index.exists())
    buildKWIndex(rootDir).apply {
      println("Serializing to $index")
      serialize(index)
    }
  else index.apply { println("Loading $absolutePath") }.deserialize() as KWIndex

// Indexes all lines in all files in the path

@OptIn(ExperimentalTime::class, ExperimentalPathApi::class)
fun buildKWIndex(rootDir: Path): KWIndex =
  measureTimedValue {
    KWIndex(DefaultCharArrayNodeFactory()).apply {
      rootDir.allFilesRecursively().toList()
        .parallelStream().forEach { src ->
          when (src.extension) {
            "tgz" -> indexCompressedFile(src)
            FILE_EXT -> indexUncompressedFile(src)
          }
        }
    }
  }.let { println("Built keyword index in ${it.duration}"); it.value }

@OptIn(ExperimentalPathApi::class)
fun KWIndex.indexUncompressedFile(src: Path) = try {
  sequenceOf(src).allCodeFragments()
    .forEach { (location, line) -> indexLine(line, location) }
} catch (e: Exception) {
  System.err.println("Unreadable …${src.fileName} due to ${e.message}")
}

fun KWIndex.indexCompressedFile(src: Path) =
  VFS.getManager().resolveFile("tgz:$src").runCatching {
    findFiles(VFS_SELECTOR).asSequence().map { it.path }.allCodeFragments()
      .forEach { (location, line) -> indexLine(line, location) }
  }.also { println("Indexed $src") }

typealias KWIndex = ConcurrentSuffixTree<Queue<Location>>

val KWIndex.defaultFilename: String by lazy { "keyword.idx" }

fun KWIndex.indexLine(line: String, location: Location) {
  if (line.length < 500)
    ConcurrentLinkedQueue(listOf(location))
      .let { putIfAbsent(line, it)?.offer(it.first()) }
}

fun main(args: Array<String>) {
  buildOrLoadKWIndex(
    index = File(DEFAULT_KWINDEX_FILENAME),
    rootDir = Path.of("data")
  )
}