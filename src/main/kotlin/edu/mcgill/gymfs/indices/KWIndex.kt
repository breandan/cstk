package edu.mcgill.gymfs.disk

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import org.apache.commons.vfs2.VFS
import java.io.File
import java.nio.charset.Charset
import java.nio.file.Path
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.io.path.*
import kotlin.time.*


fun buildOrLoadKWIndex(index: File, rootDir: Path): KWIndex =
  if (!index.exists()) buildKWIndex(rootDir).also { it.serialize(index) }
  else index.also { println("Loading index from ${it.absolutePath}") }
    .deserialize() as KWIndex

// Indexes all lines in all files in the path

@OptIn(ExperimentalTime::class, ExperimentalPathApi::class)
fun buildKWIndex(rootDir: Path): KWIndex =
  measureTimedValue {
    KWIndex(DefaultCharArrayNodeFactory()).apply {
      rootDir.allFilesRecursively().parallelStream().forEach { src ->
        when (src.extension) {
          "tgz" -> indexCompressedFile(src)
          FILE_EXT -> indexUncompressedFile(src)
        }
      }
    }
  }.let { println(it.duration); it.value }

@OptIn(ExperimentalPathApi::class)
fun KWIndex.indexUncompressedFile(src: Path) = try {
  src.readLines().forEachIndexed { lineIndex, line ->
    indexLine(line, Location(src.toUri(), lineIndex))
  }
} catch (e: Exception) {
  System.err.println("Unreadable …${src.fileName} due to ${e.message}")
}

fun KWIndex.indexCompressedFile(src: Path) =
  VFS.getManager().resolveFile("tgz:$src").runCatching {
    findFiles(VFS_SELECTOR).toList().forEach { file ->
      file.content.getString(Charset.defaultCharset())
        .lines().forEachIndexed { lineIndex, line ->
          indexLine(line, Location(src.toUri(), lineIndex))
        }
    }
  }.also { println("Indexed $src") }

typealias KWIndex = ConcurrentSuffixTree<Queue<Location>>

fun KWIndex.indexLine(line: String, location: Location) {
  if (line.length < 500)
    ConcurrentLinkedQueue(listOf(location))
      .let { putIfAbsent(line, it)?.offer(it.first()) }
}