package edu.mcgill.gymfs.disk

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import org.apache.commons.vfs2.VFS
import java.io.*
import java.net.URI
import java.nio.charset.Charset
import java.nio.file.*
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.zip.*
import kotlin.io.path.*
import kotlin.time.*

// Creates a mirror image of the HD path in memory
@OptIn(ExperimentalPathApi::class)
private fun Path.mirrorHDFS(imfs: FileSystem): Path {
  val jfsRoot = imfs.getPath(toString()).also { Files.createDirectories(it) }
  allFilesRecursively().parallelStream().filter { it.extension == FILE_EXT }
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

fun Any?.serialize(path: File) =
  ObjectOutputStream(GZIPOutputStream(FileOutputStream(path)))
    .use { it.writeObject(this) }

fun File.deserialize(): Any =
  ObjectInputStream(GZIPInputStream(FileInputStream(this)))
    .use { it.readObject() }

@OptIn(ExperimentalTime::class)
fun buildOrLoadIndex(index: File, rootDir: Path): KWIndex =
  if (!index.exists())
    measureTimedValue { indexPath(rootDir.also { println("Indexing $it") }) }
      .let { (trie, time) ->
        val file = if (index.name.isEmpty()) File("keyword.idx") else index
        trie.also { it.serialize(file); println("Indexed in $time to: $file") }
      }
  else {
    println("Loading index from ${index.absolutePath}")
    index.deserialize() as KWIndex
  }

// Indexes all lines in all files in the path

fun indexPath(rootDir: Path): KWIndex =
  KWIndex(DefaultCharArrayNodeFactory()).apply {
    rootDir.allFilesRecursively().parallelStream().forEach { src ->
      if (src.endsWith("tgz")) indexCompressedFile(src)
      else indexUncompressedFile(src)
    }
  }

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
  }

typealias KWIndex = ConcurrentSuffixTree<Queue<Location>>

fun KWIndex.indexLine(line: String, location: Location) {
  if (line.length < 500)
    ConcurrentLinkedQueue(listOf(location))
      .let { putIfAbsent(line, it)?.offer(it.first()) }
}