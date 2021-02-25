package edu.mcgill.gymfs

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import java.io.*
import java.nio.file.*
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.zip.*
import kotlin.io.path.*
import kotlin.time.*

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

fun Any?.serialize(path: File) =
  ObjectOutputStream(GZIPOutputStream(FileOutputStream(path))).use { it.writeObject(this) }

fun deserialize(file: File): Any =
  ObjectInputStream(GZIPInputStream(FileInputStream(file))).use { it.readObject() }
