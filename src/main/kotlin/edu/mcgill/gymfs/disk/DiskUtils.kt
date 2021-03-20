package edu.mcgill.gymfs

import com.googlecode.concurrenttrees.radix.node.concrete.DefaultCharArrayNodeFactory
import com.googlecode.concurrenttrees.suffix.ConcurrentSuffixTree
import edu.mcgill.gymfs.disk.allFilesRecursively
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

