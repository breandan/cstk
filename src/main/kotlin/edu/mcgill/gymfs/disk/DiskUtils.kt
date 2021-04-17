package edu.mcgill.gymfs.disk

import java.io.*
import java.nio.file.*
import java.util.zip.*
import kotlin.io.path.*

// Creates a mirror image of the HD path in memory
@ExperimentalPathApi
private fun Path.mirrorHDFS(imfs: FileSystem): Path {
  val jfsRoot = imfs.getPath(toString()).also { Files.createDirectories(it) }
  allFilesRecursively().parallelStream().filter { it.extension == "java" }
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