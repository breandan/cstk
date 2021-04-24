package edu.mcgill.gymfs.disk

import org.apache.commons.vfs2.VFS
import java.io.*
import java.net.URI
import java.nio.charset.Charset
import java.nio.file.*
import java.util.zip.*
import kotlin.io.path.*

// Creates a mirror image of the HD path in memory
@OptIn(ExperimentalPathApi::class)
private fun Path.mirrorHDFS(imfs: FileSystem): Path {
  val jfsRoot = imfs.getPath(toString()).also { Files.createDirectories(it) }
  allFilesRecursively().toList().parallelStream()
    .filter { it.extension == FILE_EXT }
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

@OptIn(ExperimentalPathApi::class)
fun URI.allLines() =
  if (scheme == "file")
    Files.newBufferedReader(toPath()).lineSequence()
  else VFS.getManager().resolveFile(this).content
    .getString(Charset.defaultCharset()).lineSequence()