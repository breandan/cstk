package edu.mcgill.gymfs.disk

import org.apache.commons.io.FileUtils.toFile
import org.apache.commons.vfs2.VFS
import java.io.*
import java.net.URI
import java.nio.file.*
import java.util.zip.*
import kotlin.io.path.*

// Creates a mirror image of the HD path in memory
private fun URI.mirrorHDFS(imfs: FileSystem): Path {
  val jfsRoot = imfs.getPath(toString()).also { Files.createDirectories(it) }
  allFilesRecursively(FILE_EXT).map { it.toPath() }.toList().parallelStream()
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

inline fun <reified T> T.serializeTo(path: File) =
//  Kryo().writeObject(Output(FileOutputStream(path)), this)
ObjectOutputStream(GZIPOutputStream(FileOutputStream(path)))
.use { it.writeObject(this) }

inline fun <reified T> File.deserializeFrom(): T =
//  Kryo().readObject(Input(FileInputStream(this)), T::class.java)
ObjectInputStream(GZIPInputStream(FileInputStream(this)))
.use { it.readObject() } as T

// Returns all files in the path matching the extension
fun URI.allFilesRecursively(ext: String? = null): Sequence<URI> =
  toPath().toFile().walkTopDown()
    .filter { it.isFile }
    .let { files ->
      ext?.let { ext -> files.filter { it.extension == ext } } ?: files
    }.map { it.toURI() }
//      toFile().walkTopDown().filter { it.extension == ext }.map { it.toURI() }

fun indexURI(src: URI, indexFn: (String, Location) -> Unit): Unit =
  when (src.scheme) {
    // TODO: HTTP_SCHEME?
    TGZ_SCHEME -> vfsManager
      .resolveFile("tgz:${src.path}")
      .runCatching {
        findFiles(VFS_SELECTOR).asSequence()
          .map { it.uri }.allCodeFragments()
          .forEach { (location, line) -> indexFn(line, location) }
      }.let {
        println(
          if (it.isSuccess) "Indexed ${"$src".substringAfterLast('/')} " +
            "(${src.toPath().fileSize() / 1000}kb)"
          else "Failed to index $src due to ${it.exceptionOrNull()}"
        )
      }
    FILE_SCHEME -> try {
      (if (src.toPath().isDirectory())
        src.allFilesRecursively()
      else sequenceOf(src))
        .allCodeFragments()
        .forEach { (location, line) -> indexFn(line, location) }
      println("Indexed $src")
    } catch (e: Exception) {
      System.err.println("Unreadable …$src due to ${e.apply {printStackTrace() }.message}")
    }
    else -> Unit
  }

val vfsManager = VFS.getManager()