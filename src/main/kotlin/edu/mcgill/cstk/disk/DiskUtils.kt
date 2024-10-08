package edu.mcgill.cstk.disk

import edu.mcgill.cstk.utils.allCodeFragments
import org.apache.commons.vfs2.*
import java.io.*
import java.io.File
import java.net.URI
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.*
import java.nio.file.FileSystem
import java.util.zip.*
import kotlin.io.path.*
import kotlin.time.*

// Creates a mirror image of the HD path in memory
private fun URI.mirrorHDFS(imfs: FileSystem): Path {
  val jfsRoot = imfs.getPath(toString()).also { Files.createDirectories(it) }
  allFilesRecursively().map { it.toPath() }.toList().parallelStream()
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

fun File.unzip(
  filename: String = path.substringAfterLast('/').substringBeforeLast('.'),
  dirname: File = Files.createTempDirectory(filename).toFile()
): File = dirname.also {
  vfsManager.runCatching {
    val from = createFileSystem(resolveFile(path))
    toFileObject(dirname).copyFrom(from, VFS_SELECTOR)
  }.onFailure { it.printStackTrace() }
}

inline fun <reified T> T.serializeTo(path: File) =
  measureTimedValue {
//  Kryo().writeObject(Output(FileOutputStream(path)), this)
    println("Writing ${T::class.java.simpleName} to $path...")
    ObjectOutputStream(GZIPOutputStream(FileOutputStream(path)))
      .use { it.writeObject(this) }
  }.let {
    println("Wrote $path in ${it.duration}")
  }

inline fun <reified T> File.deserializeFrom(): T = measureTimedValue {
//  Kryo().readObject(Input(FileInputStream(this)), T::class.java)
  println("Reading ${T::class.java.simpleName} from $path...")
  ObjectInputStream(GZIPInputStream(FileInputStream(this)))
    .use { it.readObject() } as T
}.let {
  println("Read ${T::class.java.simpleName} in ${it.duration}")
  it.value
}

// Returns all files in the URI matching the extension
fun URI.allFilesRecursively(
  ext: String? = null,
  // Should we recurse into compressed files?
  readCompressed: Boolean = true
): Sequence<URI> =
  toPath().toFile().walkTopDown()
    .filter(File::isFile)
    .map(File::toURI)
    .map { if (readCompressed) readCompressedFile(it) else sequenceOf(it) }
    .flatten()
    .filter { ext == null || it.extension() == ext }.shuffled(DEFAULT_RAND)

fun readCompressedFile(path: URI) =
  if (path.extension() in setOf("tgz", "zip"))
    vfsManager.resolveFile("${path.extension()}:${path.path}")
      .runCatching { findFiles(VFS_SELECTOR).asSequence().map(FileObject::getURI) }
      .getOrDefault(sequenceOf())
  else sequenceOf(path)

fun URI.extension() = toString().substringAfterLast('.')
fun URI.suffix() = toString().substringAfterLast('/')

fun indexURI(src: URI, indexFn: (String, Concordance) -> Unit): Unit =
  when {
    src.scheme == TGZ_SCHEME || src.extension() == TGZ_SCHEME ->
      vfsManager
        .resolveFile("tgz:${src.path}")
        .runCatching {
          println("Indexing $name")
          findFiles(VFS_SELECTOR).asSequence()
            .map { it.uri }.allCodeFragments()
            .forEach { (location, line) -> indexFn(line, location) }
        }.let {
          println(
            if (it.isSuccess) "Indexed ${src.suffix()} " +
              "(${src.toPath().fileSize() / 1000}kb)"
            else "Failed to index $src due to ${it.exceptionOrNull()}"
          )
        }
    src.scheme == FILE_SCHEME ->
      try {
        (
          if (src.toPath().isDirectory()) src.allFilesRecursively()
          else sequenceOf(src.also { println("Indexing $src") })
        ).allCodeFragments()
          .forEach { (location, line) -> indexFn(line, location) }
      } catch (e: Exception) {
        System.err.println("Unreadable …$src due to ${e.message}")
      }
    else -> Unit
  }

val vfsManager = VFS.getManager()
fun FileSystemManager.readText(uri: URI): String = resolveFile(uri).content.getString(UTF_8)