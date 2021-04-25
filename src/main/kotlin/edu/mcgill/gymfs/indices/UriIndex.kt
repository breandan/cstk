package edu.mcgill.gymfs.indices

import edu.mcgill.gymfs.disk.*
import java.net.URI
import kotlin.time.*

// Doesn't seem to be much more space efficient than storing URI directly
object UriIndex {
  operator fun get(hashCode: Int) = hashtable[hashCode] ?: URI("MISSING")

  @OptIn(ExperimentalTime::class)
  val hashtable by lazy {
    measureTimedValue {
      ROOT_DIR.allFilesRecursively(FILE_EXT, walkIntoCompressedFiles = true)
        .also { it.filter { it.scheme == "file" }.take(10).forEach { println(it) } }
        .map { it.hashCode() to it }.toMap()
    }.run { println("Indexed URIs in ${duration.inWholeSeconds}s"); value }
  }
}