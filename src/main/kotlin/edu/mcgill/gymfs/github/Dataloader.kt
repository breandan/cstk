package edu.mcgill.gymfs.github

import edu.mcgill.gymfs.disk.FILE_EXT
import org.apache.commons.vfs2.*
import java.io.File

val JAVA_FILES = FileExtensionSelector(setOf(FILE_EXT))

fun main() {
  //TODO: build keyword/KNN index on each git repo
  File("data").walk().filter { it.isFile }.forEach {
    VFS.getManager().resolveFile("tgz:${it.absolutePath}")
      .findFiles(JAVA_FILES).toList().forEach { file ->
        println(file)
//        println(it.content.inputStream.readBytes()
//          .toString(Charset.defaultCharset()).lines().take(3).joinToString("\n"))
      }
  }
}