package edu.mcgill.gymfs.github

import java.io.File
import java.net.URL

fun main() {
  File("data").apply { mkdir() }

  File("repositories.txt").readLines().forEach {
    val repo = it.substringAfter("https://github.com/").substringBefore(".git")
    print("Downloading $repo ")
    try {
      val file = File("data/" + repo.replace("/", "_"))
      val data = URL("https://api.github.com/repos/$repo/tarball").readBytes()
      file.writeBytes(data)
      print(" to ${file.path}")
    } catch (e: Exception) {}
  }
}