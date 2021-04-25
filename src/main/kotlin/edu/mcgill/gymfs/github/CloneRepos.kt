package edu.mcgill.gymfs.github

import java.io.File
import java.net.URL

fun main() {
  File("data").apply { mkdir() }

  File("repositories.txt").readLines().take(20).parallelStream().forEach {
    val repo = it.substringAfter("https://github.com/").substringBefore(".git")
    println("Downloading $repo ")
    runCatching {
      val file = File("data/" + repo.replace("/", "_") + ".tgz")
      val data = URL("https://api.github.com/repos/$repo/tarball").readBytes()
      file.writeBytes(data)
      println("Downloaded $repo to ${file.path}")
    }
  }
}