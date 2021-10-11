package edu.mcgill.cstk.github

import edu.mcgill.cstk.disk.*
import java.io.File
import java.net.URL

fun main() =
  File(MINIGITHUB_REPOS_FILE).readLines()
    .take(MINIGITHUB_SIZE).parallelStream().forEach {
    val repo = it.substringAfter("https://github.com/").substringBefore(".git")
    println("Downloading $repo ")
    runCatching {
      File("data").apply { mkdir() }
      val file = File("data/" + repo.replace("/", "_") + ".tgz")
      val data = URL("https://api.github.com/repos/$repo/tarball").readBytes()
      file.writeBytes(data)
      println("Downloaded $repo to ${file.path}")
    }
  }