package edu.mcgill.cstk.crawler

import com.gargoylesoftware.htmlunit.*
import com.gargoylesoftware.htmlunit.html.HtmlPage
import com.gargoylesoftware.htmlunit.javascript.JavaScriptEngine
import edu.mcgill.cstk.utils.execute
import java.io.File
import java.net.URL
import java.nio.file.*

fun main() {
  File("data").apply { mkdir() }
//  cloneGithub()
//  cloneGoogleCode()
  cloneGitlab()
//  cloneSelfHosted()
}

const val TO_TAKE = 100
const val GH_REPOS_FILE = "github.txt"
const val GC_REPOS_FILE = "gcode.txt"
const val GL_REPOS_FILE = "gitlab.txt"
const val SH_REPOS_FILE = "git.txt"

fun cloneGoogleCode(
  dir: String = "gcode".also {
    Files.createDirectories(Paths.get("data/$it"))
  }
) = File(GC_REPOS_FILE).readLines()
  .take(TO_TAKE).parallelStream().forEach {
    val repo = it.substringAfter("https://code.google.com/archive/p/")
    println("Downloading $repo")
    downloadFile(
      url = URL("https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/$repo/source-archive.zip"),
      filename = repo.replace("/", "_") + ".tgz"
    )
  }

fun cloneGitlab(
  dir: String = "gitlab".also {
    Files.createDirectories(Paths.get("data/$it"))
  }
) =  File(GL_REPOS_FILE).readLines()
  .take(10).forEach {
    val repo = it.substringAfter("https://gitlab.com/")
    val proj = repo.substringAfter('/')
    val tarUrl = URL("https://gitlab.com/$repo/-/archive/master/$proj-master.tar.gz")
    println("Downloading $tarUrl")
    wgetFile(url = tarUrl,
      filename = "data/gitlab_" + repo.replace("/", "_") + ".tgz")
  }

fun cloneSelfHosted(
  dir: String = "git".also {
    Files.createDirectories(Paths.get("data/$it"))
  }
) =
  File(GH_REPOS_FILE).readLines().take(TO_TAKE).parallelStream()
    .forEach { repo -> "git clone --depth=1 $repo".execute() }

fun cloneGithub(
  dir: String = "github".also {
    Files.createDirectories(Paths.get("data/$it"))
  }
) = File(GH_REPOS_FILE).readLines()
  .take(TO_TAKE).parallelStream().forEach {
    val repo = it.substringAfter("https://github.com/")
    println("Downloading $repo")
    downloadFile(
      url = URL("https://api.github.com/repos/$repo/tarball"),
      filename = "data/github_" + repo.replace("/", "_") + ".tgz"
    )
  }

fun wgetFile(url: URL, filename: String) = "wget $url -O $filename".execute()

fun downloadFile(url: URL, filename: String) =
  runCatching {
    try {
      val file = File(filename)
      val data = url.readBytes()
      file.writeBytes(data)
      println("Downloaded $url to ${file.path}")
    } catch (ex: Exception) { ex.printStackTrace(); throw ex }
  }