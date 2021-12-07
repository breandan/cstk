package edu.mcgill.cstk.crawler

import edu.mcgill.cstk.utils.execute
import java.io.File
import java.net.URL
import java.nio.file.*
import kotlin.io.path.*

fun main() {
  File("data").apply { mkdir() }
//  cloneGithub()
  cloneGoogleCode()
//  cloneGitlab()
//  cloneSelfHosted()
}

const val TO_TAKE = 100
const val GH_REPOS_FILE = "github.txt"
const val GC_REPOS_FILE = "gcode.txt"
const val GL_REPOS_FILE = "gitlab.txt"
const val SH_REPOS_FILE = "git.txt"

fun makeDataDir(dir: String) = Paths.get("data/$dir")
  .let { if (it.exists()) it.pathString else Files.createDirectories(it).pathString }

fun cloneGoogleCode(dir: String = makeDataDir("gcode")) =
  File(GC_REPOS_FILE).readLines().take(TO_TAKE).forEach {
    val repo = it.substringAfter("https://code.google.com/archive/p/")
    println("Downloading $it")
    downloadFile(
      url = URL("https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/$repo/source-archive.zip"),
      filename = "$dir/" + repo.replace("/", "_") + ".zip"
    )
  }

fun cloneGitlab(
  dir: String = makeDataDir("gitlab"),
  gitlabPrefix: String = "https://gitlab.com/"
) = File(GL_REPOS_FILE).readLines().take(10).forEach {
  val repo = it.substringAfter(gitlabPrefix)
  val proj = repo.substringAfter('/')
  println("Downloading $it")
  wgetFile(
    url = URL("$gitlabPrefix$repo/-/archive/master/$proj-master.tar.gz"),
    filename = "$dir/" + repo.replace("/", "_") + ".tgz"
  )
}

fun cloneSelfHosted(dir: String = makeDataDir("git")) =
  File(GH_REPOS_FILE).readLines().take(TO_TAKE)
    .forEach { repo -> "git clone --depth=1 $repo".execute() }

fun cloneGithub(dir: String = makeDataDir("github")) =
  File(GH_REPOS_FILE).readLines().take(TO_TAKE).forEach {
    val repo = it.substringAfter("https://github.com/")
    println("Downloading $it")
    downloadFile(
      url = URL("https://api.github.com/repos/$repo/tarball"),
      filename = "$dir/" + repo.replace("/", "_") + ".tgz"
    )
  }

fun wgetFile(url: URL, filename: String) = "wget $url -O $filename".execute()

fun downloadFile(url: URL, filename: String) = try {
  File(filename).writeBytes(url.readBytes())
  println("Downloaded $url to $filename")
} catch (ex: Exception) {
  ex.printStackTrace()
}