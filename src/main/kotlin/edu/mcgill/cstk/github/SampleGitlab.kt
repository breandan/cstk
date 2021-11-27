package edu.mcgill.cstk.github

import com.squareup.okhttp.*
import java.io.*
import java.net.URL
import java.util.concurrent.TimeUnit

fun main() {
  val exclude = mutableSetOf<String>()
  val gitlabApiToken = File(".gltoken").readText().trim()

  for (query in listOf("*", "java+%7C+android+-javascript")) {
  var noError = true
  var i = 1

  while (noError) {
    val queryUrl =
      "https://gitlab.com/api/v4/search?scope=projects&search=$query&per_page=100&page=${i++}"

    val request = Request.Builder().url(queryUrl)
      .header("PRIVATE-TOKEN", gitlabApiToken)
      .build()

    val client = OkHttpClient()
    client.newCall(request).enqueue(object: Callback {
      override fun onFailure(request: Request?, e: IOException?) {
        noError = false
      }

      override fun onResponse(response: Response?) {
        val text = response?.body()?.string()
        if (text == null || text.length < 200) {
          noError = true; return
        }
        println(response)
        val regex = Regex("https://gitlab.com/[^/]+?/[^/]+?\\.git")
        val matches = regex.find(text)
        var match = matches?.next()
        if (match == null) { noError = true; return }
        while (match != null) {
          val url =
            match.value.dropLast(4).substringAfter("https://gitlab.com/")
          if (
            exclude.add(url.substringAfter('/')) &&
            exclude.add(url.substringBefore('/')) &&
            !isGitlabRepoOnGithub(url) &&
            !shouldBeExcludedFromGitlab(url)
          ) println("https://gitlab.com/$url")

          match = match.next()
        }
      }
    })

    TimeUnit.SECONDS.sleep(13)
  }
}
}

fun shouldBeExcludedFromGitlab(
  repo: String, strsToExclude: Set<String> = setOf(
    "The repository for this project is empty",
    "forked_from_link",
    "github",
    "mirror",
  )
): Boolean {
  try {
    val text = URL("https://gitlab.com/${repo}").readText().drop(3500)
    for (str in strsToExclude) if (str in text) return true
    if ("repository-language-bar-tooltip-language&quot;&gt;Java&lt;" in text) return false
    return true
  } catch (e: Exception) {
    return true
  }
}

fun isGitlabRepoOnGithub(repo: String) =
  try {
    URL("https://github.com/${repo}").readText()
    true
  } catch (e: Exception) {
    false
  }
