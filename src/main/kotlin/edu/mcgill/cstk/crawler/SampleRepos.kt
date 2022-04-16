package edu.mcgill.cstk.crawler

import com.gargoylesoftware.htmlunit.*
import com.gargoylesoftware.htmlunit.BrowserVersion.CHROME
import com.gargoylesoftware.htmlunit.html.HtmlPage
import com.gargoylesoftware.htmlunit.javascript.JavaScriptEngine
import com.squareup.okhttp.*
import org.gitlab4j.api.GitLabApi
import org.kohsuke.github.*
import java.io.File
import java.net.URL
import java.util.stream.Collectors

fun main() {
  sampleGithub()
//  sampleGitlab()
//  sampleGoogleCode("java")
}

fun sampleGoogleCode(query: String) =
  (1..835).forEach { pg ->
    val url = "https://code.google.com/archive/search?q=$query&page=$pg"
    val text = fetchJSWebpage(url)

    val regex = Regex("href=\"/archive/p/([^\"]*)\"")
    val matches = regex.findAll(text).map { it.groups[1]!!.value }.toList()
    val uniqueMatches = matches.filter { !wasCopiedToGithub(it, "google") }
    val gcodePrefix = "https://code.google.com/archive/p/"
    uniqueMatches.forEach { File(GC_REPOS_FILE).appendText(gcodePrefix + it + "\n") }
    println("Page $pg of 835 contained ${uniqueMatches.size} / ${matches.size} repos not on GitHub")
  }

fun wasCopiedToGithub(name: String, origin: String) = try {
    GitHubBuilder()
      .withJwtToken(File(".ghtoken").readText())
      .build().searchRepositories()
      .q(name)
      .list()
      .take(1).isNotEmpty()
      //.any { it.name.lowercase() == name.lowercase() || origin.lowercase() in (it.description?.lowercase() ?: "") }
  } catch (e: Exception) { e.printStackTrace(); true } // Assume copied by default

fun fetchJSWebpage(url: String) =
  (WebClient(CHROME).apply {
    javaScriptEngine = JavaScriptEngine(this)
    options.isJavaScriptEnabled = true
    cookieManager.isCookiesEnabled = true
    options.isThrowExceptionOnScriptError = false
    options.isThrowExceptionOnFailingStatusCode = false
    ajaxController = NicelyResynchronizingAjaxController()
    cache.maxSize = 0
    options.isRedirectEnabled = true
  }.getPage(url) as HtmlPage).asXml()

fun sampleGithub() =
  GitHubBuilder()
    .withJwtToken(File(".ghtoken").readText())
    .build().searchRepositories()
    .language("kotlin")
    .forks(">=100")
    .size("1000..10000")
    .sort(GHRepositorySearchBuilder.Sort.STARS)
    .list()
    .take(2000)
    .parallelStream()
    .filter { repo ->
      repo.owner.type == "Organization" &&
        try {
          (repo.description + repo.readme.read().toString())
            .none { c -> c.code in 19968..40869 }
        } catch (e: Exception) { false }
    }
    .collect(Collectors.toList())
    .sortedBy { -it.openIssueCount }
    .forEach {
      File(GH_REPOS_FILE).appendText(it.httpTransportUrl.dropLast(4) + "\n")
      println(
        it.openIssueCount.toString().padEnd(6) +
          it.stargazersCount.toString().padEnd(9) +
          it.size.toString().padEnd(9) +
          it.httpTransportUrl
      )
    }

val gitlabApiToken = File(".gltoken").readText().trim()
val gitLabApi = GitLabApi("https://gitlab.com", gitlabApiToken)

fun sendGET(queryUrl: String) =
  try {
    OkHttpClient().newCall(
      Request.Builder().url(queryUrl)
        .header("PRIVATE-TOKEN", gitlabApiToken)
        .build()
    ).execute()
    .let { if (it.isSuccessful) it.body().string() else "" }
  } catch (e: Exception) { "" }

fun sampleGitlab() {
  val exclude = mutableSetOf<String>()

  val queries = listOf("*", "java+%7C+android+-javascript") +
    ('a'..'z').map { it.toString() }

  for (query in queries) {
    var noError = true
    var i = 1

    while (noError) {
      val queryUrl =
        "https://gitlab.com/api/v4/search?scope=projects&search=$query&per_page=96&page=${i++}"
      val text = sendGET(queryUrl)
      if (text.length < 200) noError = false

      val regex = Regex("https://gitlab.com/[^/]+?/[^/]+?\\.git")
      val matches = regex.findAll(text).map { it.value.dropLast(4).substringAfter("https://gitlab.com/") }.toList()
      if (matches.isEmpty()) noError = false
      val uniqueMatches = matches.filter { url ->
        val name = url.substringAfterLast('/')
        val owner = url.substringBeforeLast('/')
        (exclude.add(name) && exclude.add(owner)) &&
          shouldBeIncludedFromGitlab(url) &&
          !wasCopiedToGithub(name, "gitlab") &&
          !wasCopiedToGithub(url, "gitlab")
      }

      println("$query (pg. $i/n contained ${uniqueMatches.size}/${matches.size} uniques)")

      uniqueMatches.forEach { url ->
        val dedupedRepo = "\nhttps://gitlab.com/$url"
        File(GL_REPOS_FILE).appendText(dedupedRepo)
        println(dedupedRepo)
      }
    }
  }
}

fun shouldBeIncludedFromGitlab(
  repo: String,
  lang: String = "Java",
  strsToExclude: Set<String> = setOf(
    "The repository for this project is empty",
    "Exported from",
    "forked_from_link",
    "github",
    "mirror",
  ),
) = try {
  val text = URL("https://gitlab.com/${repo}").readText().drop(3500)
  strsToExclude.none { it in text } &&
    "repository-language-bar-tooltip-language&quot;&gt;$lang&lt;" in text
} catch (e: Exception) { false }