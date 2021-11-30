package edu.mcgill.cstk.crawler

import com.gargoylesoftware.htmlunit.*
import com.gargoylesoftware.htmlunit.html.HtmlPage
import com.gargoylesoftware.htmlunit.javascript.JavaScriptEngine
import com.squareup.okhttp.*
import org.gitlab4j.api.GitLabApi
import org.kohsuke.github.*
import java.io.*
import java.net.URL
import java.util.concurrent.TimeUnit
import java.util.stream.Collectors

fun main() {
//  sampleGithub()
  sampleGitlab()
//  sampleGoogleCode()
}

fun sampleGoogleCode() =
  (39..835).forEach { pg ->
    val url = "https://code.google.com/archive/search?q=java&page=$pg"
    val text = fetchJSWebpage(url)

    val regex = Regex("href=\"/archive/p/([^\"]*)\"")
    val matches = regex.findAll(text).map { it.groups[1]!!.value }.toList()
    val uniqueMatches = matches.filter { !isRepoOnGitHub(it) }
    val gcodePrefix = "https://code.google.com/archive/p/"
    uniqueMatches.forEach { File(GC_REPOS_FILE).appendText(gcodePrefix + it + "\n") }
    println("Page $pg of 835 contained ${uniqueMatches.size} / ${matches.size} repos not on GitHub")
  }

fun isRepoOnGitHub(name: String) =
//  "We couldn’t find any repositories matching" !in fetchJSWebpage("https://github.com/search?q=$name").also { println(it) }
  try {
    GitHubBuilder()
      .withJwtToken(File(".ghtoken").readText())
      .build().searchRepositories()
      .q(name)
      .list()
      .take(1).isNotEmpty()
  } catch (e :Exception) { e.printStackTrace();false }

fun fetchJSWebpage(url: String) =
  (WebClient(BrowserVersion.CHROME).apply {
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
        } catch (e: Exception) {
          false
        }
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
  OkHttpClient().newCall(
    Request.Builder().url(queryUrl)
      .header("PRIVATE-TOKEN", gitlabApiToken)
      .build()
    ).execute().let { response ->
    if(response.isSuccessful) response.body().toString() else ""
  }

fun sampleGitlab() {
  val exclude = mutableSetOf<String>()

  val queries = //listOf("*", "java+%7C+android+-javascript") +
    ('a'..'z').map { it.toString() }

  for (query in queries) {
    var noError = true
    var i = 1

    while (noError) {
      val queryUrl =
        "https://gitlab.com/api/v4/search?scope=projects&search=$query&per_page=100&page=${i++}"
      val text = sendGET(queryUrl)
      if (text.length < 200) noError = false

      val regex = Regex("https://gitlab.com/[^/]+?/[^/]+?\\.git")
      val matches = regex.findAll(text).map { it.value.dropLast(4).substringAfter("https://gitlab.com/") }.toList()
      if (matches.isEmpty()) noError = false
      val uniqueMatches = matches.filter { url ->
        val name = url.substringAfterLast('/')
        val owner = url.substringBeforeLast('/')
        exclude.add(name) &&
          exclude.add(owner) &&
          !shouldBeExcludedFromGitlab(url) &&
          !isRepoOnGitHub(name)
      }

      uniqueMatches.forEach { url ->
        val dedupedRepo = "https://gitlab.com/$url"
        File(GL_REPOS_FILE).appendText(dedupedRepo)
        println(dedupedRepo)
      }

      TimeUnit.SECONDS.sleep(13)
    }
  }
}

fun shouldBeExcludedFromGitlab(
  repo: String, strsToExclude: Set<String> = setOf(
    "The repository for this project is empty",
    "Exported from",
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