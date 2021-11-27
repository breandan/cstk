package edu.mcgill.cstk.github

import edu.mcgill.cstk.disk.MINIGITHUB_REPOS_FILE
import org.kohsuke.github.*
import java.io.File
import java.util.stream.Collectors

fun main() =
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
      File(MINIGITHUB_REPOS_FILE).appendText(it.httpTransportUrl.dropLast(4) + "\n")
      println(
        it.openIssueCount.toString().padEnd(6) +
          it.stargazersCount.toString().padEnd(9) +
          it.size.toString().padEnd(9) +
          it.httpTransportUrl
      )
    }