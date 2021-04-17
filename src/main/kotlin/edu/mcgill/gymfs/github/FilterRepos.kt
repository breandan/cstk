package edu.mcgill.gymfs.github

import org.kohsuke.github.*
import java.io.File
import java.util.stream.Collectors

fun main() {
  GitHubBuilder()
    .withJwtToken(File(".ghtoken").readText())
    .build().searchRepositories()
    .language("java")
    .forks(">=100")
    .size(">=10000")
    .sort(GHRepositorySearchBuilder.Sort.STARS)
    .list()
    .take(2000)
    .parallelStream()
    .filter { repo ->
      repo.owner.type == "Organization" &&
      try {
        (repo.description + repo.readme.content)
          .none { c -> c.toInt() in 19968..40869 }
      } catch (e: Exception) { false }
    }
    .collect(Collectors.toList())
    .sortedBy { -it.openIssueCount }
    .forEach {
      File("repositories.txt").appendText(it.httpTransportUrl + "\n")
      println(
        it.openIssueCount.toString().padEnd(6) +
          it.stargazersCount.toString().padEnd(9) +
          it.httpTransportUrl
      )
    }
}