package edu.mcgill.gymfs.disk

import java.io.Serializable
import java.net.URI
import java.nio.file.Files
import kotlin.io.path.*

data class Location constructor(val file: URI, val line: Int): Serializable {
  @OptIn(ExperimentalPathApi::class)
  fun getContext(surroundingLines: Int) =
    Files.newBufferedReader(file.toPath()).use {
      it.lineSequence()
        .drop((line - surroundingLines).coerceAtLeast(0))
        .take(surroundingLines + 1).joinToString("\n") { it.trim() }
    }

  /*
   * Fetches the most salient keywords from the context
   */

  private fun topKeywordsFromContext(
    mostKeywordsToTake: Int = 5,
    score: (String) -> Double
  ): List<Pair<String, Double>> =
    getContext(3).split(Regex("[^\\w']+"))
      .asSequence()
      .filter(String::isNotEmpty)
      .distinct()
      .map { it to score(it) }
      .filter { (_, score) -> 1.0 != score } // Score of 1.0 is the current loc
      .sortedBy { it.second }
      .take(mostKeywordsToTake)
      .toList()

  /*
   * Expand keywords by least common first. Common keywords are
   * unlikely to contain any useful information.
   *
   * TODO: Reweight score by other metrics?
   */

  fun expand(grepper: TrieSearch): List<Pair<String, Location>> =
    topKeywordsFromContext { grepper.search(it).size.toDouble() }
      .also { println("Salient keywords: $it") }
      .map { (kw, _) ->
        grepper.search(kw)
          .filter { it != this }
          .take(5).map { kw to it }
      }.flatten()

  override fun toString() = "…${file.path.substringAfterLast('/')}:L${line + 1}"
}