package edu.mcgill.cstk.disk

import ai.hypergraph.kaliningraph.types.*
import edu.mcgill.cstk.indices.*
import edu.mcgill.cstk.nlp.allLines
import java.io.Serializable
import java.net.URI

// https://en.wikipedia.org/wiki/Concordance_(publishing)
data class Concordance constructor(val uri: URI, val line: Int): Serializable {
  fun getContext(surroundingLines: Int) =
    uri.allLines().drop((line - surroundingLines).coerceAtLeast(0))
      .take(surroundingLines + 1).joinToString("\n") { it.trim() }

  /*
   * Fetches the most salient keywords from the context
   */

  private fun topKeywordsFromContext(
    mostKeywordsToTake: Int = 5,
    score: (String) -> Double
  ): List<Π2<String, Double>> =
    getContext(3).split(Regex("[^\\w']+"))
      .asSequence()
      .filter(String::isNotEmpty)
      .distinct()
      .map { it pp score(it) }
      .filter { (_, score) -> 1.0 != score } // Score of 1.0 is the current loc
      .sortedBy { it.second }
      .take(mostKeywordsToTake)
      .toList()

  fun matchesRegex(regex: Regex) = getContext(0).matches(regex)

  /*
   * Expand keywords by least common first. Common keywords are
   * unlikely to contain any useful information.
   *
   * TODO: Reweight score by other metrics?
   */

  fun expand(grepper: KWIndex): List<Π2<String, Concordance>> =
    topKeywordsFromContext { grepper.search(it).size.toDouble() }
      .also { println("Salient keywords: $it") }
      .map { (kw, _) ->
        grepper.search(kw)
          .filter { it != this }
          .take(5).map { kw pp it }
      }.flatten()

  override fun toString() =
    uri.toString().substringBefore(".tgz").substringAfterLast('/') +
      "/…/" + uri.toString().substringAfterLast("/") + ":L${line + 1}"

  fun fileSummary() = toString().substringBeforeLast(':')
}