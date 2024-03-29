package edu.mcgill.cstk.disk.indices

import edu.mcgill.cstk.disk.Concordance
import edu.mcgill.cstk.utils.vectorize

fun VecIndex.knn(query: String, k: Int = 10) = knn(vectorize(query), k)

fun VecIndex.knn(v: DoubleArray, i: Int, exact: Boolean = false) =
  if (exact) exactKNNSearch(v, i + 10)
  else findNearest(v, i + 10)
    .filter { !it.item().embedding.contentEquals(v) }
    .distinctBy { it.item().toString() }.take(i)

fun KWIndex.search(query: String): List<Concordance> =
  getValuesForKeysContaining(query).flatten()

// Returns a map of files to concordances with all files containing all keywords
fun KWIndex.search(vararg keywords: String) =
  keywords.map { search(it) }.reduce { acc, results ->
    val common = results.map { it.uri }.intersect(results.map { it.uri }.toSet())
    (acc + results).filter { it.uri in common }
  }.groupBy { it.uri }