package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import java.io.File

fun fetchOrLoadSampleData(sampleSize: Int = 1000):
  Pair<List<String>, Array<DoubleArray>> =
  (File("sample$sampleSize.data")
    .let { if (it.exists()) it else null }
    ?.deserializeFrom()
    ?: TEST_DIR
      .allFilesRecursively()
      .allCodeFragments()
      .shuffled()
      .take(sampleSize)
      .map { it.second to vectorize(it.second) }
      .unzip().let { (l, v) -> l to v.toTypedArray() }
      .also { it.serializeTo(File("sample$sampleSize.data")) })