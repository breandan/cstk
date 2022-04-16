package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import java.io.File

data class CodesAndVecs(val cfs: List<String>, val vecs: Array<DoubleArray>)

fun fetchOrLoadSampleData(sampleSize: Int = 1000): CodesAndVecs =
  File("sample$sampleSize.data")
    .let { if (it.exists()) it else null }
    ?.deserializeFrom()
    ?: TEST_DIR
      .allFilesRecursively()
      .allCodeFragments()
      .shuffled(DEFAULT_RAND)
      .take(sampleSize)
      .map { it.second to vectorize(it.second) }
      .unzip().let { (l, v) -> CodesAndVecs(l, v.toTypedArray()) }
      .also { it.serializeTo(File("sample$sampleSize.data")) }