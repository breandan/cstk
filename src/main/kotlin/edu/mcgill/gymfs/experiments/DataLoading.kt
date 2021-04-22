package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*
import java.io.File
import kotlin.io.path.*

@OptIn(ExperimentalPathApi::class)
fun fetchOrLoadSampleData(sampleSize: Int = 1000) =
  (File("sample$sampleSize.data")
    .let { if (it.exists()) it else null }
    ?.deserialize() as? Pair<List<String>, Array<DoubleArray>>
    ?: ROOT_DIR
      .allFilesRecursively()
      .filter { it.extension == FILE_EXT }
      .allCodeFragments()
      .shuffled()
      .take(sampleSize)
      .map { it.second to vectorize(it.second) }
      .unzip().let { (l, v) -> l to v.toTypedArray() }
      .also { it.serialize(File("sample$sampleSize.data")) })