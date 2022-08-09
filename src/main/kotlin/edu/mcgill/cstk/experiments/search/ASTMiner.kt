package edu.mcgill.cstk.experiments.search

import astminer.common.model.LabeledResult
import astminer.parse.gumtree.GumTreeNode
import astminer.parse.gumtree.java.srcML.*
import astminer.storage.path.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.utils.allLines

// Retrieve paths from Java files, using a GumTree parser.

fun main() {
  val config = PathBasedStorageConfig(5, 5, null, null, null)
  val code2vecStorage: PathBasedStorage = Code2VecPathStorage(DATA_DIR.path, config)
  DATA_DIR
    .allFilesRecursively()
    .map { it.toString() to it.allLines().joinToString("\n") }
    .forEach { (path, contents) ->
      val fileTree =
        GumTreeJavaSrcmlParser().parseInputStream(contents.byteInputStream())
      val labeledResult: LabeledResult<GumTreeNode> = LabeledResult(fileTree,path,path)

      val methodNodes = GumTreeJavaSrcmlFunctionSplitter().splitIntoFunctions(labeledResult.root, path)
      methodNodes.forEach {it.root.prettyPrint()}

      code2vecStorage.store(labeledResult)
    }

  code2vecStorage.close()
}