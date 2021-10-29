package edu.mcgill.cstk.experiments

import astminer.common.model.LabeledResult
import astminer.parse.gumtree.GumTreeNode
import astminer.parse.gumtree.java.GumTreeJavaParser
import astminer.parse.gumtree.java.GumTreeJavaFunctionSplitter
import astminer.storage.path.*
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.allLines

// Retrieve paths from Java files, using a GumTree parser.

fun main() {
  val config = PathBasedStorageConfig(5, 5, null, null, null)
  val code2vecStorage: PathBasedStorage = Code2VecPathStorage(DATA_DIR.path, config)
  DATA_DIR
    .allFilesRecursively()
    .map { it.toString() to it.allLines().joinToString("\n") }
    .forEach { (path, contents) ->
      val fileTree =
        GumTreeJavaParser().parseInputStream(contents.byteInputStream())
      val labeledResult: LabeledResult<GumTreeNode> = LabeledResult(fileTree,path,path)

      val methodNodes = GumTreeJavaFunctionSplitter().splitIntoFunctions(labeledResult.root, path)
      methodNodes.forEach { println(it.root.wrappedNode.toPrettyString(it.root.context)) }

      code2vecStorage.store(labeledResult)
    }

  code2vecStorage.close()
}