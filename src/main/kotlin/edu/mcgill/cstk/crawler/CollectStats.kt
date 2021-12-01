package edu.mcgill.cstk.crawler

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.defaultTokenizer
import edu.mcgill.cstk.nlp.allMethods
import spoon.Launcher
import spoon.reflect.declaration.CtType
import spoon.reflect.reference.CtTypeReference
import spoon.support.compiler.FileSystemFolder
import java.io.File
import kotlin.io.path.toPath

fun main() {
  //collectLengthStats()
  collectTreeStats()
}

fun collectTreeStats() {
  DATA_DIR.allFilesRecursively("tgz", false).toList()
    .map {
      println(it)
      val f = File(it).unzip()
      val model = Launcher().apply {
        f.toURI().allFilesRecursively().forEach {
          runCatching { addInputResource(it.path) }
        }
        //addInputResource(FileSystemFolder(f))
        buildModel()
      }.model

      println(model.allTypes.joinToString(", ") { it: CtType<*> ->
        val name = it.simpleName
        val superclass = it.superclass?.simpleName
        val supertypes = it.superInterfaces.map { it.simpleName }
        val numMethods = it.methods.size
        val numMethodsInherited = it.countInheritedMethods()
        "$name($numMethods/$numMethodsInherited)" + if (superclass != null) "<:$superclass" +
          if(supertypes.isEmpty()) "" else supertypes.joinToString(",", "{", "}") else ""
      })
    }.take(10)
}

fun CtType<*>.countInheritedMethods(superType: CtTypeReference<*>? = superclass): Int =
  if(superType == null) 0
  else methods.size + (superType.typeDeclaration?.countInheritedMethods() ?: 0)

fun collectLengthStats() {
  println("total lines, total tokens, avg line len, len comments, len code")
  DATA_DIR.allFilesRecursively()
    .allMethods()
    .forEach { (method, uri) ->
      try {
        val string = method.toString()
        val lines = string.lines()
        println(
          "" +
            lines.size + ", " +
            defaultTokenizer.tokenize(string).size +
            lines.size + ", " +
            lines.map { defaultTokenizer.tokenize(it).size }.average()
              .toInt() + ", " +
            defaultTokenizer.tokenize(method.docComment ?: "").size + ", " +
            defaultTokenizer.tokenize(method.body?.toString() ?: "").size
        )
      } catch (exception: Exception) {
      }
    }
}