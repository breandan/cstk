package edu.mcgill.cstk.experiments

import edu.mcgill.cstk.disk.*
import spoon.Launcher
import spoon.support.compiler.FileSystemFolder
import java.io.File

fun main() {
  val l =
    Launcher.parseClass("""
      class T {
      /**
       * Testing m
       */
      void m() { q = 1; System.out.println("yeah"); }
      
      /**
       * Testing q
       */
      void q() { q = 1; System.out.println("yeah"); }
      }
    """.trimIndent())

  //println(l.superclass.simpleName)
  //  Launcher().apply {
  //    addInputResource(FileSystemFolder(File()))
  //  }

  val f = File("$DATA_DIR/Netflix_zuul.tgz").unzip()
  val model = Launcher().apply {
    addInputResource(FileSystemFolder(f))
    buildModel()
  }.model

  println(model.allTypes.joinToString(", ") { it.simpleName + "->" + (it.superclass?.simpleName ?: "*") })

//  l.methods.forEach { println(it.docComment); println() }

  // https://github.com/SpoonLabs/spoon-examples/blob/master/src/main/java/fr/inria/gforge/spoon/transformation/OnTheFlyTransfoTest.java
}