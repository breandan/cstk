package edu.mcgill.cstk.experiments

import spoon.Launcher
import spoon.reflect.declaration.CtClass

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

  l.methods.forEach { println(it.docComment); println() }

  // https://github.com/SpoonLabs/spoon-examples/blob/master/src/main/java/fr/inria/gforge/spoon/transformation/OnTheFlyTransfoTest.java
}