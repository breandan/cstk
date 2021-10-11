package edu.mcgill.cstk.experiments

import org.openrewrite.Recipe
import org.openrewrite.java.*

fun main() {
  val cus = JavaParser.fromJavaVersion()
//    .relaxedClassTypeMatching(true)
//    .logCompilationWarningsAndErrors(true)
    .build()
    .parse(
      """
      class Bar {}
      class B { static void foo(Bar a, Bar b, Bar c) {} }
      class Test {{  Bar a = Bar(); Bar b = Bar(); Bar c = Bar(); B.foo(a, b, c); }}
      """
    )

  val recipe: Recipe = DeleteMethodArgument("B foo(Bar, Bar, Bar)", 1)

  recipe.run(cus).forEach { println(it.after!!.print()) }
}