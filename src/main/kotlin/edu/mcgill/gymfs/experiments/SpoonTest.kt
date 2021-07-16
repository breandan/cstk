package edu.mcgill.gymfs.experiments

import spoon.Launcher
import spoon.reflect.code.*
import spoon.reflect.declaration.*
import spoon.reflect.factory.Factory


fun main() {
  val launcher = Launcher()
  val factory = launcher.factory
  val t = factory.createCodeSnippetStatement("int x = 1")
  val q = factory.createCodeSnippetStatement("x=x+1")
//  val m = t.compile<CtStatement>().replace()
//  println(m)

//  val aClass = factory.createClass("my.org.MyClass")
//  aClass.setSimpleName<CtNamedElement>("myNewName")
//  val myMethod = factory.createMethod()
//  aClass.addMethod(myMethod)
}