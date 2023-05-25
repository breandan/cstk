package edu.mcgill.cstk.experiments.repair

import javax.tools.*
import org.jetbrains.kotlin.cli.common.messages.*
import org.jetbrains.kotlin.config.*
import java.io.*
import java.net.URI
import kotlin.system.measureTimeMillis
import javax.tools.JavaCompiler
import javax.tools.ToolProvider
import java.io.StringWriter
import javax.tools.JavaFileObject
import javax.tools.SimpleJavaFileObject
import javax.tools.JavaFileObject.Kind
import java.nio.charset.StandardCharsets

/*
./gradlew javaSemanticRepair
 */

fun main() {
  measureTimeMillis {
    for (i in 1..100) println(sourceCode.isCompileableJava())
  }.also { println("Total time: ${it/1000.0}s") } // About ~157ms / statement :(
}

val javac: JavaCompiler = ToolProvider.getSystemJavaCompiler()

val sourceCode = """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}
""".trimIndent()

// Maybe we can get this down to about ~10ms by avoiding the write to disk?
fun String.isCompileableJava(): Boolean {
  val sourceFile = object : SimpleJavaFileObject(URI.create("string:///Hello.java"), Kind.SOURCE) {
    override fun getCharContent(ignoreEncodingErrors: Boolean) = this@isCompileableJava
  }

  val task =
    javac.getTask(null, null, null, null, null, listOf(sourceFile))
  val success = task.call()
  return success
}