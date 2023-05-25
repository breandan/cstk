package edu.mcgill.cstk.experiments.repair

import javax.tools.*
import org.jetbrains.kotlin.cli.common.arguments.K2JVMCompilerArguments
import org.jetbrains.kotlin.cli.common.messages.*
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.config.*
import java.io.*
import kotlin.system.measureTimeMillis

val javaCompiler: JavaCompiler = ToolProvider.getSystemJavaCompiler()

/*
./gradlew kotlinFunctionRepair
 */

fun main() {
  // Write simple file:
  for (i in 0..100) {
    // Don't actually create the file on disk, but a virtual file
    measureTimeMillis {
      println("typealias KWIndex = List<String<String>>".isCompilableKotlin())
    }.also { println("Millis: $it") }
  }
}

fun String.isCompilableKotlin(): Boolean = K2JVMCompiler().run {
  val args = K2JVMCompilerArguments().apply {
    val file = createTempFile(suffix = ".kt").apply { writeText(this@isCompilableKotlin) }
    freeArgs = listOf(file.absolutePath)
    classpath = System.getProperty("java.class.path")
      .split(System.getProperty("path.separator"))
      .filter { File(it).exists() && File(it).canRead() }.joinToString(":")
    noStdlib = true
    noReflect = true
    reportPerf = true
  }
//  output.deleteOnExit()
  execImpl(
    PrintingMessageCollector(
      System.out,
      MessageRenderer.WITHOUT_PATHS, true),
    Services.EMPTY,
    args)
}.code == 0