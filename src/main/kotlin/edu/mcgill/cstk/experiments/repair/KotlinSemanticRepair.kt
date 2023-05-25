package edu.mcgill.cstk.experiments.repair

import org.apache.commons.io.output.NullPrintStream
import org.jetbrains.kotlin.cli.common.arguments.K2JVMCompilerArguments
import org.jetbrains.kotlin.cli.common.messages.*
import org.jetbrains.kotlin.cli.common.messages.MessageRenderer.WITHOUT_PATHS
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.config.*
import java.io.*
import kotlin.system.measureTimeMillis

/*
./gradlew kotlinSemanticRepair
 */

fun main() {
  measureTimeMillis {
    for (i in 0..100) {
      // We need to collect a dataset of fully-compilable Kotlin/Java code snippets
      // This means injecting imports or only accepting snippets with builtin types
      println("typealias KWIndex = List<List<String>>".isCompilableKotlin())
      println("fun test() = println(\"hello\")".isCompilableKotlin())
    }
  }.also { println("Total time: ${it/1000.0}s") } // About ~173ms / statement :(
}

val kotlinc = K2JVMCompiler()

val compilerArgs =
  K2JVMCompilerArguments().apply {
    // Instead of hammering the disk, maybe create a virtual file system somehow?
    freeArgs = listOf(File.createTempFile("tmp", "kt").absolutePath)
    classpath = System.getProperty("java.class.path")
      .split(System.getProperty("path.separator"))
      .filter { File(it).exists() && File(it).canRead() }.joinToString(":")
    noStdlib = true
    noReflect = true
    reportPerf = false
    suppressWarnings = true
  }

val msgCollector = PrintingMessageCollector(NullPrintStream(), WITHOUT_PATHS, true)

fun String.isCompilableKotlin(): Boolean =
  File("temp.kt").apply { delete(); writeText(this@isCompilableKotlin) }
    .run { kotlinc.execImpl(msgCollector, Services.EMPTY, compilerArgs) }.code == 0