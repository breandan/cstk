package edu.mcgill.cstk.experiments.repair

import org.apache.commons.io.output.NullPrintStream
import org.intellij.lang.annotations.Language
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
    standaloneCompileableKotlinStatements.lines().parallelStream()
      .forEach { println(if (it.isCompilableKotlin()) "✅ $it" else "❌ $it") }
  }.also { println("Total time: ${it/1000.0}s") } // About ~173ms / statement :(

  evaluateSyntheticRepairBenchmarkOn(standaloneCompileableKotlinStatements) {
    print("Generated $size syntactic repairs")
    val time = System.currentTimeMillis()
    parallelStream().filter { it.resToStr().isCompilableKotlin() }.toList()
      .also {
        println(", and ${it.size} of them could be compiled." +
        " (${System.currentTimeMillis() - time}ms)")
      }
  }
}

val kotlinc = K2JVMCompiler()

fun createCompileArgs(absolutePath: String)=
  K2JVMCompilerArguments().apply {
    // Instead of hammering the disk, maybe create a virtual file system somehow?
    freeArgs = listOf(absolutePath)
    classpath = System.getProperty("java.class.path")
      .split(System.getProperty("path.separator"))
      .filter { File(it).exists() && File(it).canRead() }.joinToString(":")
    noStdlib = true
    noReflect = true
    reportPerf = false
    suppressWarnings = true
    destination = "/tmp/"
  }

val msgCollector = PrintingMessageCollector(NullPrintStream.INSTANCE, WITHOUT_PATHS, true)

fun String.isCompilableKotlin(): Boolean =
  File.createTempFile("tmp", ".kt").apply { delete(); writeText(this@isCompilableKotlin)  }
    .run {
      kotlinc.exec(msgCollector, Services.EMPTY, createCompileArgs(absolutePath))
        .apply { File("/tmp/" + nameWithoutExtension + "Kt.class").delete() }
    }.code == 0

@Language("kotlin")
val standaloneCompileableKotlinStatements =
  """
  fun greet(name: String) = println("Hello, " + name)
  val cube = { num: Int -> num * num * num }
  typealias StringList = List<String>; val names: StringList = listOf("John", "Jane", "Joe")
  val squareRoot = Math.sqrt(16.0)
  val randomString = (1..10).map { it.toChar().toString() }.joinToString("")
  val trimmedText = with(" Hello Kotlin ") { trim() }
  fun Double.half() = this / 2
  val list1 = mutableListOf("Kotlin", "Java").also { it.add("Scala") }
  val numbers = listOf(1, 2, 3).mapIndexed { idx, num -> idx to num }
  val names = listOf("Alice", "Bob", "Charlie").onEach(::println)
  val sentence = StringBuilder().apply { append("Hello, "); append("world!") }.toString()
  val primeNumbers = listOf(2, 3, 5, 7, 11, 13).takeLastWhile { it > 7 }
  val doubled = (1..10).fold(1) { acc, i -> acc * 2 }
  val stringLength = with("Hello") { length }
  val emptyString = "".ifEmpty { "Default String" }
  val length = "Hello, World!".count()
  val randomBool = listOf(true, false).shuffled().first()
  val chars = mutableListOf<Char>().apply { addAll('a'..'z') }
  val unique = setOf(1, 1, 2, 2, 3, 3)
  val map = mutableMapOf("one" to 1).apply { this["two"] = 2 }
  fun List<Int>.sum(): Int = this.reduce { acc, i -> acc + i }
  val reversed = StringBuilder("Kotlin").reverse().toString()
  val list = List(5) { it + 1 }
  val formatted = "%.2f".format(1.239)
  val message = StringBuilder().also { sb -> sb.append("Hello "); sb.append("world!") }.toString()
  val intSet = setOf(1, 2, 3).filter { it % 2 == 0 }.toSet()
  val result = runCatching { "100".toInt() }.getOrElse { 0 }
  val person = object { val name = "John"; val age = 30 }
  val triple = Triple("Kotlin", "is", "awesome")
  val score = mapOf("Alice" to 90, "Bob" to 85)
  val isPositive: (Int) -> Boolean = { it > 0 }
""".trimIndent()