package edu.mcgill.cstk.experiments.probing

import com.google.common.util.concurrent.AtomicLongMap
import com.google.testing.compile.Compiler.javac
import com.google.testing.compile.JavaFileObjects
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import java.util.*
import java.util.concurrent.atomic.LongAdder
import javax.tools.*
import kotlin.streams.*

fun main() {
  val map = AtomicLongMap.create<Model>()
  val total = LongAdder()

  DATA_DIR
    .also { println("Evaluating $MODELS using compiler on $it...") }
    .allFilesRecursively().allMethods()
    .asStream().parallel()
    .filter { it.first.startsWith("public static") && it.first.lines().size < 10 }
    .map { it.first.lines().joinToString("  ") }
    .filter { compilesWithoutSyntaxErrors(it) }
    .forEach { code ->
      total.increment()
      MODELS.forEach { model ->
        val prompt = code.constructPrompt(model.mask)
        val completion = model.complete(prompt, maxTokens = 1)
        if (compilesWithoutSyntaxErrors(completion)) map.incrementAndGet(model)
      }

      val summary = map.asMap().entries
        .map { (k, v) -> k to "$v/$total" }.joinToString("\n")
      println("\nScores [model=(valid, total)]:\n$summary")
    }
}

private fun String.constructPrompt(mask: String) =
  replaceFirst(");", "$mask;")

fun compilesWithoutSyntaxErrors(
  code: String,
  file: JavaFileObject? = JavaFileObjects.forSourceString(
    "CompileTest",
    """class CompileTest { $code }"""
  ),
  syntaxErrors: List<Diagnostic<out JavaFileObject>> =
    javac().compile(file).errors()
      .filter { "cannot find symbol" !in it.getMessage(Locale.ENGLISH) }
): Boolean = syntaxErrors.isEmpty()