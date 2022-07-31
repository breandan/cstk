package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.dyckCheck
import com.google.testing.compile.*
import com.google.testing.compile.Compiler.javac
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import java.util.*
import javax.tools.*

fun main() {
  DATA_DIR
    .also { println("Evaluating $MODELS using compiler on $it...") }
    .allFilesRecursively().allMethods()
    .filter { it.first.startsWith("public static") && it.first.lines().size < 10 }
    .map { it.first.lines().joinToString("  ") }
    .filter { compilesWithoutSyntaxErrors(it) }
    .runningFold(MODELS.associateWith { (0 to 0) }) { scores, code ->
      MODELS.associateWith { model ->
        val prompt = code.constructPrompt(model.mask)
        val completion = model.complete(prompt, maxTokens = 1)
        scores[model]!!.let { (n, d) ->
          if (compilesWithoutSyntaxErrors(completion)) (n + 1) to (d+1)
          else n to (d + 1)
        }
      }
    }
//    .filterIndexed { i, _ -> i % 10 == 0 }
    .forEach { println("\nScores [model=(valid, total)]:\n${it.entries.joinToString("\n")}") }
}

private fun String.constructPrompt(mask: String) =
  replaceFirst(");", "$mask;")

fun compilesWithoutSyntaxErrors(
  code: String,
  file: JavaFileObject? =
    JavaFileObjects.forSourceString(
      "CompileTest",
      """class CompileTest { $code }"""
    ),
  syntaxErrors: List<Diagnostic<out JavaFileObject>> =
    javac().compile(file).errors()
      .filter { "cannot find symbol" !in it.getMessage(Locale.ENGLISH) }
): Boolean = syntaxErrors.isEmpty()