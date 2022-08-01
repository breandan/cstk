package edu.mcgill.cstk.experiments.probing

import com.google.common.util.concurrent.AtomicLongMap
import com.google.testing.compile.Compiler.javac
import com.google.testing.compile.JavaFileObjects
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import java.util.Locale.ENGLISH
import java.util.concurrent.atomic.LongAdder
import javax.tools.*
import kotlin.streams.asStream

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
      .filter { "cannot find symbol" !in it.getMessage(ENGLISH) }
): Boolean = syntaxErrors.isEmpty()

/** TODO: Experiment idea
//https://github.com/huggingface/transformers/pull/10222
def fix_broken_code(code, lang_model):
    num_holes <- 1
    while [code] does not compile:
        sketchy_tokens <- calculate_highest_perplexity_tokens([num_holes], [code], [lang_model])
        code_template <- Replace or insert holes at [sketchy_tokens] in [code]
        llm_fixes <- Sample top-K insertions for each hole in [code_template] using [lang_model]
        admissible_set <- Solve [code_template] using SAT solver with [llm_fixes] as multi-choice
        if [admissible_set] is not empty:
            fixes <- rerank [admissible_set] with [lang_model] using maximum-likelihood criterion
            code <- Apply top fix in [fixes] to [code_template]
            return code
        num_holes <- num_holes + 1
 */

