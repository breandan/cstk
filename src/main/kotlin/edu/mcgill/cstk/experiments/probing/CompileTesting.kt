package edu.mcgill.cstk.experiments.probing

import com.google.common.util.concurrent.AtomicLongMap
import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.nlp.*
import java.io.*
import java.util.concurrent.atomic.LongAdder
import javax.tools.*
import kotlin.streams.asStream

//Scores [model=(valid, total)]:
//(microsoft/graphcodebert-base, 1267/1839)
//(dbernsohn/roberta-java, 433/1839)
//(huggingface/CodeBERTa-small-v1, 874/1839)
//(microsoft/codebert-base-mlm, 1377/1839)

fun main() {
  val map = AtomicLongMap.create<Model>()
  val total = LongAdder()

  DATA_DIR
    .also { println("Evaluating $MODELS using compiler on $it...") }
    .allFilesRecursively().allMethods().map { it.first }
    .filter { it.startsWith("public") && it.lines().size < 10 }
    .asStream().parallel()
    .filter { !containsSyntaxError(it) }
    .forEach { code ->
      MODELS.forEach { model ->
        val prompt = code.constructPrompt(model.mask)
        val completion = model.complete(prompt, maxTokens = 1)
        if (prompt.lines().all { it.length < 50 }) {
          printSideBySide(code, prompt, "code", "prompt")
          println("============================")
          printSideBySide(prompt, completion, "prompt", "completion")
        }

        if (containsSyntaxError(completion)) map.incrementAndGet(model)
      }

      total.increment()
      val summary = map.asMap().entries
        .map { (k, v) -> k to "$v/$total" }.joinToString("\n")
      println("\nScores [model=(valid, total)]:\n$summary")
    }
}

private fun String.constructPrompt(
  mask: String,
  maskChars: String = "(){}<>[]",
  escaped: String = Regex.escape(maskChars),
  split: List<String> = split(Regex("((?<=[$escaped])|(?=[$escaped]))")),
  toMask: Int = split.indices.filter { split[it] in maskChars }.random(),
  maskedSeq: String = split.toMutableList().apply { this[toMask] = mask }.joinToString("")
) = maskedSeq

fun containsSyntaxError(src: String): Boolean {
  val sourceFile = File("Test_${src.hashCode()}.java")
  sourceFile.writeText("class CompileTest { $src }")
  val compiler = ToolProvider.getSystemJavaCompiler()
  val output: OutputStream = object: OutputStream() {
    private val string = StringBuilder()

    @Throws(IOException::class)
    override fun write(b: Int) { string.append(b.toChar()) }

    override fun toString(): String = string.toString()
  }
  compiler.run(null, null, output, sourceFile.path)

  sourceFile.delete()
  return Regex("error: (.*)").findAll(output.toString())
    .any { it.destructured.component1() != "cannot find symbol" }
}

//https://github.com/huggingface/transformers/pull/10222
/** TODO: Experiment idea
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