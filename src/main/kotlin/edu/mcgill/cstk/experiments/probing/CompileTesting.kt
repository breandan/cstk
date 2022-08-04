package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.parsing.*
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
    .filter { !it.containsSyntaxError() }
//    .flatMap { src -> src.constructPrompts().map { src to it }.distinct().take(3).asStream() }
    .forEach { code ->
      MODELS.forEach { model ->
        val prompt = code.constructPrompt(model.mask)
        val completion = model.complete(prompt, maxTokens = 1)
        val annotatedCompletion =
          if (completion.containsSyntaxError()) "$completion// Syntax error!"
          else { map.incrementAndGet(model); completion }
        if (prompt.lines().all { it.length < 50 }) {
          print(prettyDiffs(
            listOf(code, prompt, annotatedCompletion),
            listOf("code", "prompt", "completion")
          ))
        }
      }

      total.increment()
      val summary = map.asMap().entries
        .map { (k, v) -> k to "$v/$total" }.joinToString("\n")
      println("\nScores [model=(valid, total)]:\n$summary")
    }
}

val javaCFG ="""
    START -> D T P
    P -> A | ( P ) | ( P ) | { P } | [ P ] | < P > | P P
    D -> public | static | void | D D
    A -> ε | w | . | ; | , | = | A A
    W -> w
    T -> W | W < T >
  """.parseCFG()
fun String.quickSyntaxCheck() =
  javaCFG.isValid(
    splitByNonWordChars().filter { it.isNotBlank() }
      .joinToString(" ") { if (it in "(){}<>[]") it else "w" }
      // TODO: maybe we can apply this transformation automatically using the parser
      .replace(Regex("w( w)*"), "w")
      .also { println("Simplified code: $it") }
  )

fun String.constructPrompts() =
  generateSequence { constructPrompt(this) }

private fun String.constructPrompt(
  mask: String = "[[MASK]]",
  maskChars: String = "(){}<>[]", // "(){}<>[].,="
  escaped: String = Regex.escape(maskChars),
  split: List<String> = split(Regex("((?<=[$escaped])|(?=[$escaped]))")),
  toMask: Int = split.indices.filter { split[it] in maskChars }.random(),
  maskedSeq: String = split.toMutableList().apply { this[toMask] = mask }.joinToString("")
): String = if(!maskedSeq.quickSyntaxCheck()) maskedSeq else constructPrompt()

fun String.containsSyntaxError(): Boolean {
  val file = File.createTempFile("Test_${hashCode()}", ".java")
  file.writeText("class CompileTest { $this }")
  val errors = StringBuilder()
  object: OutputStream() { override fun write(b: Int) { errors.append(b.toChar()) } }
    .let { ToolProvider.getSystemJavaCompiler().run(null, null, it, file.path) }
  file.delete()

  return Regex("error: (.*)").findAll(errors.toString())
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