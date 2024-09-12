package edu.mcgill.cstk.experiments.probing

import ai.hypergraph.kaliningraph.repair.extractPatch
import de.kherud.llama.*
import de.kherud.llama.args.Sampler.TYPICAL_P
import edu.mcgill.cstk.experiments.repair.invalidPythonStatements
import edu.mcgill.cstk.utils.*
import java.io.*
import java.nio.charset.StandardCharsets


/**
./gradlew probeLLaMA
 */
fun main() {
//  LlamaModel.setLogger { level: LogLevel?, message: String? -> print(message) }

  val modelPath = File("").absolutePath + "/models/ggml-model-Q6_K.gguf"
  val modelParams = ModelParameters().setNGpuLayers(43)
    .setModelFilePath(modelPath)

  LlamaModel(modelParams).use { model ->
    invalidPythonStatements.lines().forEach { invalidCodeSnippet ->
      val prompt = """
        The following line of Python code contains a syntax error:
        
        ```
        $invalidCodeSnippet 
        ``` 
         
        Below is the most likely syntactically valid repair:
        
        ```
        """.trimIndent()

      BufferedReader(InputStreamReader(System.`in`, StandardCharsets.UTF_8))
      val grammar = """root  ::= (expr "=" term "\\n")+
		expr  ::= term ([-+*/] term)*
		term  ::= [0-9]"""

      val sb = StringBuilder()
      val inferParams = InferenceParameters(prompt)
        .setTemperature(0.7f)
        .setSamplers(TYPICAL_P)
        .setPenalizeNl(true) //                .setNProbs(10)
//        .setGrammar(grammar)
//    .setMirostat(InferenceParameters.MiroStat.V2)
//    .setAntiPrompt("User:")

      for (output in model.generate(inferParams)) {
        sb.append(output)
      }

      val line = sb.toString().substringBefore("```").trim().lines().last()
      println(invalidCodeSnippet)
      println(prettyDiffNoFrills(invalidCodeSnippet, line))
      println("Was valid: " + sb.toString().isValidPython())
    }
  }
}