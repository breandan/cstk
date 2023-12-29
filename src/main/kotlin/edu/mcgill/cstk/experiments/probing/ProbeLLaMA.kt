package edu.mcgill.cstk.experiments.probing

import de.kherud.llama.*
import java.io.*
import java.nio.charset.StandardCharsets


/**
./gradlew probeLLaMA
 */
fun main() {
  LlamaModel.setLogger { level: LogLevel?, message: String? -> print(message) }

  val modelParams = ModelParameters().setNGpuLayers(43)
  val inferParams = InferenceParameters()
    .setTemperature(0.7f)
    .setPenalizeNl(true) //                .setNProbs(10)
    .setMirostat(InferenceParameters.MiroStat.V2)
    .setAntiPrompt("User:")

  val modelPath = "/models/mistral-7b-instruct-v0.2.Q6_K.gguf"
  val system =
    """
    This is a conversation between User and Llama, a friendly chatbot.
    Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.
    
    User: Hello Llama
    Llama: Hello.  How may I help you today?
    """.trimIndent()

  val reader = BufferedReader(InputStreamReader(System.`in`, StandardCharsets.UTF_8))
  LlamaModel(modelPath, modelParams).use { model ->
    print(system)
    var prompt: String? = system
    while (true) {
      prompt += "\nUser: "
      print("\nUser: ")
      val input = reader.readLine()
      prompt += input
      print("Llama: ")
      prompt += "\nLlama: "
      for (output in model.generate(prompt, inferParams)) {
        print(output)
        prompt += output
      }
    }
  }
}