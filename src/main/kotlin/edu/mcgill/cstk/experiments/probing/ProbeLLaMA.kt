package edu.mcgill.cstk.experiments.probing

import edu.mcgill.cstk.llama3.Llama3

/*
./gradlew probeLLaMA
 */
fun main() {
  Llama3.main(arrayOf("-i", "--model", "models/Llama-3.2-3B-Instruct-Q4_0.gguf"))
}