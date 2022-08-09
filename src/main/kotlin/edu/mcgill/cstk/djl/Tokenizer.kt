package edu.mcgill.cstk.utils

import ai.djl.huggingface.tokenizers.*


fun main() {
  val inputs = arrayOf("Hello, y'all!", "How are you 😁 ?")

  val expected = arrayOf(
    "[CLS]", "Hello", ",", "y", "'", "all",
    "!", "How", "are", "you", "[UNK]", "?",
    "[SEP]"
  )
  expected.joinToString(",").let { println(it) }

  // RuntimeException: Model "microsoft/codebert-base-mlm" on the Hub doesn't have a tokenizer??
  HuggingFaceTokenizer.newInstance("microsoft/codebert-base-mlm", mapOf("addSpecialTokens" to "false"))
    .use { tokenizer ->
      val encoding: Encoding = tokenizer.encode(inputs.asList())
      encoding.tokens.joinToString(",").let { println(it) }
    }
}