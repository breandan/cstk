package edu.mcgill.cstk.inference

import ai.djl.*
import ai.djl.engine.Engine
import ai.djl.ndarray.*
import ai.djl.repository.zoo.*
import ai.djl.training.util.ProgressBar
import ai.djl.translate.*
import java.util.*
import kotlin.system.measureTimeMillis

/*
 * An example of inference using an universal sentence
 * encoder model from TensorFlow Hub.
 *
 *
 * https://tfhub.dev/google/universal-sentence-encoder/4
 */

fun main() {
  val use = UniversalSentenceEncoder
  println("Millis " + measureTimeMillis {
    repeat(100) {
      val inputs: MutableList<String> = ArrayList()
      repeat(100) {
        inputs.add("The quick brown fox jumps over the lazy dog $it")
      }
      val embeddings = use.predict(inputs)!!
//    if (embeddings == null) println("This example only works for TensorFlow Engine")
//    else for (i in inputs.indices) println(embeddings[i].joinToString(","))
      println(embeddings.size)
    }
  })
}

object UniversalSentenceEncoder {
  fun predict(inputs: List<String>): Array<FloatArray>? {
    if ("TensorFlow" != Engine.getInstance().engineName) return null

    val modelUrl =
      "https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz"
    val criteria = Criteria.builder()
      .optApplication(Application.NLP.TEXT_EMBEDDING)
      .setTypes(Array<String>::class.java, Array<FloatArray>::class.java)
      .optModelUrls(modelUrl)
      .optTranslator(MyTranslator())
      .optProgress(ProgressBar())
      .build()
    ModelZoo.loadModel(criteria).use { model ->
      model.newPredictor().use { predictor ->
        return@predict predictor.predict(inputs.toTypedArray())
      }
    }
  }

  private class MyTranslator: Translator<Array<String>, Array<FloatArray>> {
    override fun processInput(
      ctx: TranslatorContext,
      inputs: Array<String>
    ): NDList = NDList(
      NDArrays.stack(NDList(inputs.map { ctx.ndManager.create(it) }))
    )

    override fun processOutput(
      ctx: TranslatorContext,
      list: NDList
    ): Array<FloatArray> {
      val result = NDList()
      val numOutputs = list.singletonOrThrow().shape[0]
      for (i in 0 until numOutputs) result.add(list.singletonOrThrow()[i])
      return result.toList().map { obj: NDArray -> obj.toFloatArray() }
        .toTypedArray()
    }

    override fun getBatchifier(): Batchifier? = null
  }
}
