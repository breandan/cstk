package edu.mcgill.gymfs.disk

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.common.io.ClassPathResource

fun main() {
  val path = "/home/breandan/Downloads/openai-gpt-tf_model.h5"
  val model = KerasModelImport.importKerasSequentialModelAndWeights(path)
  println(model.input.shape())
}