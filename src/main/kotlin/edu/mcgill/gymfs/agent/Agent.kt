package edu.mcgill.gymfs.agent

import ai.djl.Model
import ai.djl.ndarray.types.Shape
import ai.djl.nn.*
import ai.djl.nn.convolutional.Conv2d
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.loss.Loss

// https://github.com/kingyuluk/RL-FlappyBird/blob/master/src/main/java/com/kingyu/rlbird/ai/TrainBird.java
fun main() {
  val model = DefaultTrainingConfig(Loss.l2Loss())
  println(model.devices[0])
  createOrLoadModel()
}

fun createOrLoadModel() =
  Model.newInstance("QNetwork").apply {
    block = SequentialBlock().add(
      Conv2d.builder()
        .setKernelShape(Shape(8, 8))
        .optStride(Shape(4, 4))
        .optPadding(Shape(3, 3))
        .setFilters(4).build()
    ).add {
      println(it.shapes)
      Activation.relu(it)
    }
  }