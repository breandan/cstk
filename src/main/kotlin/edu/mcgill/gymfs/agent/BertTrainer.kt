package edu.mcgill.gymfs.agent

import ai.djl.*
import ai.djl.ndarray.types.Shape
import ai.djl.nn.transformer.*
import ai.djl.training.*
import ai.djl.training.listener.*
import ai.djl.training.optimizer.*
import ai.djl.training.tracker.*
import edu.mcgill.gymfs.disk.*


fun main() {
  val dataset = BertCodeDataset(BATCH_SIZE, 1000L).apply { prepare() }
  createBertPretrainingModel(dataset.getDictionarySize()).use { model ->
    val config = createTrainingConfig()
    model.newTrainer(config).use { trainer ->
      // Initialize training
      val inputShape = Shape(MAX_SEQUENCE_LENGTH.toLong(), 512)
      trainer.initialize(inputShape, inputShape, inputShape, inputShape)
      EasyTrain.fit(trainer, EPOCHS, dataset, null)
      trainer.trainingResult
    }
  }
}

private fun createBertPretrainingModel(dictionarySize: Int): Model {
  val block = BertPretrainingBlock(
    BertBlock.builder().micro().setTokenDictionarySize(dictionarySize)
  )
//  block.setInitializer(
//    TruncatedNormalInitializer(0.02f),
//
//  )
  val model = Model.newInstance("Bert Pretraining")
  model.block = block
  return model
}

private fun createTrainingConfig(): TrainingConfig {
  val learningRateTracker: Tracker = WarmUpTracker.builder()
    .optWarmUpBeginValue(0f)
    .optWarmUpSteps(1000)
    .optWarmUpMode(WarmUpTracker.Mode.LINEAR)
    .setMainTracker(
      PolynomialDecayTracker.builder()
        .setBaseValue(5e-5f)
        .setEndLearningRate(5e-5f / 1000)
        .setDecaySteps(100000)
        .optPower(1f)
        .build()
    )
    .build()
  val optimizer: Optimizer = Adam.builder()
    .optEpsilon(1e-5f)
    .optLearningRateTracker(learningRateTracker)
    .build()
  return DefaultTrainingConfig(BertPretrainingLoss())
    .optOptimizer(optimizer)
    .optDevices(Device.getDevices(MAX_GPUS))
    .addTrainingListeners(*TrainingListener.Defaults.logging(),
      LoggingTrainingListener(1),
      SaveModelTrainingListener("", "codebert", 20),
  )
}