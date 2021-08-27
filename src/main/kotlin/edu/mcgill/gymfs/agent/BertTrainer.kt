package edu.mcgill.gymfs.agent

import ai.djl.*
import ai.djl.ndarray.types.Shape
import ai.djl.nn.transformer.*
import ai.djl.training.*
import ai.djl.training.listener.*
import ai.djl.training.listener.EvaluatorTrainingListener.TRAIN_ALL
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.*
import edu.mcgill.gymfs.disk.*


fun main() {
  val dataset = BertCodeDataset().apply { prepare() }
  createBertPretrainingModel(dataset.getDictionarySize()).use { model ->
    model.newTrainer(createTrainingConfig()).use { trainer ->
      // Initialize training
      val inputShape = Shape(MAX_SEQUENCE_LENGTH.toLong(), 512)
      trainer.initialize(inputShape, inputShape, inputShape, inputShape)
      EasyTrain.fit(trainer, EPOCHS, dataset, null)
      trainer.trainingResult
    }
  }
}

private fun createBertPretrainingModel(dictionarySize: Int) =
  Model.newInstance("Bert Pretraining").apply {
    block = BertPretrainingBlock(
      BertBlock.builder().micro().setTokenDictionarySize(dictionarySize)
    )
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

  val optimizer = Adam.builder()
    .optEpsilon(1e-5f)
    .optLearningRateTracker(learningRateTracker)
    .build()

  val lossListener = object: DivergenceCheckTrainingListener() {
    var batch = 0
    override fun onTrainingBatch(
      trainer: Trainer?,
      batchData: TrainingListener.BatchData?
    ) = trainer?.loss?.getAccumulator(TRAIN_ALL)
      .let { if (batch++ % 200 == 0) println(it) }
  }

  return DefaultTrainingConfig(BertPretrainingLoss())
    .optOptimizer(optimizer)
    .optDevices(Device.getDevices(MAX_GPUS))
    .addTrainingListeners(
      *TrainingListener.Defaults.logging(),
      lossListener,
      SaveModelTrainingListener("", "codebert", 20),
    )
}