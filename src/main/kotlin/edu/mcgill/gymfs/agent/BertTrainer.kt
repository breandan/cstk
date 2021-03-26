package edu.mcgill.gymfs.agent

import ai.djl.*
import ai.djl.ndarray.*
import ai.djl.ndarray.types.Shape
import ai.djl.nn.transformer.*
import ai.djl.training.*
import ai.djl.training.dataset.Batch
import ai.djl.training.initializer.TruncatedNormalInitializer
import ai.djl.training.listener.*
import ai.djl.training.listener.EvaluatorTrainingListener.TRAIN_ALL
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.*
import ai.djl.translate.Batchifier
import edu.mcgill.gymfs.disk.*
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import kotlin.io.path.*
import kotlin.random.Random

@ExperimentalPathApi
@ExperimentalStdlibApi
fun main() {
  println("training bert on code...")
  TrainBertOnCode.runExample()
}

// https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainBertOnCode.java
/** Simple example that performs Bert pretraining on the source files in this repo.  */
@ExperimentalPathApi
@ExperimentalStdlibApi
object TrainBertOnCode {
  fun runExample(): TrainingResult {
    val files = ROOT_DIR.allFilesRecursively().map { ParsedFile(it) }
    val countedTokens = countTokens(files)
    val dictionary = buildDictionary(countedTokens)
    createBertPretrainingModel(dictionary).use { model ->
      createBertPretrainingTrainer(model).use { trainer ->
        // Initialize training
        val inputShape = Shape(MAX_SEQUENCE_LENGTH.toLong(), 512)
        trainer.initialize(inputShape, inputShape, inputShape, inputShape)
        trainer.notifyListeners { it.onTrainingBegin(trainer) }
        for (epoch in 0 until EPOCHS) {
          val maskedInstances = createEpochData(rand, dictionary, files)
          for (idx in BATCH_SIZE until maskedInstances.size) {
            trainer.manager.newSubManager().use { ndManager ->
              val batchData = maskedInstances.subList(idx - BATCH_SIZE, idx)
              val batch = createBatch(ndManager, batchData, idx, maskedInstances.size)
              // the following uses the GPUs alternating
              // EasyTrain.trainBatch(trainer, batch);
              // this actually uses both GPUs at once
              EasyTrain.trainBatch(trainer, batch)
            }
          }
          trainer.notifyListeners { it.onEpoch(trainer) }
        }

        trainer.notifyListeners { it.onEpoch(trainer) }
        return trainer.trainingResult
      }
    }
  }

  private fun createBertPretrainingModel(dictionary: Dictionary): Model =
    Model.newInstance("Bert Pretraining").apply {
      block = BertPretrainingBlock(
        BERT_BUILDER.setTokenDictionarySize(dictionary.tokens.size)
      )
      block.setInitializer(TruncatedNormalInitializer(0.02f))
    }

  private fun createBertPretrainingTrainer(model: Model): Trainer =
    Adam.builder()
      .optLearningRateTracker(
        WarmUpTracker.builder()
          .optWarmUpBeginValue(0f)
          .optWarmUpSteps(1000)
          .optWarmUpMode(WarmUpTracker.Mode.LINEAR)
          .setMainTracker(
            PolynomialDecayTracker.builder()
              .setBaseValue(5e-5f)
              .setEndLearningRate(5e-5f / 100)
              .setDecaySteps(100000)
              .optPower(1f)
              .build()
          ).build()
      )
      .optEpsilon(1e-5f)
      .build()
      .let { optimizer: Adam ->
        model.newTrainer(
          DefaultTrainingConfig(BertPretrainingLoss())
            .optOptimizer(optimizer)
            .optDevices(Device.getDevices(MAX_GPUS))
            .addTrainingListeners(
              *TrainingListener.Defaults.logging(10),
              SaveModelTrainingListener("", "codebert", 200),
              object: DivergenceCheckTrainingListener() {
                var numEpochs = 0
                override fun onTrainingBatch(
                  trainer: Trainer,
                  batchData: TrainingListener.BatchData
                ) {
                  if (numEpochs++ % 20 == 0)
                    println(trainer.loss.getAccumulator(TRAIN_ALL))
//                  println(batchData.batch.data)
                }
              }
            )
        )
      }

  private fun createEpochData(
    rand: Random,
    dictionary: Dictionary,
    parsedFiles: List<ParsedFile>,
  ): List<MaskedInstance> =
    // turn data into sentence pairs containing consecutive lines
    parsedFiles.flatMap { it.allSentencePairs() }.shuffled(rand)
      // swap sentences with 50% probability for next sentence task
      .chunked(2).apply { map { it[0].maybeSwap(rand, it[1]) } }.flatten()
      // Create masked instances for training
      .take(MAX_BATCH).map { sentencePair: SentencePair ->
        MaskedInstance(
          rand, dictionary,
          sentencePair.apply { truncateToTotalLength(MAX_SEQUENCE_LENGTH - 3) },
        )
      }

  private fun createBatch(
    ndManager: NDManager,
    instances: List<MaskedInstance>,
    idx: Int, dataSize: Int
  ): Batch {
    val inputs = NDList(
      batchFromList(ndManager, instances, MaskedInstance::tokenIds),
      batchFromList(ndManager, instances, MaskedInstance::getTypeIds),
      batchFromList(ndManager, instances, MaskedInstance::inputMask),
      batchFromList(ndManager, instances, MaskedInstance::maskedPositions)
    )
    val labels = NDList(
      nextSentenceLabelsFromList(ndManager, instances),
      batchFromList(ndManager, instances, MaskedInstance::maskedIds),
      batchFromList(ndManager, instances, MaskedInstance::labelMask)
    )
    return Batch(
      ndManager,
      inputs,
      labels,
      instances.size,
      Batchifier.STACK,
      Batchifier.STACK,
      idx.toLong(),
      dataSize.toLong()
    )
  }

  private fun batchFromList(
    ndManager: NDManager,
    instances: List<MaskedInstance>,
    f: (MaskedInstance) -> IntArray
  ): NDArray = ndManager.create(instances.map(f).toTypedArray())

  private fun nextSentenceLabelsFromList(
    ndManager: NDManager, instances: List<MaskedInstance>
  ): NDArray = ndManager.create(instances.map { it.nextSentenceLabel }.toIntArray())

  private fun countTokens(files: List<ParsedFile>): Map<String, Long> =
    HashMap<String, Long>(50000).also { result ->
      files.forEach { file ->
        file.tokenizedLines.forEach { line ->
          line.forEach { token ->
            result[token] = result.getOrDefault(token, 0L) + 1
          }
        }
      }
    }

  private fun buildDictionary(counts: Map<String, Long>) =
    Dictionary(counts.entries.sortedByDescending { it.value }.map { it.key })

  private class ParsedFile(
    val sourceFile: Path,
    val normalizedLines: List<String> =
      sourceFile.readLines(UTF_8)
        .filter { line -> line.trim { it <= ' ' }.isNotEmpty() },
    val tokenizedLines: List<List<String>> =
      normalizedLines.map(TOKENIZER::tokenize)
  ) {
    fun allSentencePairs() =
      tokenizedLines.map { ArrayList(it) }.zipWithNext()
        .map { (a, b) -> SentencePair(a, b) }
  }
}