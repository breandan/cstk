package edu.mcgill.gymfs.agent

/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
import ai.djl.*
import ai.djl.basicdataset.nlp.*
import ai.djl.basicdataset.utils.TextData
import ai.djl.basicmodelzoo.nlp.*
import ai.djl.metric.Metrics
import ai.djl.modality.nlp.EncoderDecoder
import ai.djl.modality.nlp.embedding.*
import ai.djl.modality.nlp.preprocess.*
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Block
import ai.djl.nn.recurrent.LSTM
import ai.djl.training.*
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.*
import ai.djl.training.loss.MaskedSoftmaxCrossEntropyLoss
import ai.djl.training.util.ProgressBar
import ai.djl.translate.*
import edu.mcgill.gymfs.BATCH_SIZE
import java.io.IOException
import java.util.*
import java.util.concurrent.*

fun main() {
  TrainSeq2Seq.runExample()
}

object TrainSeq2Seq {
  @Throws(IOException::class, TranslateException::class)
  fun runExample(): TrainingResult? {
    val executorService = Executors.newFixedThreadPool(8)
    Model.newInstance("seq2seqMTEn-Fr").use { model ->
      // get training and validation dataset
      val trainingSet = getDataset(
        Dataset.Usage.TRAIN,
        executorService,
        null,
        null
      )
      // Fetch TextEmbedding from dataset
      val sourceEmbedding =
        trainingSet.getTextEmbedding(true) as TrainableTextEmbedding
      val targetEmbedding =
        trainingSet.getTextEmbedding(false) as TrainableTextEmbedding

      // Build the model with the TextEmbedding so that embeddings can be trained
      val block = getSeq2SeqModel(
        sourceEmbedding,
        targetEmbedding,
        trainingSet.getVocabulary(false).size()
      )
      model.block = block

      // setup training configuration
      val config =
        setupTrainingConfig()
      try {
        model.newTrainer(config).use { trainer ->
          trainer.metrics = Metrics()
          /*
  In Sequence-Sequence model for MT, the decoder input must be staggered by one wrt
  the label during training.
   */
          val encoderInputShape = Shape(BATCH_SIZE.toLong(), 10)
          val decoderInputShape = Shape(BATCH_SIZE.toLong(), 9)

          // initialize trainer with proper input shape
          trainer.initialize(encoderInputShape, decoderInputShape)

          // EncoderDecoder don't implement inference, set validateDataset to null
          EasyTrain.fit(trainer, 1000/*numEpoch*/, trainingSet, null)
          return trainer.trainingResult
        }
      } finally {
        executorService.shutdownNow()
      }
    }
  }

  private fun getSeq2SeqModel(
    sourceEmbedding: TrainableTextEmbedding,
    targetEmbedding: TrainableTextEmbedding,
    vocabSize: Long
  ): Block {
    val simpleTextEncoder = SimpleTextEncoder(
      sourceEmbedding,
      LSTM.Builder()
        .setStateSize(32)
        .setNumLayers(2)
        .optDropRate(0f)
        .optBatchFirst(true)
        .optReturnState(true)
        .build()
    )
    val simpleTextDecoder = SimpleTextDecoder(
      targetEmbedding,
      LSTM.Builder()
        .setStateSize(32)
        .setNumLayers(2)
        .optDropRate(0f)
        .optBatchFirst(true)
        .optReturnState(false)
        .build(),
      vocabSize
    )
    return EncoderDecoder(simpleTextEncoder, simpleTextDecoder)
  }

  fun setupTrainingConfig(): DefaultTrainingConfig {
    val listener = SaveModelTrainingListener(".")
    listener.setSaveModelCallback { trainer: Trainer ->
      val result = trainer.trainingResult
      val model = trainer.model
      val accuracy =
        result.getValidateEvaluation("Accuracy")
      model.setProperty(
        "Accuracy",
        String.format("%.5f", accuracy)
      )
      model.setProperty(
        "Loss",
        String.format("%.5f", result.validateLoss)
      )
    }
    return DefaultTrainingConfig(MaskedSoftmaxCrossEntropyLoss())
      .addEvaluator(Accuracy("Accuracy", 0, 2))
      .optDevices(Device.getDevices(1))
      .addTrainingListeners(*TrainingListener.Defaults.logging("."))
      .addTrainingListeners(listener)
  }

  @Throws(IOException::class, TranslateException::class)
  fun getDataset(
    usage: Dataset.Usage,
    executorService: ExecutorService?,
    sourceEmbedding: TextEmbedding?,
    targetEmbedding: TextEmbedding?
  ): TextDataset {
    val datasetBuilder = TatoebaEnglishFrenchDataset.builder()
      .setSampling(BATCH_SIZE, true, false)
      .optDataBatchifier(
        PaddingStackBatchifier.builder()
          .optIncludeValidLengths(true)
          .addPad(0, 0,
            { m: NDManager -> m.zeros(Shape(1)) }, 10
          )
          .build()
      )
      .optLabelBatchifier(
        PaddingStackBatchifier.builder()
          .optIncludeValidLengths(true)
          .addPad(0, 0,
            { m: NDManager -> m.ones(Shape(1)) }, 10
          )
          .build()
      )
      .optUsage(usage)
      .optExecutor(executorService, 8)
      .optLimit(100)
    val sourceConfig = TextData.Configuration()
      .setTextProcessors(
        listOf(
          SimpleTokenizer(),
          LowerCaseConvertor(Locale.ENGLISH),
          PunctuationSeparator(),
          TextTruncator(10)
        )
      )
    val targetConfig = TextData.Configuration()
      .setTextProcessors(
        listOf(
          SimpleTokenizer(),
          LowerCaseConvertor(Locale.FRENCH),
          PunctuationSeparator(),
          TextTruncator(8),
          TextTerminator()
        )
      )
    if (sourceEmbedding != null) {
      sourceConfig.setTextEmbedding(sourceEmbedding)
    } else {
      sourceConfig.setEmbeddingSize(32)
    }
    if (targetEmbedding != null) {
      targetConfig.setTextEmbedding(targetEmbedding)
    } else {
      targetConfig.setEmbeddingSize(32)
    }
    val dataset = datasetBuilder
      .setSourceConfiguration(sourceConfig)
      .setTargetConfiguration(targetConfig)
      .build()
    dataset.prepare(ProgressBar())
    return dataset
  }
}
