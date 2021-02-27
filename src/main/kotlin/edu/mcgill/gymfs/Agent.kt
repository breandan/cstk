package edu.mcgill.gymfs

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
import ai.djl.nn.*
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.recurrent.LSTM
import ai.djl.training.*
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.*
import ai.djl.training.loss.*
import ai.djl.training.util.ProgressBar
import ai.djl.translate.*
import ai.djl.util.JsonUtils
import com.google.gson.reflect.TypeToken
import org.apache.commons.cli.*
import java.io.IOException
import java.util.*
import java.util.concurrent.*

// https://github.com/kingyuluk/RL-FlappyBird/blob/master/src/main/java/com/kingyu/rlbird/ai/TrainBird.java

fun main() {
  val model = DefaultTrainingConfig(Loss.l2Loss())
  println(model.devices[0])
  createOrLoadModel()
  seq2seq()
}

fun seq2seq() {
  TrainSeq2Seq.runExample(arrayOf("-e", "100", "-m", "10"))
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

// https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainBertOnCode.java
fun bertOnCode(): Model = TODO()


object TrainSeq2Seq {
  @Throws(IOException::class, TranslateException::class)
  fun runExample(args: Array<String?>?): TrainingResult? {
    val arguments: Arguments =
      Arguments().parseArgs(args)
        ?: return null
    val executorService = Executors.newFixedThreadPool(8)
    Model.newInstance("seq2seqMTEn-Fr").use { model ->
      // get training and validation dataset
      val trainingSet: TextDataset = getDataset(
        Dataset.Usage.TRAIN,
        arguments,
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
        setupTrainingConfig(arguments)
      try {
        model.newTrainer(config).use { trainer ->
          trainer.metrics = Metrics()
          /*
  In Sequence-Sequence model for MT, the decoder input must be staggered by one wrt
  the label during training.
   */
          val encoderInputShape: Shape =
            Shape(arguments.batchSize.toLong(), 10L)
          val decoderInputShape: Shape =
            Shape(arguments.batchSize.toLong(), 9L)

          // initialize trainer with proper input shape
          trainer.initialize(encoderInputShape, decoderInputShape)

          // EncoderDecoder don't implement inference, set validateDataset to null
          EasyTrain.fit(trainer, arguments.epoch, trainingSet, null)
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

  fun setupTrainingConfig(arguments: Arguments): DefaultTrainingConfig {
    val outputDir = arguments.outputDir
    val listener = SaveModelTrainingListener(outputDir)
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
      .optDevices(Device.getDevices(arguments.maxGpus))
      .addTrainingListeners(*TrainingListener.Defaults.logging(outputDir))
      .addTrainingListeners(listener)
  }

  @Throws(IOException::class, TranslateException::class)
  fun getDataset(
    usage: Dataset.Usage,
    arguments: Arguments,
    executorService: ExecutorService?,
    sourceEmbedding: TextEmbedding?,
    targetEmbedding: TextEmbedding?
  ): TextDataset {
    val limit =
      if (usage == Dataset.Usage.TRAIN) arguments.limit else arguments.limit / 10
    val datasetBuilder: TatoebaEnglishFrenchDataset.Builder =
      TatoebaEnglishFrenchDataset.builder()
        .setSampling(arguments.batchSize, true, false)
        .optDataBatchifier(
          PaddingStackBatchifier.builder()
            .optIncludeValidLengths(true)
            .addPad(
              0, 0,
              { m: NDManager -> m.zeros(Shape(1)) }, 10
            )
            .build()
        )
        .optLabelBatchifier(
          PaddingStackBatchifier.builder()
            .optIncludeValidLengths(true)
            .addPad(
              0, 0,
              { m: NDManager -> m.ones(Shape(1)) }, 10
            )
            .build()
        )
        .optUsage(usage)
        .optExecutor(executorService, 8)
        .optLimit(limit)
    val sourceConfig: TextData.Configuration = TextData.Configuration()
      .setTextProcessors(
        Arrays.asList(
          SimpleTokenizer(),
          LowerCaseConvertor(Locale.ENGLISH),
          PunctuationSeparator(),
          TextTruncator(10)
        )
      )
    val targetConfig: TextData.Configuration = TextData.Configuration()
      .setTextProcessors(
        Arrays.asList(
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
    val dataset: TatoebaEnglishFrenchDataset = datasetBuilder
      .setSourceConfiguration(sourceConfig)
      .setTargetConfiguration(targetConfig)
      .build()
    dataset.prepare(ProgressBar())
    return dataset
  }
}

class Arguments {
  var epoch = 0
    protected set
  var batchSize = 0
    protected set
  var maxGpus = 0
    protected set
  var isSymbolic = false
    protected set
  var isPreTrained = false
    protected set
  var outputDir: String? = null
    protected set
  var limit: Long = 0
    protected set
  var modelDir: String? = null
    protected set
  var criteria: Map<String, String>? = null
    protected set

  protected fun initialize() {
    epoch = 2
    maxGpus = Device.getGpuCount()
    outputDir = "build/model"
    limit = Long.MAX_VALUE
    modelDir = null
  }

  protected fun setCmd(cmd: CommandLine) {
    if (cmd.hasOption("epoch")) {
      epoch = cmd.getOptionValue("epoch").toInt()
    }
    if (cmd.hasOption("max-gpus")) {
      maxGpus = Math.min(cmd.getOptionValue("max-gpus").toInt(), maxGpus)
    }
    batchSize = if (cmd.hasOption("batch-size")) {
      cmd.getOptionValue("batch-size").toInt()
    } else {
      if (maxGpus > 0) 32 * maxGpus else 32
    }
    isSymbolic = cmd.hasOption("symbolic-model")
    isPreTrained = cmd.hasOption("pre-trained")
    if (cmd.hasOption("output-dir")) {
      outputDir = cmd.getOptionValue("output-dir")
    }
    if (cmd.hasOption("max-batches")) {
      limit = cmd.getOptionValue("max-batches").toLong() * batchSize
    }
    if (cmd.hasOption("model-dir")) {
      modelDir = cmd.getOptionValue("model-dir")
    }
    if (cmd.hasOption("criteria")) {
      val type = object: TypeToken<Map<String?, Any?>?>() {}.type
      criteria = JsonUtils.GSON.fromJson(cmd.getOptionValue("criteria"), type)
    }
  }

  fun parseArgs(args: Array<String?>?): Arguments? {
    initialize()
    val options = options
    try {
      val parser = DefaultParser()
      val cmd = parser.parse(options, args, null, false)
      if (cmd.hasOption("help")) {
        printHelp("./gradlew run --args='[OPTIONS]'", options)
        return null
      }
      setCmd(cmd)
      return this
    } catch (e: ParseException) {
      printHelp("./gradlew run --args='[OPTIONS]'", options)
    }
    return null
  }

  val options: Options
    get() {
      val options = Options()
      options.addOption(
        Option.builder("h").longOpt("help").hasArg(false)
          .desc("Print this help.").build()
      )
      options.addOption(
        Option.builder("e")
          .longOpt("epoch")
          .hasArg()
          .argName("EPOCH")
          .desc("Numbers of epochs user would like to run")
          .build()
      )
      options.addOption(
        Option.builder("b")
          .longOpt("batch-size")
          .hasArg()
          .argName("BATCH-SIZE")
          .desc("The batch size of the training data.")
          .build()
      )
      options.addOption(
        Option.builder("g")
          .longOpt("max-gpus")
          .hasArg()
          .argName("MAXGPUS")
          .desc("Max number of GPUs to use for training")
          .build()
      )
      options.addOption(
        Option.builder("s")
          .longOpt("symbolic-model")
          .argName("SYMBOLIC")
          .desc("Use symbolic model, use imperative model if false")
          .build()
      )
      options.addOption(
        Option.builder("p")
          .longOpt("pre-trained")
          .argName("PRE-TRAINED")
          .desc("Use pre-trained weights")
          .build()
      )
      options.addOption(
        Option.builder("o")
          .longOpt("output-dir")
          .hasArg()
          .argName("OUTPUT-DIR")
          .desc("Use output to determine directory to save your model parameters")
          .build()
      )
      options.addOption(
        Option.builder("m")
          .longOpt("max-batches")
          .hasArg()
          .argName("max-batches")
          .desc(
            "Limit each epoch to a fixed number of iterations to test the training script"
          )
          .build()
      )
      options.addOption(
        Option.builder("d")
          .longOpt("model-dir")
          .hasArg()
          .argName("MODEL-DIR")
          .desc("pre-trained model file directory")
          .build()
      )
      options.addOption(
        Option.builder("r")
          .longOpt("criteria")
          .hasArg()
          .argName("CRITERIA")
          .desc("The criteria used for the model.")
          .build()
      )
      return options
    }

  private fun printHelp(msg: String, options: Options) {
    val formatter = HelpFormatter()
    formatter.leftPadding = 1
    formatter.width = 120
    formatter.printHelp(msg, options)
  }
}