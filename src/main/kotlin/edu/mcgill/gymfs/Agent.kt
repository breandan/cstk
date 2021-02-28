package edu.mcgill.gymfs

import ai.djl.*
import ai.djl.ndarray.types.Shape
import ai.djl.nn.*
import ai.djl.nn.convolutional.Conv2d
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.loss.Loss
import ai.djl.util.JsonUtils
import com.google.gson.reflect.TypeToken
import org.apache.commons.cli.*

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

class Arguments {
  open var epoch = 0
  var batchSize = 0
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

  fun setCmd(cmd: CommandLine) {
    if (cmd.hasOption("epoch")) epoch = cmd.getOptionValue("epoch").toInt()
    if (cmd.hasOption("max-gpus"))
      maxGpus = cmd.getOptionValue("max-gpus").toInt().coerceAtMost(maxGpus)
    batchSize = if (cmd.hasOption("batch-size")) {
      cmd.getOptionValue("batch-size").toInt()
    } else {
      if (maxGpus > 0) 32 * maxGpus else 32
    }
    isSymbolic = cmd.hasOption("symbolic-model")
    isPreTrained = cmd.hasOption("pre-trained")
    if (cmd.hasOption("output-dir"))
      outputDir = cmd.getOptionValue("output-dir")
    if (cmd.hasOption("max-batches"))
      limit = cmd.getOptionValue("max-batches").toLong() * batchSize
    if (cmd.hasOption("model-dir")) modelDir = cmd.getOptionValue("model-dir")
    if (cmd.hasOption("criteria")) {
      val type = object: TypeToken<Map<String?, Any?>?>() {}.type
      criteria = JsonUtils.GSON.fromJson(cmd.getOptionValue("criteria"), type)
    }
  }

  fun parseArgs(args: Array<String>): Arguments {
    initialize()
    val options = options
    val parser = DefaultParser()
    val cmd = parser.parse(options, args, null, false)
    if (cmd.hasOption("help"))
      printHelp("./gradlew run --args='[OPTIONS]'", options)
    setCmd(cmd)
    return this
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