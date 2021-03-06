package com.kingyu.rlbird.util

import org.apache.commons.cli.*

class Arguments(cmd: CommandLine) {
  var batchSize =
    if (cmd.hasOption("batch-size")) cmd.getOptionValue("batch-size").toInt()
    else 32
  private val graphics: Boolean = true
  private val preTrained: Boolean = true
  val isTesting: Boolean = true

  fun withGraphics(): Boolean = graphics

  fun usePreTrained(): Boolean = preTrained

  companion object {
    @Throws(ParseException::class)
    fun parseArgs(args: Array<String>?): Arguments =
      Arguments(DefaultParser().parse(options, args, null, false))

    val options: Options
      get() {
        val options = Options()
        options.addOption(
          Option.builder("g")
            .longOpt("graphics")
            .argName("GRAPHICS")
            .desc("Training with graphics")
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
          Option.builder("p")
            .longOpt("pre-trained")
            .argName("PRE-TRAINED")
            .desc("Use pre-trained weights")
            .build()
        )
        options.addOption(
          Option.builder("t")
            .longOpt("testing")
            .argName("TESTING")
            .desc("test the trained model")
            .build()
        )
        return options
      }
  }
}