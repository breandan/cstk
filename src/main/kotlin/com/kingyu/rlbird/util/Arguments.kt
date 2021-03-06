package com.kingyu.rlbird.util

import org.apache.commons.cli.*

class Arguments(cmd: CommandLine) {
  constructor(args: Array<String>?):
    this(DefaultParser().parse(options, args, null, false))

  var batchSize =
    if (cmd.hasOption("batch-size")) cmd.getOptionValue("batch-size").toInt()
    else 32
  val graphics: Boolean = cmd.hasOption("graphics")
  val preTrained: Boolean = cmd.hasOption("pre-trained")
  val isTesting: Boolean = cmd.hasOption("testing")

  companion object {
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