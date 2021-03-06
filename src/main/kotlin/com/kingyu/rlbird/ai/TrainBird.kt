package com.kingyu.rlbird.ai

import ai.djl.*
import ai.djl.modality.rl.agent.*
import ai.djl.modality.rl.env.RlEnv
import ai.djl.ndarray.*
import ai.djl.ndarray.types.Shape
import ai.djl.nn.*
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.*
import com.kingyu.rlbird.game.FlappyBird
import com.kingyu.rlbird.rl.agent.QAgent
import com.kingyu.rlbird.util.*
import org.apache.commons.cli.ParseException
import org.slf4j.LoggerFactory
import java.io.IOException
import java.nio.file.Paths
import java.util.concurrent.*

object TrainBird {
  private val logger = LoggerFactory.getLogger(TrainBird::class.java)
  const val OBSERVE = 1000 // gameSteps to observe before training
  const val EXPLORE = 3000000 // frames over which to anneal epsilon
  const val SAVE_EVERY_STEPS = 100000 // save model every 100,000 step
  const val REPLAY_BUFFER_SIZE = 50000 // number of previous transitions to remember
  const val REWARD_DISCOUNT = 0.9f // decay rate of past observations
  const val INITIAL_EPSILON = 0.01f
  const val FINAL_EPSILON = 0.0001f
  const val PARAMS_PREFIX = "dqn-trained"
  var batchSteps = arrayOf<RlEnv.Step>()

  @Throws(
    ParseException::class,
    IOException::class,
    MalformedModelException::class
  )
  @JvmStatic
  fun main(args: Array<String>) {
    val arguments: Arguments = Arguments.parseArgs(args)
    val model = createOrLoadModel(arguments)
    if (arguments.isTesting) test(model) else train(arguments, model)
  }

  @Throws(IOException::class, MalformedModelException::class)
  fun createOrLoadModel(arguments: Arguments): Model {
    val model = Model.newInstance("QNetwork")
    model.block = block
    if (arguments.usePreTrained())
      model.load(Paths.get(Constant.MODEL_PATH), PARAMS_PREFIX)
    return model
  }

  fun train(arguments: Arguments, model: Model) {
    val withGraphics = arguments.withGraphics()
    val training = !arguments.isTesting
    val batchSize = arguments.batchSize // size of mini batch
    val game = FlappyBird(
      NDManager.newBaseManager(),
      batchSize,
      REPLAY_BUFFER_SIZE,
      withGraphics
    )
    model.newTrainer(trainingConfig).use { trainer ->
      trainer.initialize(Shape(batchSize.toLong(), 4, 80, 80))
      trainer.notifyListeners { listener: TrainingListener ->
        listener.onTrainingBegin(trainer)
      }
      var agent: RlAgent = QAgent(trainer, REWARD_DISCOUNT)
      val exploreRate: Tracker = LinearTracker.builder()
        .setBaseValue(INITIAL_EPSILON)
        .optSlope(-(INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE)
        .optMinValue(FINAL_EPSILON)
        .build()
      agent = EpsilonGreedy(agent, exploreRate)

      val numOfThreads = 2
      val callables: MutableList<Callable<Any?>> = ArrayList(numOfThreads)
      callables.add(GeneratorCallable(game, agent, training))
      if (training) callables.add(TrainerCallable(model, agent))

      val executorService = Executors.newFixedThreadPool(numOfThreads)
      try {
        try {
          val futures: MutableList<Future<Any?>> = ArrayList()
          for (callable in callables)
            futures.add(executorService.submit(callable))
          for (future in futures) future.get()
        } catch (e: InterruptedException) {
          logger.error("", e)
        } catch (e: ExecutionException) {
          logger.error("", e)
        }
      } finally {
        executorService.shutdown()
      }
    }
  }

  fun test(model: Model) =
    model.newTrainer(trainingConfig).use { trainer ->
      val agent: RlAgent = QAgent(trainer, REWARD_DISCOUNT)
      val game = FlappyBird(NDManager.newBaseManager(), 1, 1, true)
      while (true) game.runEnvironment(agent, false)
    }

  // conv -> conv -> conv -> fc -> fc
  val block: SequentialBlock = SequentialBlock()
        .add(
          Conv2d.builder()
            .setKernelShape(Shape(8, 8))
            .optStride(Shape(4, 4))
            .optPadding(Shape(3, 3))
            .setFilters(4).build()
        )
        .add { arrays: NDList? -> Activation.relu(arrays) }
        .add(
          Conv2d.builder()
            .setKernelShape(Shape(4, 4))
            .optStride(Shape(2, 2))
            .setFilters(32).build()
        )
        .add { arrays: NDList? -> Activation.relu(arrays) }
        .add(
          Conv2d.builder()
            .setKernelShape(Shape(3, 3))
            .optStride(Shape(1, 1))
            .setFilters(64).build()
        )
        .add { arrays: NDList? -> Activation.relu(arrays) }
        .add(Blocks.batchFlattenBlock())
        .add(Linear.builder().setUnits(512).build())
        .add { arrays: NDList? -> Activation.relu(arrays) }
        .add(Linear.builder().setUnits(2).build())

  val trainingConfig =
    DefaultTrainingConfig(Loss.l2Loss())
      .optOptimizer(
        Adam.builder().optLearningRateTracker(Tracker.fixed(1e-6f)).build()
      )
      .addEvaluator(Accuracy())
      .optInitializer(NormalInitializer())
      .addTrainingListeners(*TrainingListener.Defaults.basic())

  private class TrainerCallable(
    private val model: Model,
    private val agent: RlAgent
  ): Callable<Any?> {
    @Throws(Exception::class)
    override fun call(): Any? {
      while (FlappyBird.trainStep < EXPLORE) {
        Thread.sleep(0)
        if (FlappyBird.gameStep > OBSERVE) {
          agent.trainBatch(batchSteps)
          FlappyBird.trainStep++
          if (FlappyBird.trainStep > 0 && FlappyBird.trainStep % SAVE_EVERY_STEPS == 0) {
            model.save(
              Paths.get(Constant.MODEL_PATH),
              "dqn-" + FlappyBird.trainStep
            )
          }
        }
      }
      return null
    }
  }

  private class GeneratorCallable(
    private val game: FlappyBird,
    private val agent: RlAgent,
    private val training: Boolean
  ): Callable<Any?> {
    override fun call(): Any? {
      while (FlappyBird.trainStep < EXPLORE) {
        game.runEnvironment(agent, training)
        batchSteps = game.batch
      }
      return null
    }
  }
}