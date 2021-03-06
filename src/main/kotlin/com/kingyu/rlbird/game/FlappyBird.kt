package com.kingyu.rlbird.game

import ai.djl.modality.rl.*
import ai.djl.modality.rl.env.RlEnv
import ai.djl.ndarray.*
import com.kingyu.rlbird.ai.TrainBird
import com.kingyu.rlbird.game.component.*
import com.kingyu.rlbird.util.*
import org.slf4j.LoggerFactory
import java.awt.*
import java.awt.event.*
import java.awt.image.BufferedImage
import java.util.*

/**
 * Constructs a [FlappyBird].
 *
 * @param manager      the manager for creating the game in
 * @param replayBuffer the replay buffer for storing data
 */
class FlappyBird(private val manager: NDManager, private val replayBuffer: ReplayBuffer): Frame(), RlEnv {
  private val ground: Ground = Ground()
  private val bird = Bird()
  private val gameElement: GameElementLayer = GameElementLayer()
  private var withGraphics = false
  private var currentImg: BufferedImage? = null
  private var currentObservation: NDList? = null
  private val actionSpace = ActionSpace()

  /**
   * Constructs a [FlappyBird] with a basic [LruReplayBuffer].
   *
   * @param manager          the manager for creating the game in
   * @param batchSize        the number of steps to train on per batch
   * @param replayBufferSize the number of steps to hold in the buffer
   */
  constructor(
    manager: NDManager,
    batchSize: Int,
    replayBufferSize: Int,
    withGraphics: Boolean
  ): this(manager, LruReplayBuffer(batchSize, replayBufferSize)) {
    this.withGraphics = withGraphics
    if (this.withGraphics) {
      initFrame()
      this.isVisible = true
    }

    actionSpace.add(NDList(manager.create(Constant.DO_NOTHING)))
    actionSpace.add(NDList(manager.create(Constant.FLAP)))
    currentImg = BufferedImage(
      Constant.FRAME_WIDTH,
      Constant.FRAME_HEIGHT,
      BufferedImage.TYPE_4BYTE_ABGR
    )
    currentObservation = createObservation(currentImg)
    setGameState(GAME_START)
  }

  var trainState = "observe"
    private set

  /**
   * {@inheritDoc}
   * action[0] == 1 : do nothing
   * action[1] == 1 : flap the bird
   */
  override fun step(action: NDList, training: Boolean): RlEnv.Step {
    trainState = if (gameStep <= TrainBird.OBSERVE) "observe" else "explore"
    gameStep++
    if (action.singletonOrThrow().getInt(1) == 1) bird.birdFlap()
    stepFrame()

    if (withGraphics) {
      repaint()
      try {
        Thread.sleep(Constant.FPS.toLong())
      } catch (e: InterruptedException) {
        e.printStackTrace()
      }
    }

    val preObservation = currentObservation
    currentObservation = createObservation(currentImg)
    val step = FlappyBirdStep(
      manager.newSubManager(),
      preObservation!!,
      currentObservation!!,
      action,
      currentReward,
      currentTerminal
    )

    if (training) replayBuffer.addStep(step)

    logger.info(
      "GAME_STEP " + gameStep +
        " / " + "TRAIN_STEP " + trainStep +
        " / " + trainState +
        " / " + "ACTION " + Arrays.toString(
        action.singletonOrThrow().toArray()
      ) +
        " / " + "REWARD " + step.reward.getFloat() +
        " / " + "SCORE " + ScoreCounter.currentScore
    )
    if (gameState == GAME_OVER) restartGame()

    return step
  }

  /**
   * {@inheritDoc}
   */
  override fun getObservation(): NDList = currentObservation!!

  /**
   * {@inheritDoc}
   */
  override fun getActionSpace(): ActionSpace = actionSpace

  /**
   * {@inheritDoc}
   */
  override fun getBatch(): Array<RlEnv.Step> = replayBuffer.batch

  /**
   * {@inheritDoc}
   */
  override fun close() = manager.close()

  /**
   * {@inheritDoc}
   */
  override fun reset() {
    currentReward = 0.2f
    currentTerminal = false
  }

  private val imgQueue: Queue<NDArray?> = ArrayDeque(4)

  /**
   * Convert image to CNN input.
   * Copy the initial frame image, stack into NDList,
   * then replace the fourth frame with the current frame to ensure that the batch picture is continuous.
   *
   * @param currentImg the image of current frame
   * @return the CNN input
   */
  fun createObservation(currentImg: BufferedImage?): NDList {
    val observation = GameUtil.imgPreprocess(currentImg)
    return if (imgQueue.isEmpty()) {
      for (i in 0..3) imgQueue.offer(observation)
      NDList(
        NDArrays.stack(
          NDList(
            observation,
            observation,
            observation,
            observation
          ), 1
        )
      )
    } else {
      imgQueue.remove()
      imgQueue.offer(observation)
      val buf = imgQueue.take(4)
      NDList(NDArrays.stack(NDList(buf[0], buf[1], buf[2], buf[3]), 1))
    }
  }

  internal class FlappyBirdStep(
    private val manager: NDManager,
    private val preObservation: NDList,
    private val postObservation: NDList,
    private val action: NDList,
    private val reward: Float,
    private val terminal: Boolean
  ): RlEnv.Step {
    /**
     * {@inheritDoc}
     */
    override fun getPreObservation(): NDList {
      return preObservation
    }

    /**
     * {@inheritDoc}
     */
    override fun getPostObservation(): NDList {
      return postObservation
    }

    override fun getPostActionSpace(): ActionSpace = TODO()

    override fun isDone(): Boolean = gameStep % 5000 == 0 || terminal

    /**
     * {@inheritDoc}
     */
    override fun getAction(): NDList = action

    /**
     * {@inheritDoc}
     */
    override fun getReward(): NDArray = manager.create(reward)

    /**
     * {@inheritDoc}
     */
    override fun close() = manager.close()
  }

  /**
   * Draw one frame by performing all elements' draw function.
   */
  fun stepFrame() {
    val bufG = currentImg!!.graphics
    bufG.color = Constant.BG_COLOR
    bufG.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT)
    ground.draw(bufG, bird)
    bird.draw(bufG)
    gameElement.draw(bufG, bird)
  }

  /**
   * Initialize the game frame
   */
  private fun initFrame() {
    setSize(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT)
    title = Constant.GAME_TITLE
    setLocation(Constant.FRAME_X, Constant.FRAME_Y)
    isResizable = false
    isVisible = true
    addWindowListener(object: WindowAdapter() {
      override fun windowClosing(e: WindowEvent) {
        System.exit(0)
      }
    })
  }

  /**
   * Restart game
   */
  private fun restartGame() {
    setGameState(GAME_START)
    gameElement.reset()
    bird.reset()
  }

  /**
   * {@inheritDoc}
   */
  override fun update(g: Graphics) {
    g.drawImage(currentImg, 0, 0, null)
  }

  companion object {
    private const val serialVersionUID = 1L
    private val logger = LoggerFactory.getLogger(FlappyBird::class.java)
    private var gameState = 0
    const val GAME_START = 1
    const val GAME_OVER = 2
    var gameStep = 0
    var trainStep = 0
    private var currentTerminal = false
    private var currentReward = 0.2f
    fun setGameState(gameState: Int) {
      this.gameState = gameState
    }

    fun setCurrentTerminal(currentTerminal: Boolean) {
      this.currentTerminal = currentTerminal
    }

    fun setCurrentReward(currentReward: Float) {
      this.currentReward = currentReward
    }
  }
}