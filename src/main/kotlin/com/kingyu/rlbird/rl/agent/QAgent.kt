package com.kingyu.rlbird.rl.agent

import ai.djl.modality.rl.env.RlEnv
import ai.djl.ndarray.NDList
import ai.djl.training.Trainer
import org.slf4j.LoggerFactory
import java.util.*

/**
 * Constructs a [ai.djl.modality.rl.agent.QAgent] with a custom [Batchifier].
 *
 * @param trainer        the trainer for the model to learn
 * @param rewardDiscount the reward discount to apply to rewards from future states
 */
class QAgent(private val trainer: Trainer, rewardDiscount: Float):
  ai.djl.modality.rl.agent.QAgent(trainer, rewardDiscount) {
  private val logger = LoggerFactory.getLogger(QAgent::class.java)

  override fun chooseAction(env: RlEnv, training: Boolean): NDList {
    val actionSpace = env.actionSpace
    val actionReward = trainer.evaluate(env.observation).singletonOrThrow()[0]
    logger.info(Arrays.toString(actionReward.toFloatArray()))
    val bestAction = actionReward.argMax().getLong().toInt()
    return actionSpace[bestAction]
  }
}