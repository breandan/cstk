package com.kingyu.rlbird.game.component

import com.kingyu.rlbird.game.FlappyBird

/**
 * 记分类, 单例类
 *
 * @author Kingyu
 */
object ScoreCounter {
  var currentScore: Long = 0
    private set

  fun score(bird: Bird?) {
    if (!bird!!.isDead) {
      FlappyBird.setCurrentReward(1f)
      currentScore += 1
    }
  }

  fun reset() {
    currentScore = 0
  }
}