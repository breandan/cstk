package com.kingyu.rlbird.game.component

import com.kingyu.rlbird.game.FlappyBird
import com.kingyu.rlbird.game.component.Pipe.PipePool
import com.kingyu.rlbird.util.*
import java.awt.Graphics

/**
 * 游戏元素层，水管的生成方法
 *
 * @author Kingyu
 */
class GameElementLayer {
  private val pipes : MutableList<Pipe?> = ArrayList()

  fun draw(g: Graphics, bird: Bird?) {
    // 遍历水管容器，如果可见则绘制，不可见则归还
    var i = 0
    while (i < pipes.size) {
      val pipe = pipes[i]
      if (pipe!!.isVisible) {
        pipe.draw(g, bird)
      } else {
        val remove = pipes.removeAt(i)
        PipePool.giveBack(remove)
        i--
      }
      i++
    }
    bird!!.drawBirdImg(g)
    isCollideBird(bird)
    generatePipe(bird)
  }

  /**
   * 当容器中添加的最后一个水管完全显示到屏幕后，添加下一对；
   */
  private fun generatePipe(bird: Bird?) {
    if (bird!!.isDead) {
      return
    }
    if (pipes.size == 0) {
      // 若容器为空，则添加一对水管
      val topHeight =
        GameUtil.getRandomNumber(MIN_HEIGHT, MAX_HEIGHT + 1) // 随机生成水管高度
      val top = PipePool.get()
      top!!.setAttribute(
        Constant.FRAME_WIDTH,
        -Pipe.TOP_PIPE_LENGTHENING,
        topHeight + Pipe.TOP_PIPE_LENGTHENING,
        Pipe.TYPE_TOP_NORMAL,
        true
      )
      val bottom = PipePool.get()
      bottom!!.setAttribute(
        Constant.FRAME_WIDTH,
        topHeight + VERTICAL_INTERVAL,
        Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL,
        Pipe.TYPE_BOTTOM_NORMAL,
        true
      )
      pipes.add(top)
      pipes.add(bottom)
    } else {
      // 判断最后一对水管是否完全进入游戏窗口，若进入则添加水管
      val lastPipe = pipes[pipes.size - 1]!! // 获得容器中最后一个水管
      val currentDistance: Int =
        lastPipe.x - bird.birdX + Bird.BIRD_WIDTH / 2 // 小鸟和最后一根水管的距离
      val SCORE_DISTANCE: Int =
        Pipe.PIPE_WIDTH * 2 + HORIZONTAL_INTERVAL // 小于得分距离则得分
      if (pipes.size >= PipePool.FULL_PIPE && currentDistance <= SCORE_DISTANCE + Pipe.PIPE_WIDTH * 3 / 2 && currentDistance > SCORE_DISTANCE + Pipe.PIPE_WIDTH * 3 / 2 - Constant.GAME_SPEED) {
        FlappyBird.setCurrentReward(0.8f)
      }
      if (pipes.size >= PipePool.FULL_PIPE && currentDistance <= SCORE_DISTANCE && currentDistance > SCORE_DISTANCE - Constant.GAME_SPEED) {
        ScoreCounter.score(bird)
      }
      if (lastPipe.isInFrame) {
        addNormalPipe(lastPipe)
      }
    }
  }

  /**
   * 添加普通水管
   *
   * @param lastPipe 最后一根水管
   */
  private fun addNormalPipe(lastPipe: Pipe) {
    val topHeight =
      GameUtil.getRandomNumber(MIN_HEIGHT, MAX_HEIGHT + 1) // 随机生成水管高度
    val x =
      lastPipe.x + HORIZONTAL_INTERVAL // 新水管的x坐标 = 最后一对水管的x坐标 + 水管的间隔
    val top = PipePool.get()
    top!!.setAttribute(
      x,
      -Pipe.TOP_PIPE_LENGTHENING,
      topHeight + Pipe.TOP_PIPE_LENGTHENING,
      Pipe.TYPE_TOP_NORMAL,
      true
    )
    val bottom = PipePool.get()
    bottom!!.setAttribute(
      x,
      topHeight + VERTICAL_INTERVAL,
      Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL,
      Pipe.TYPE_BOTTOM_NORMAL,
      true
    )
    pipes.add(top)
    pipes.add(bottom)
  }

  /**
   * 判断元素和小鸟是否发生碰撞
   *
   * @param bird bird
   */
  fun isCollideBird(bird: Bird?) {
    if (bird!!.isDead) {
      return
    }
    for (pipe in pipes) {
      if (pipe!!.pipeCollisionRect.intersects(bird.birdCollisionRect)) {
        bird.die()
        return
      }
    }
  }

  fun reset() {
    for (pipe in pipes) PipePool.giveBack(pipe)
    pipes.clear()
  }

  companion object {
    const val VERTICAL_INTERVAL = Constant.FRAME_HEIGHT shr 2
    const val HORIZONTAL_INTERVAL = Constant.FRAME_HEIGHT shr 2
    const val MIN_HEIGHT = Constant.FRAME_HEIGHT / 5
    const val MAX_HEIGHT = Constant.FRAME_HEIGHT / 3
  }
}