package com.kingyu.rlbird.game.component

import com.kingyu.rlbird.game.FlappyBird
import com.kingyu.rlbird.util.*
import java.awt.*
import java.awt.image.BufferedImage

/**
 * 小鸟类，小鸟的绘制与飞行逻辑都在此类
 *
 * @author Kingyu
 */
class Bird {
  val birdX: Int = Constant.FRAME_WIDTH shr 2
  private var y: Int = Constant.FRAME_HEIGHT shr 1

  // 小鸟的状态
  private var birdState = 0
  val rectX = birdX - (BIRD_WIDTH shr 1)
  val rectY = y - (BIRD_HEIGHT shr 1) + RECT_DESCALE * 2
  val birdCollisionRect: Rectangle = Rectangle(
  rectX + RECT_DESCALE,
  rectY + RECT_DESCALE * 2,
  BIRD_WIDTH - RECT_DESCALE * 3,
  BIRD_HEIGHT - RECT_DESCALE * 4
  ) // 碰撞矩形的坐标与小鸟相同


  companion object {
    const val BIRD_READY = 0
    const val BIRD_FALL = 1
    const val BIRD_DEAD = 2
    const val RECT_DESCALE = 2 // 碰撞矩形宽高的补偿参数
    var birdImages: BufferedImage = GameUtil.loadBufferedImage(Constant.BIRDS_IMG_PATH)!!
    val BIRD_WIDTH = birdImages.width
    val BIRD_HEIGHT = birdImages.height
    const val ACC_FLAP = 15 // players speed on flapping
    const val ACC_Y = 4.0 // players downward acceleration
    const val MAX_VEL_Y = -25 // max vel along Y, max descend speed
    val BOTTOM_BOUNDARY: Int =
      Constant.FRAME_HEIGHT - Ground.GROUND_HEIGHT - (BIRD_HEIGHT shr 1)
  }

  fun draw(g: Graphics) {
    movement()
    drawBirdImg(g)
    //        g.setColor(Color.white);
//        g.drawRect((int) birdCollisionRect.getX(), (int)birdCollisionRect.getY(), (int) birdCollisionRect.getWidth(), (int) birdCollisionRect.getHeight());
  }

  fun drawBirdImg(g: Graphics) {
    g.drawImage(
      birdImages,
      birdX - (BIRD_WIDTH shr 1),
      y - (BIRD_HEIGHT shr 1),
      null
    )
  }

  private var velocity = 0 // bird's velocity along Y, default same as playerFlapped

  private fun movement() {
    if (velocity > MAX_VEL_Y) velocity -= ACC_Y.toInt()
    y = (y - velocity).coerceAtMost(BOTTOM_BOUNDARY)
    birdCollisionRect.y = birdCollisionRect.y - velocity
    if (birdCollisionRect.y < GameElementLayer.MIN_HEIGHT ||
      birdCollisionRect.y > GameElementLayer.MAX_HEIGHT + GameElementLayer.VERTICAL_INTERVAL
    ) {
      FlappyBird.setCurrentReward(0.1f)
    }
    if (birdCollisionRect.y < Constant.WINDOW_BAR_HEIGHT) {
      die()
    }
    if (birdCollisionRect.y >= BOTTOM_BOUNDARY - 10) {
      die()
    }
  }

  fun birdFlap() {
    if (isDead) return
    velocity = ACC_FLAP
  }

  fun die() {
    FlappyBird.setCurrentReward(-1f)
    FlappyBird.setCurrentTerminal(true)
    FlappyBird.setGameState(FlappyBird.GAME_OVER)
    birdState = BIRD_DEAD
  }

  val isDead: Boolean
    get() = birdState == BIRD_FALL || birdState == BIRD_DEAD

  fun reset() {
    birdState = BIRD_READY
    y = Constant.FRAME_HEIGHT shr 1
    velocity = 0
    birdCollisionRect.y = y + RECT_DESCALE * 4 - birdImages.height / 2
    ScoreCounter.reset()
  }
}