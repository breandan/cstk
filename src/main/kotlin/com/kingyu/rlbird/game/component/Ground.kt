package com.kingyu.rlbird.game.component

import com.kingyu.rlbird.util.*
import java.awt.Graphics

/**
 *
 * @author Kingyu
 */
class Ground {
  private val velocity: Int = Constant.GAME_SPEED
  private var layerX: Int = 0

  companion object {
    private val BackgroundImg = GameUtil.loadBufferedImage(Constant.BG_IMG_PATH)!!
    val GROUND_HEIGHT = BackgroundImg.height
  }

  fun draw(g: Graphics, bird: Bird?) {
    if (bird!!.isDead) { return }
    val imgWidth = BackgroundImg.width
    val count = Constant.FRAME_WIDTH / imgWidth + 2 // 根据窗口宽度得到图片的绘制次数
    for (i in 0 until count) {
      g.drawImage(
        BackgroundImg,
        imgWidth * i - layerX,
        Constant.FRAME_HEIGHT - GROUND_HEIGHT,
        null
      )
    }
    movement()
  }

  private fun movement() {
    layerX += velocity
    if (layerX > BackgroundImg.width) layerX = 0
  }
}