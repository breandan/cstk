package com.kingyu.rlbird.game.component

import com.kingyu.rlbird.util.*
import java.awt.*
import java.awt.image.BufferedImage

/**
 * 水管类
 *
 * @author Kingyu
 */
class Pipe {
  companion object {
    var images: Array<BufferedImage> =
      Array(3) { GameUtil.loadBufferedImage(Constant.PIPE_IMG_PATH[it])!!}

    // 水管图片的宽高
    val PIPE_WIDTH = images[0].width
    val PIPE_HEIGHT = images[0].height
    val PIPE_HEAD_WIDTH = images[1].width
    val PIPE_HEAD_HEIGHT = images[1].height
    const val TYPE_TOP_NORMAL = 0
    const val TYPE_BOTTOM_NORMAL = 1

    // 上方管道加长
    const val TOP_PIPE_LENGTHENING = 100
  }

  var x = 0
    private set
  private var y = 0
  private val width : Int = PIPE_WIDTH
  private var height = 0
  var isVisible = false

  // 水管的类型
  var type = 0
  private val velocity: Int = Constant.GAME_SPEED
  var pipeCollisionRect: Rectangle = Rectangle().apply { width = PIPE_WIDTH }

  /**
   * 设置水管参数
   *
   * @param x: x坐标
   * @param y：y坐标
   * @param height：水管高度
   * @param type：水管类型
   * @param visible：水管可见性
   */
  fun setAttribute(x: Int, y: Int, height: Int, type: Int, visible: Boolean) {
    this.x = x
    this.y = y
    this.height = height
    this.type = type
    isVisible = visible
    setRectangle(this.x + 5, this.y, this.height) // 碰撞矩形位置补偿
  }

  /**
   * 设置碰撞矩形参数
   */
  fun setRectangle(x: Int, y: Int, height: Int) {
    pipeCollisionRect.x = x
    pipeCollisionRect.y = y
    pipeCollisionRect.height = height
  }

  fun draw(g: Graphics, bird: Bird?) {
    when (type) {
      TYPE_TOP_NORMAL -> drawTopNormal(g)
      TYPE_BOTTOM_NORMAL -> drawBottomNormal(g)
    }
    if (bird!!.isDead) {
      return
    }
    movement()
    //      //绘制碰撞矩形
//        g.setColor(Color.white);
//        g.drawRect((int) pipeRect.getX(), (int) pipeRect.getY(), (int) pipeRect.getWidth(), (int) pipeRect.getHeight());
  }

  // 绘制从上往下的普通水管
  private fun drawTopNormal(g: Graphics) {
    // 拼接的个数
    val count = (height - PIPE_HEAD_HEIGHT) / PIPE_HEIGHT + 1 // 取整+1
    // 绘制水管的主体
    for (i in 0 until count) {
      g.drawImage(images[0], x, y + i * PIPE_HEIGHT, null)
    }
    // 绘制水管的顶部
    g.drawImage(
      images[1], x - (PIPE_HEAD_WIDTH - width shr 1),
      height - TOP_PIPE_LENGTHENING - PIPE_HEAD_HEIGHT, null
    ) // 水管头部与水管主体的宽度不同，x坐标需要处理
  }

  // 绘制从下往上的普通水管
  private fun drawBottomNormal(g: Graphics) {
    // 拼接的个数
    val count: Int =
      (height - PIPE_HEAD_HEIGHT - Ground.GROUND_HEIGHT) / PIPE_HEIGHT + 1
    // 绘制水管的主体
    for (i in 0 until count) {
      g.drawImage(
        images[0],
        x,
        Constant.FRAME_HEIGHT - PIPE_HEIGHT - Ground.GROUND_HEIGHT - i * PIPE_HEIGHT,
        null
      )
    }
    // 绘制水管的顶部
    g.drawImage(
      images[2],
      x - (PIPE_HEAD_WIDTH - width shr 1),
      Constant.FRAME_HEIGHT - height,
      null
    )
  }

  private fun movement() {
    x -= velocity
    pipeCollisionRect.x -= velocity
    if (x < -1 * PIPE_HEAD_WIDTH) { // 水管完全离开了窗口
      isVisible = false
    }
  }

  /**
   * 判断当前水管是否完全出现在窗口中
   */
  val isInFrame: Boolean
    get() = x + width < Constant.FRAME_WIDTH

  internal object PipePool {
    val FULL_PIPE: Int = (Constant.FRAME_WIDTH / (PIPE_HEAD_WIDTH + GameElementLayer.HORIZONTAL_INTERVAL) + 2) * 2
    private val pool: MutableList<Pipe?> = Array(FULL_PIPE) { Pipe() }.toMutableList()

    // 容器内水管数量 = 窗口可容纳的水管数量+2， 由窗口宽度、水管宽度、水管间距算得
    const val MAX_PIPE_COUNT = 30 // 对象池中对象的最大个数

    /**
     * 从对象池中获取一个对象
     *
     * @return pipe from pipePool
     */
    fun get(): Pipe? = if (pool.size > 0) {
      pool.removeAt(pool.size - 1) // 移除并返回最后一个
    } else {
      Pipe() // 空对象池，返回一个新对象
    }

    /**
     * 归还对象给容器
     */
    fun giveBack(pipe: Pipe?) {
      if (pool.size < MAX_PIPE_COUNT) {
        pool.add(pipe)
      }
    }
  }
}