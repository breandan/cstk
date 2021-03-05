package com.kingyu.rlbird.util

import java.awt.Color

/**
 * 常量类
 *
 * @author Kingyu 后续优化可写入数据库或文件中，便于修改
 */
object Constant {
  // 窗口尺寸
  const val FRAME_WIDTH = 288
  const val FRAME_HEIGHT = 512

  // 游戏标题
  const val GAME_TITLE = "RL Flappy Bird written by Kingyu"

  // 窗口位置
  const val FRAME_X = 0
  const val FRAME_Y = 0

  // 游戏速度（水管及背景层的移动速度）
  const val GAME_SPEED = 6

  // 游戏背景色
  val BG_COLOR = Color(0x000000)

  // 游戏刷新率
  const val FPS = 1000 / 30

  // 标题栏高度
  const val WINDOW_BAR_HEIGHT = 30

  // 小鸟动作
  val DO_NOTHING = intArrayOf(1, 0)
  val FLAP = intArrayOf(0, 1)

  // 图像资源路径
  const val BG_IMG_PATH = "src/main/resources/img/background.png"

  // 小鸟图片
  const val BIRDS_IMG_PATH = "src/main/resources/img/0.png"

  // 水管图片
  val PIPE_IMG_PATH = arrayOf<String?>(
    "src/main/resources/img/pipe.png", "src/main/resources/img/pipe_top.png",
    "src/main/resources/img/pipe_bottom.png"
  )
  const val SCORE_FILE_PATH = "src/main/resources/score" // 分数文件路径
  const val MODEL_PATH = "src/main/resources/model"
}