package com.kingyu.rlbird.util

import ai.djl.modality.cv.*
import ai.djl.modality.cv.util.NDImageUtils
import ai.djl.ndarray.*
import java.awt.image.BufferedImage
import java.io.*
import javax.imageio.ImageIO

/**
 * 工具类，游戏中用到的工具都在此类
 *
 * @author Kingyu
 */
object GameUtil {
  /**
   * 装载图片
   *
   * @param imgPath 图片路径
   * @return 图片资源
   */
  fun loadBufferedImage(imgPath: String?): BufferedImage? {
    try {
      return ImageIO.read(FileInputStream(imgPath))
    } catch (e: IOException) {
      e.printStackTrace()
    }
    return null
  }

  /**
   * 返回指定区间的一个随机数
   *
   * @param min 区间最小值，包含
   * @param max 区间最大值，不包含
   * @return 该区间的随机数
   */
  fun getRandomNumber(min: Int, max: Int): Int =
    (Math.random() * (max - min) + min).toInt()

  /**
   * Image preprocess
   *
   * @param observation input BufferedImage
   * @return NDArray:Shape(80,80,1)
   */
  fun imgPreprocess(observation: BufferedImage?): NDArray =
    NDImageUtils.toTensor(
      NDImageUtils.resize(
        ImageFactory.getInstance().fromImage(observation)
          .toNDArray(NDManager.newBaseManager(), Image.Flag.GRAYSCALE), 80, 80
      )
    )
}