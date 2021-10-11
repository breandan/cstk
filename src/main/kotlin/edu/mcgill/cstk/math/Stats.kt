package edu.mcgill.cstk.math

import kotlin.math.pow

fun List<Double>.variance() =
  average().let { mean -> map { (it - mean).pow(2) } }.average()