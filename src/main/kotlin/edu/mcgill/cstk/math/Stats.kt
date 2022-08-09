package edu.mcgill.cstk.math

import kotlin.math.pow

fun List<Double>.variance() =
  average().let { μ -> map { (it - μ).pow(2) } }.average()