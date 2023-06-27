package edu.mcgill.cstk.math

import java.math.*
import kotlin.math.pow

fun List<Double>.variance() =
  average().let { μ -> map { (it - μ).pow(2) } }.average()

fun List<BigDecimal>.mean() =
  sumOf { it }.divide(size.toBigDecimal(),15, RoundingMode.HALF_UP)
    .round(MathContext(5))

fun List<BigDecimal>.variance() =
  mean().let { μ -> map { (it - μ).pow(2) } }.mean()