package edu.mcgill.cstk.utils

fun String.execute() =
  ProcessBuilder( split(" ") ).start().waitFor()
