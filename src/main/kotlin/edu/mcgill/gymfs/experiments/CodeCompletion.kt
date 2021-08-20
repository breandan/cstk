package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.*

fun main() {
  val query = "System.out."
  println("Query: $query")
  println("Completion: " + complete(query, 3))
}