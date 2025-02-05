package edu.mcgill.cstk.utils

import ai.hypergraph.kaliningraph.parsing.PSingleton
import ai.hypergraph.kaliningraph.parsing.PTree
import ai.hypergraph.kaliningraph.parsing.bimap
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.makeLevFSA
import ai.hypergraph.kaliningraph.parsing.noEpsilonOrNonterminalStubs
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.parsing.pretty
import ai.hypergraph.kaliningraph.parsing.prettyPrint
import ai.hypergraph.kaliningraph.parsing.vindex
import kotlin.random.Random
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.visualization.html
import ai.hypergraph.kaliningraph.visualization.show


fun generatePrettyLaTeXArray(
  data: Array<Array<Array<Boolean>>>,
  cond: (Int, Int) -> Boolean = { r, c -> r < c },
  rowLabels: List<String> = (0..<data.size).map { it.toString() },
  colLabels: List<String> = rowLabels
): String {
  // A StringBuilder to accumulate the lines
  val sb = StringBuilder()

  // Start the LaTeX array environment
  sb.append("\\[\n")
  sb.append("\\begin{array}{c|")  // 'c|' for the row label, then columns
  repeat(colLabels.size) {
    sb.append("c")
  }
  sb.append("}\n")

  // Top row: blank space (for the row label column) + column labels
  sb.append("  & ")
  colLabels.forEachIndexed { index, label ->
    sb.append(label)
    if (index < colLabels.size - 1) sb.append(" & ")
  }
  sb.append(" \\\\ \\hline\n")

  // Fill in each row
  for (r in data.indices) {
    // Row label
    sb.append(rowLabels[r])
    sb.append(" & ")

    // Fill each column in row r
    val rowData = data[r]
    for (c in rowData.indices) {
      val cell = rowData[c]

      // Convert each Boolean in the cell to \bs or \ws
      val cellContent = cell.joinToString(separator = "") { isBlack ->
        if (cond(r, c)) {
          if (isBlack) "\\bs" else "\\ws"
        } else "   "
      }
      sb.append(cellContent)

      // Separate columns with '&', except after the last column
      if (c < rowData.size - 1) {
        sb.append(" & ")
      }
    }

    // End of the row
    sb.append(" \\\\ [6pt]\n")
  }

  // End the LaTeX array environment
  sb.append("\\end{array}\n")
  sb.append("\\]\n")

  // Return the completed string
  return sb.toString()
}

fun String.lsaStateToString() = "q_{" + drop(2).replace("/", "") + "}"


fun main() {
  println(dyck.prettyPrint())
  println()
  println(dyck.nonterminals)
  println()

  val lfsa = makeLevFSA("( ) ) ) )", 3)
  val states = lfsa.stateLst.map { it.lsaStateToString() }

  println(states)
  println(lfsa.graph.toDot())

//  val cfg = dyck
//  val ap: Map<Pair<Int, Int>, Set<Int>> = lfsa.allPairs
//
//
//  val dp = Array(8) { Array(8) { Array(4) { false } } }
//
//  val aitx = lfsa.allIndexedTxs1(cfg)
//  for ((p, σ, q) in aitx) {
//    val Aidxs = cfg.bimap.TDEPS[σ]!!.map { cfg.bindex[it] }
//    for (Aidx in Aidxs) {
//      dp[p][q][Aidx] = true
//    }
//  }
//
//  println(generatePrettyLaTeXArray(dp, rowLabels = states, colLabels = states))
//  println()
//
//  for (dist in 0 until lfsa.numStates) {
//    for (iP in 0 until lfsa.numStates - dist) {
//      val p = iP
//      val q = iP + dist
//      if (p to q !in ap) continue
//      val appq = ap[p to q]!!
//      for ((A, indexArray) in cfg.vindex.withIndex()) {
//        outerloop@for(j in 0..<indexArray.size step 2) {
//          val B = indexArray[j]
//          val C = indexArray[j + 1]
//          for (r in appq) {
//            if (dp[p][r][B] && dp[r][q][C]) {
//              dp[p][q][A] = true
//            }
//          }
//        }
//      }
//    }
//  }
//
//  println(generatePrettyLaTeXArray(dp, rowLabels = states, colLabels = states))
//  println()
}