package edu.mcgill.cstk.disk

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.Graph
import guru.nidi.graphviz.*
import guru.nidi.graphviz.attribute.*
import guru.nidi.graphviz.attribute.Color.TRANSPARENT
import guru.nidi.graphviz.attribute.GraphAttr.COMPOUND
import guru.nidi.graphviz.attribute.GraphAttr.CONCENTRATE
import guru.nidi.graphviz.engine.*
import guru.nidi.graphviz.model.*
import java.io.File

fun <G: Graph<G, E, V>, E: Edge<G, E, V>, V: Vertex<G, E, V>>
  Graph<G, E, V>.renderVKG(): MutableGraph =
  graph(directed = true, strict = true) {
    edge[Arrow.NORMAL, Style.lineWidth(THICKNESS)]
    graph[CONCENTRATE, TRANSPARENT.background(), COMPOUND]
    node[
      Attributes.attr("shape", "circle"),
      Attributes.attr("style", "filled"),
      Attributes.attr("fillcolor", "black"),
      Attributes.attr("label", ""),
    ]

    for (vertex in vertices) {
      Factory.mutNode(vertex.id).also {
        if (vertex is LGVertex && vertex.occupied) it.add("fillcolor", "red")
      }
      for (n in vertex.neighbors)
        Factory.mutNode(n.id) - Factory.mutNode(vertex.id)
    }
  }

fun MutableGraph.show(filename: String = "temp") =
  render(Format.PNG).run {
    toFile(File.createTempFile(filename, ".png"))
  }.show()

// TODO: replace with adj list constructor
fun <T> List<Pair<T, T>>.toLabeledGraph(
  toVertex: T.() -> LGVertex = { LGVertex(hashCode().toString()) }
): LabeledGraph =
  fold(first().first.toVertex().graph) { acc, (s, t) ->
    val (v, w) = s.toVertex() to t.toVertex()
    acc + LabeledGraph { v - w; w - v }
  }