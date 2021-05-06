package edu.mcgill.gymfs.experiments

import net.automatalib.automata.fsa.impl.compact.CompactDFA
import net.automatalib.util.automata.builders.AutomatonBuilders
import net.automatalib.words.impl.Alphabets

fun main() {
//  println(synthesizeRegex("asdf", "testasdf"))
  val dfa: CompactDFA<Char> =
    AutomatonBuilders.forDFA(CompactDFA(Alphabets.characters('a', 'z')))
    .withInitial("q0")
    .from("q0")
    .on('a').to("q1")
    .on('b').to("q2").from("q1")
    .on('a').to("q0")
    .on('b').to("q3").from("q2")
    .on('a').to("q3")
    .on('b').to("q0").from("q3")
    .on('a').to("q2")
    .on('b').to("q3")
    .withAccepting("q0")
    .create()

  println(dfa.accepts("abababa".toCharArray().toList()))
}