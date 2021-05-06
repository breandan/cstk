package edu.mcgill.gymfs.experiments

import edu.mcgill.gymfs.disk.synthesizeRegex
import net.automatalib.automata.fsa.impl.compact.CompactDFA
import net.automatalib.util.automata.builders.AutomatonBuilders
import net.automatalib.words.impl.Alphabets

// TODO: DFA/RegEx or BoW query?

// https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example1.java
// https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example2.java
// https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/Example3.java
// Passive: https://github.com/LearnLib/learnlib/blob/develop/examples/src/main/java/de/learnlib/examples/passive/Example1.java

fun main() {
//  https://github.com/LearnLib/learnlib/blob/develop/test-support/learning-examples/src/main/java/de/learnlib/examples/dfa/ExampleAngluin.java
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
  println(synthesizeRegex("asdf", "testasdf"))
}