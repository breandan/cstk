package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.defaultTokenizer
import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.disk.defaultModel
import edu.mcgill.cstk.utils.*
import java.io.File

val example_code = """
  def out_of_date(dep, targ):
    if not os.access(targ, os.F_OK:
        return True
    if not os.access(dep, os.F_OK):
        raise Exception, "File " + dep + " does not exist"
    if os.stat(dep)[8] > os.stat(targ)[8]:
        return True
    else:
        return False
""".defaultTokenizer()

// A very minimal CFG that can parse a subset of Python, e.g., the above code

val python_cfg: CFG = """
S -> w | w ( S ) | ( ) | w = S | w . S | S S | ( S )
S -> S , S | S ; S | S : S
S -> S + S | S - S | S * S | S / S | S % S
S -> S < S | S > S | S <= S | S >= S | S == S | S != S
S -> S and S | S or S | S not S | S in S | S not in S | S is S | S not is S
S -> S if S else S | S for S in S | S while S | S with S | S as S
S -> S [ S ] | S [ S : S ] | S [ S : S : S ] | S [ S , S ] | S [ S , S , S ]
S -> w | S S
S -> if S : S else : S
S -> return S
FUN -> def w ( ARGS ) :
ARGS -> w , ARGS | w
""".parseCFG()

val user_prompt = """
  # Is the following code syntactically valid?
  $example_code
  # Answer (Y/N): <MASK: {Y_prob/N_prob}>
"""

/*
./gradlew promptRepair
 */

fun main() {
  val badJson = File("/src/main/resources/datasets/python/bifi/data/orig_bad_code/orig.bad.json").readText()
  val goodJson = File("/src/main/resources/datasets/python/bifi/data/orig_good_code/orig.good.json").readLines().take(997).joinToString("\n") .dropLast(1) + "}"
  val badCode = Klaxon().parse<Map<String, Map<String, Any>>>(badJson)
  val goodCode = Klaxon().parse<Map<String, Map<String, Any>>>(goodJson)

  val allCodeSnippets = badCode!!.values.take(100).map { cs ->
    cs["code_string"].toString()
      .let { CodeSnippet(it, it.coarsen(), cs["msg"].toString()) } to false
  } + goodCode!!.values.take(100).map { cs ->
    cs["code_string"].toString()
      .let { CodeSnippet(it, it.coarsen(), cs["msg"].toString()) } to true
  }

  var runningAverageAccuracy = 0.0

  allCodeSnippets.shuffled().forEachIndexed { i, (cs, isValid) ->
    val prompt = """
      # Is the following code syntactically valid?
      ${cs.originalCode}
      # Answer (Y/N): ${defaultModel.mask}
    """

    val answers = defaultModel.makeQueryAndScore(prompt, listOf("Y", "N"))
    val prediction = answers.maxByOrNull { it.second }!!.first == "Y"
    val accuracy = if (prediction == isValid) 1.0 else 0.0
    runningAverageAccuracy = (runningAverageAccuracy * i + accuracy) / (i + 1)

    println("Accuracy: $runningAverageAccuracy")
  }
}