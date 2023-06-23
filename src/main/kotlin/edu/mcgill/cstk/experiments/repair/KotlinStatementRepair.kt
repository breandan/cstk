package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.markovian.mcmc.*
import edu.mcgill.cstk.utils.*
import org.apache.datasketches.frequencies.ErrorType
import org.intellij.lang.annotations.Language
import java.io.*
import kotlin.time.*


const val keywordFile = "src/main/resources/datasets/kotlin/keywords.txt"

val mostCommonKotlinKeywords by lazy {
  File(keywordFile)
    .readLines().map { it.trim() }.filter { it.isNotBlank() }
}

val mostCommonSymbols by lazy {
  mostCommonKotlinKeywords.filter { it.none { it.isLetterOrDigit() } }
}

val memory = 3
val windowsSize = 3
val P_kotlin: MarkovChain<Σᐩ> by lazy {
//  println("Top 1k most common tuples: ${P.topTuples(1000).joinToString("\n")}\n\n")
//  println("Top 1k most common tuples: ${P.topTuples(1000).joinToString("\n")}\n\n")
// TODO: prepare the prompt in the same way as the training data
  fetchKotlinExamples().map { "BOS $it EOS" }.map {
    it.tokenizeByWhitespace().asSequence().toMarkovChain(memory)
  }.fold(MarkovChain(memory = memory)) { a, b -> a + b }
}

/*
./gradlew kotlinStatementRepair
 */

fun main() {
//  fetchKotlinExamples()
//  collectMostCommonKeywords()

  evaluateSyntheticRepairBenchmarkOn(originalKotlinLines)
}

fun evaluateSyntheticRepairBenchmarkOn(dataset: String, postprocess: List<Repair>.() -> List<Repair> = { this }) {
  System.setErr(FilteredOutputStream(System.err))
  System.setOut(FilteredOutputStream(System.out))

  val scoreEdit: (List<Σᐩ>) -> Double = constructScoringFunction()

  val deck =
    (commonKotlinKeywords + "ε" - "w")
      .also { println("Full deck: $it") }
      .sortedBy { P_kotlin[it] }.reversed().take(32)
      .also { println("High frequency deck: $it") }

  val edits = 2
  // Generate synthetic error dataset
  List(100) { dataset }.joinToString("\n").lines().map {
    val original = it.lexAsKotlin().joinToString(" ").trim()
    val prompt = original.constructPromptByDeletingRandomSyntax(
      eligibleTokensForDeletion = officialKotlinKeywords + commonKotlinKeywords,
      tokensToDelete = edits,
      tokenizer = Σᐩ::lexAsKotlin
    )
    original to prompt
  }
//      listOf("val query = seed ? : queries . first ( )" to "query = seed ? : queries . first ( )")

   .filter { !it.second.isSyntacticallyValidKotlin() }.distinct().shuffled()
   // Run repair
   .forEach { (groundTruth, prompt) ->
     println("Original:  $groundTruth\nCorrupted: ${prettyDiffNoFrills(groundTruth, prompt)}")
     val startTime = System.currentTimeMillis()
     parallelRepair(prompt, deck, edits + 1, { joinToString("").isSyntacticallyValidKotlin() }, scoreEdit).postprocess()
       .also {
         //    repairKotlinStatement(prompt).also {
         val gtSeq = groundTruth.tokenizeByWhitespace().joinToString(" ")
         val fullESEC = it.map { listOf(it.resToStr()) + it.equivalenceClass.map { it.resToStr() } }
         val contained = fullESEC.any { gtSeq in it }
         val elapsed = System.currentTimeMillis() - startTime

         it.take(20).apply { println("\nTop $size repairs:\n") }.forEach {
           println("Δ=${it.scoreStr()} repair (${it.elapsed()}): ${prettyDiffNoFrills(prompt, it.resToStr())}")
           //        println("(LATEX) Δ=${levenshtein(prompt, it)} repair: ${latexDiffSingleLOC(prompt, it)}")
         }

         println("\nFound ${it.size} valid repairs in ${elapsed}ms, or roughly " +
           "${(it.size / (elapsed/1000.0)).toString().take(5)} repairs per second.")
         println("Original string was ${if (contained) "#${fullESEC.indexOfFirst { gtSeq in it }}" else "NOT"} in repair proposals!\n")
       }
   }
}

// Clearly, there is a qualitative difference in perception and intent between a idiomatic figure of speech like, "the facts speak for themselves" and a system that is explicitly impersonating a human being. One is simply a linguistic jest or

val projectDir = File(File("").absolutePath)
val allProjectsDir = projectDir.parentFile

fun collectMostCommonKeywords() {
  P_kotlin.counter.rawCounts.getFrequentItems(ErrorType.NO_FALSE_NEGATIVES)
    .map { it.item to it.estimate }.toList()
    .sortedByDescending { it.second }.take(10000)
    .joinToString("\n") { it.first }
    .let { File(keywordFile).writeText(it) }
}

private fun constructScoringFunction(): (List<Σᐩ>) -> Double =
  { P_kotlin.score(listOf("BOS") + it + listOf("EOS")) }

// Get top level directory and all Kotlin files in all subdirectories
fun fetchKotlinExamples() =
  allProjectsDir.also { println("Working directory: $it") }
    .walkTopDown().asSequence()
    .filter { it.extension == "kt" }
    .flatMap { it.readLines() }
    .filter { it.isSyntacticallyValidKotlin() }
    .filter { str -> ignoredKeywords.none { it in str } }
//    .filter { str -> str.lexAsKotlin().filter { it.isNotBlank() }.all { it in allNames } }
//    .filter { it.isCompilableKotlin() }
    .map { it.coarsenAsKotlin() }
    .map { it.trim() }.distinct()
//    .take(10)

fun repairKotlinStatement(
  prompt: Σᐩ,
  clock: TimeMark = TimeSource.Monotonic.markNow()
): List<Σᐩ> =
//  newRepair(prompt, permissiveKotlinCFG)
  repair(
    prompt = prompt,
    cfg = permissiveKotlinCFG,
    coarsen = Σᐩ::coarsenAsKotlin,
    uncoarsen = Σᐩ::uncoarsenAsKotlin,
  //  updateProgress = { println(it) },
    synthesizer = bruteForceKotlinRepair(clock), // Enumerative search
    diagnostic = { println("Δ=${levenshtein(prompt, it) - 1} repair: ${prettyDiffNoFrills(prompt, it)}") },
    filter = { isSyntacticallyValidKotlin() },
  )

private fun bruteForceKotlinRepair(clock: TimeMark): CFG.(List<Σᐩ>) -> Sequence<Σᐩ> =
  { a: List<Σᐩ> ->
    try {
      a.genCandidates(setOf(), commonKotlinKeywords + "ε" - "w" )
      a.solve(this, takeMoreWhile = { clock.elapsedNow().inWholeMilliseconds < TIMEOUT_MS })
        //  .also { println("Solving: ${it.joinToString(" ")}") }
    } catch (e: Exception) { e.printStackTrace(); emptySequence()}
  }

@Language("kt")
val originalKotlinLines = """
    val common = results.map { it.uri }.intersect(results.map { it.uri }.toSet())
    val query = seed ?: queries.first()
    val nearestResults = findNearest(vectorize(query), 100)
    val newEdges = nearestResults.map { query cc it }
    typealias VecIndex = HnswIndex<Concordance, DoubleArray, CodeEmbedding, Double>
    override fun dimensions(): Int = embedding.size
    override fun toString() = loc.getContext(0)
    typealias KWIndex = ConcurrentSuffixTree<Queue<Concordance>>
    fun isInvalid(ch: Char): Boolean = ch.code == 0 || ch.code == 0xfffd
    val list = runSplitOnPunc(if (doLowerCase) { Ascii.toLowerCase(token) } else token)
    val splitTokens: MutableList<String> = ArrayList()
    val outputIds: MutableList<Int?> = ArrayList()
    val outputTokens: MutableList<String> = ArrayList()
    val subTokens: MutableList<String> = ArrayList()
    val knnIndex: VecIndex by lazy { buildOrLoadVecIndex(File(index), URI(path)) }
    val id = query.hashCode().toString()
    val mostSimilarHits = nearestNeighbors.sortedByDist(query, metric)
    val originalIndex = nearestNeighbors.indexOf(s).toString().padStart(3)
    fun main(args: Array<String>) = KNNSearch().main(args)
    val trie: KWIndex by lazy { buildOrLoadKWIndex(File(index), URI(path)) }
    fun main(args: Array<String>) = TrieSearch().main(args)
""".trimIndent()

val protected = 10

val coarsenedKotlinLines = """
  val w = ( w .. w ) . w { w . w ( ) }
  val w = w . w ( w ) . w { ( w , w ) -> w / w } . w ( )
  val w = w ( w ) { ( w .. w ) . w ( ) }
  val w = w { w - w - w }
  val w = w { w - w - w - w - w - w - w - w - w - w - w }
  val w = w { w - w - w - w }
  val w = w . w . w ( )
  val w = w . w ( w ) . w . w ( )
  val w = w { w = w + w }
  fun w . w ( ) : w < * > = w . w . w . w ( )
  val w = w . w { ( w , w ) -> w . w ( ) w w }
  val w = w . w ( ) . w ( )
  val w = ( w . w w w . w ) . w ( w )
  val w = w [ it ] ! ! . w { w -> w . w ( w , w + it . w ) }
  fun < w > w < w > . w ( ) = ( w .. w ) . w { w ( w , it ) }
  val w = w . w < w , w > ( )
  val w = w . w { w [ ( w + it ) % w . w ] }
  val w = it . w { it . w }
  val w = w . w { w ( it ) } . w ( ) . w ( )
  val w = w . w ( w ) . w ( )
  val w = ( w - w . w ( w ) ) . w ( )
  val w = w ( object : w { } , object : w { } , object : w { } ) w
  val w = w ( w , object : w { } )
  interface w : w , w , w , w , w , w , w , w , w { override val w get ( ) = w }
  interface w : w , w , w , w , w , w , w { override val w get ( ) = w }
  interface w : w , w , w , w , w , w { override val w get ( ) = w }
  interface w : w , w , w , w , w { override val w get ( ) = w }
  interface w : w , w , w { override val w get ( ) = w }
  interface w : w , w { override val w get ( ) = w }
  interface w : w { override val w get ( ) = w }
  interface w : w , w , w , w , w , w { override val w get ( ) = w ( ) }
  interface w : w , w , w , w , w { override val w get ( ) = w ( ) }
  interface w : w , w , w , w { override val w get ( ) = w ( ) }
  fun < w : w , w : w , w : w , w : w , w > w ( w : w , w : w , w : w ) : w = w ( )
  fun < w : w , w : w , w : w , w > w ( w : w , w : w ) : w = w ( )
  fun w ( ) { w ( w ) { true } . w ( w = w ) }
  val init = w ( w , w , w , w )
  val w = ( w w w . w ( ) ) . w ( ) w ( w w w . w ( w = w ) )
  val w = w . w { w [ it ] ! ! } . w ( )
  val w = ( w w w ) . w ( ) w ( w w w ) . w ( ) w ( w w w ) . w ( )
  val w = w . w ( ) w w
  val w = w - ( w + w ) * w
  val w = ( ( w w w ) w ( w w w ) ) w ( ( w w w ) w ( w w w ) )
  val w = ( w w w ) w ( w w w ) w ( w w w )
  val w = w ( w w w - w , w w w , w w w , w w w )
  val w = ( w w w ) w ( w w w ) w ( w w w ) w ( w w w w w )
  fun w . w ( w : w ) = w ( ) . w ( w ) . w ( )
  fun w . w ( ) = w . w ( w ) + w . w ( w ) == w . w ( w ) + w
  val w = ( - w .. w ) . w ( )
  val w = ( w .. w ) . w ( )
  val w = ( ( w w w ) w ( w w w ) ) w ( ( w w w ) w w )
  val w = w . w ( ) w w . w . w ( )
  val w = w ( w w w w w )
  val ( w , w ) = w [ w ] w w [ w ]
  val w = ( ( w + w ) + w ) w ( w + ( w + w ) )
  val w = ( w * ( w + w ) ) w ( w * w + w * w )
  val w = w ( w , w ) { w , w -> w ( w + w ) }
  val w = ( w * w * w ) w w
  val w = w . w . w { w -> w . w { w [ it ] ! ! } }
  val w = w . w ( ) . w { it . w ( ) . w }
  val w = w ( w , w ) { w , w -> ( w + w ) . w ( ) }
  val w = w ( w , w ) { w , w -> w [ w ] [ w ] . w ( ) }
  val w by w ( w , w , w ) { w + w + w }
  val w : w < w > = w { w . w ( w ) . w { it . w ( ) } . w ( ) }
  val w : w < w > = w { w . w ( w . w ) . w ( w . w ( ) ) }
  val w = w . w { if ( it in w . w ) it else w }
  val w = if ( ! w ) w . w else w
  val w = w . w { it . w } . w ( )
  val w : w = w . w ( ) . w ( )
  val w = w . w ( ) . w ( w = false )
  val w = w . w ( w , w )
  val w : w = w . w ( w ) . w ( )
  val w : w = w . w ( w . w ( w , w ) ) . w ( )
  val w = ( w . w ( w , w ) w w ) . w ( )
  val w = w . w { w [ it ] ! ! } . w ( ) . w ( w )
  private fun w . w ( ) : w ? = w . w ( w ) . w ( ) ? . w
  val ( w , w ) = w . w ( w ) ! ! w w . w ( w ) ! !
  val ( w , w ) = w . w ( ) w w . w ( )
  val ( w , w , w ) = it . w { it == w } w it . w { it == w } w it . w { it == w }
  val w = w . w ( w ) . w ( ) w
  val w = w . w ( w , w , w = { w ( it ) } ) . w ( )
  val w = w . w ( w , w , w = { w ( it ) } )
  val w = w ( w , w , w = { w ( it ) } , w = w )
  val w = w . w . w ( ) . w ( w )
  val w = w . w ( ) . w ( w )
  val w = ( w w w ) . w ( w )
  val w = w ( w , w ) { w , w -> w ( w == w [ w ] ) }
  val w = w ( w ) { w , w -> w == w [ w ] }
  val w = w ( w ) { w , w -> w [ w ] [ w ] }
  val w = ( ( w * w ) w w ) w w [ w , w ] w w [ w , w ]
  val w : w < w > = w ( w , w ( w * w ) { w ( w . w ( ) ) } )
  val w = w ( w w w - w )
  val w = ( w w w ) . w ( )
  fun w ( ) = w . w ( w ) . w ( w ) . w { w . w ( it ) }
  val w = w ( ) . w ( )
  val w = w . w ( w . w ( ) )
  val w = ( w * w * w . w ) w w
  val w = ( ( w w w ) w ( w w w ) ) w ( w w w )
  val w = w { w - w - w - w - w ; w - w - w }
  val w = w . w { w ( it . w ) . w ( ) }
  val w : w = w . w ( )
  val w : w < w > = w . w ( )
  private fun w ( w : w ) = w ( w . w ( w ) . w )
  val w : w < w > = w ( w , w )
  val w : w < w < w > > = w . w ( )
  val w : w < w . w > = w ( )
  val w = w . w ( ) - w
  override fun w ( ) : w = w * w . w + if ( w ) w else w
  val w = w ! ! . w ( w , w ) . w ( )
  val w : w < w > = w ( )
  var w : w < w > = w ( )
  val w = w < w > ( w )
  val w = w < w > ( w + w - w )
  private val w = w < w > ( ) . w { w ( w ) }
  val w = w < w < w > > ( )
  val w = w < w > ( w . w + w )
  val w : w < w > = w ( w )
  fun w ( w : w < w > ) : w = w ( w . w ( ) )
  val w : w < w < w > > = w ( )
  var w = w . w ( w , w . w )
  override fun w ( ) : w = w . w ( ) + w . w ( )
  private var w : w < w < w > > ? = null
  val w : w < w < w > > = w < w < w > > ( )
  val w : w < w < w > > = w ( w )
  val w : w < w > = w . w
  val w : w < w > = if ( w > w ) w else w
  val w : w < w > = if ( w < w ) w else w
  var w : w < w < w > > = w ( )
  fun w ( ) : w < w < w > > = w ( ) . w { w ! ! . w { w ( it ) } . w ( ) }
  protected var w : w < w > = w ( )
  abstract fun w ( w : w < w > )
  abstract fun w ( w : w < w > , w : w ) : w
  private val w = w < w > ( )
  class w ( val w : w , val w : w < w > )
  fun w ( ) : w = w ( ) . w ( )
  val w = w ( w . w ( ) ) { w < w > ( ) }
  val w : w < w > ? = w ( w )
  val w = ( w . w . w { it . w } ) . w ( )
  fun w ( w : w ) : w = w / w
  val w : w < w > = w ( w ) { w ( w . w , null , - w ) }
  private val w : w < w > = w ( )
  private val w = w ( w * w ) { w < w > ( ) }
  private val w = w ( w * w ) { w }
  private val w : w = w ( w , this )
  private fun w < w > . w ( ) = this . w { it . w ( ) }
  val w = w . w { w ( it ) } ? : w
  private var w : w < w > = w ( )
  var w = w . w ( w , w )
  private val w = w ( w ) { false }
  val w = w ( w [ w -- ] )
  val w = w . w { w ( it ) == w }
  val w = w . w ( w ) . w . w { w [ w ( w [ it + w ] ) ] . w } ? : w
  fun w ( w : w , w : w )
  fun w ( w : w , w : w , w : w )
  fun w ( w : w < w > )
  fun w ( w : w < w > ) : w
  override val w get ( ) = w . w
  override val w get ( ) = w . w . w + w . w . w
  private var w : w < w > ? = null
  private val w = w ( w . w { w ( it ) } . w ( ) , w )
  override fun w ( w : w ) = w ( w ( w ) )
  override fun w ( w : w , w : w ) = w ( w ( w , w ) )
  override fun w ( w : w , w : w , w : w ) = w ( w ( w , w , w ) )
  override fun w ( w : w < w > ) = w ( w . w ( ) )
  override fun w ( w : w < w > ) : w = w ( w . w ( ) )
  override fun w ( ) : w < w > = w ? : w ( )
  val w = w ( w . w . w { w ( it . w . w ( ) ) } )
  val w = w . w { it . w == w } ? : return w
  val w = w ( w ) ? : return null
  val w = w < w < w > > ( w . w )
  fun w ( w : w ) = w * w + w
  fun w ( w : w ) = w * w
  class w ( w : ( w ) -> w ) { val w by w { w ( this ) } }
  class w ( w : ( ) -> w ) { val w by w { w ( ) } }
  fun w ( ) : w < w < w > > = w ( this )
  override fun w ( ) : w < w < w > > = w ( this )
  fun w ( ) : w = if ( w == null ) w else w . w ( )
  val w : w < w > by w { w ( this ) }
  val w : w by w { w + ( if ( ! w ( ) ) w else this . w . w ) }
  val w : w by w { if ( ! w ( ) ) w else this . w . w }
  operator fun get ( w : w ) : w < w > = if ( w == w ) this else w [ w - w ]
  fun w ( ) : w < w > = if ( ! w ( ) ) this else w . w ( ) + w
  val w . w by w { w ( ) . w ( ) }
  val w . w by w { w ( ) . w { it . w } . w ( ) }
  val w . w by w { w . w { w , w -> w w w + w } . w ( ) }
  val w . w : w by w { w . w ( ) }
  val w = w . w { if ( ! w ( ) ) return @ w null }
  val w = w [ w . w ] ! !
  operator fun get ( w : w ) : w ? = w [ w . w ( ) . w ( ) ]
  override fun get ( w : w ) : w ? = w [ w ]
  override fun w ( ) = w . w ( )
  fun w . w ( ) : w = if ( this ) w else w
  fun w . w ( ) : w = w ( w ( this ) )
  object w : w by w ( w ( ) )
  infix fun w . w ( w : w ) = w
  fun w < w > . w ( ) = w { ( it w - it ) } . w ( w as w ) { w , w -> w w w }
  operator fun w . w ( ) : w = w ( )
  fun w ( w : w ) : w = w . w ( ) . w . w ( )
  fun w ( w : w ) : w = w . w ( )
  fun w ( w : w ) : w = w . w { it . w ( ) } . w ( )
  fun w ( w : w , w : ( w ) -> w ) : w = w ( w ) { w ( it ) }
  infix fun w < w , * , * > . w ( w : w < w , * , * > ) : w = ( this w w ) . w ( )
  override fun w ( w : w ? ) = w ( ) == w . w ( )
  fun w ( ) = w ( ) . w ( ) . w ( )
  val w : w < w , w > = w . w . w { ( w , w ) -> w w w }
  interface w { val w : w get ( ) = w }
  interface w { val w : w get ( ) = true }
  class w < out w , out w , out w >
  fun < w , w , w > w ( ) : w < w , w , w > = w ( )
  typealias w < w > = w < w , w >
  typealias w < w > = w < w , w , w >
  typealias w < w > = w < w , w , w , w >
  typealias w < w , w > = w < w , w >
  val < w , w > w < w , w > . w : w get ( ) = w
  typealias w < w , w , w > = w < w , w , w >
  val < w , w , w > w < w , w , w > . w : w get ( ) = w
  fun < w : w , w : w , w > w < w , w > . w ( ) : w < w > = w ( w , w )
  fun < w , w > w ( w : w , w : w ) = w ( w , w )
  fun < w , w , w > w ( w : w , w : w , w : w ) = w ( w , w , w )
  fun < w , w , w , w > w ( w : w , w : w , w : w , w : w ) = w ( w , w , w , w )
  infix fun < w , w , w > w < w , w > . w ( w : w ) = w ( w , w , w )
  infix fun < w , w , w , w > w < w , w , w > . w ( w : w ) = w ( w , w , w , w )
  infix fun w . w ( w : w ) = w . w <= w && w <= w . w
  operator fun < w > w < w > . w ( w : w < w > ) = w ( w )
  public inline fun < w > w < w > . w ( w : ( w ) -> w ) : w < w > = w ( w ( ) , w )
  override fun w ( w : w ? ) = ( w as? w < w , w > ) ? . w == w
  fun < w > w < w < w > > . w ( ) = w { it [ w ] w it [ w ] } . w ( )
  infix fun < w > w . w ( w : w ) = w ( this , w )
  fun < w > w ( w : w ) : w < w , w > = w . w ( w , w )
  fun < w > w ( w : w , w : w ) : w < w , w > = w . w ( w , w , w )
  fun < w > w ( w : w , w : w , w : w ) : w < w , w > = w . w ( w , w , w , w )
  typealias w < w , w , w > = w < w < w , w > , w >
  inline fun < reified w : w < * > > w ( ) = w :: class . w ! ! . w ( w ) . w ( )
  fun < w > w ( w : w , w : w ) : w < w , w , w > = w ( w , w , w , w )
  val < w , w : w < w > , w > w < w , w > . w : w get ( ) = w ( )
  operator fun < w > w < w > . get ( w : w ) = w ( w )
  operator fun < w , w > w < w > . w ( w : w < w > ) : w < w < w > , w < w > > = this w w
  fun w ( ) : w = if ( w == null ) w else if ( w is w < * , * > ) w + w . w ( ) else w
  typealias w < w > = w < w , w < w > >
  typealias w < w , w > = w < w , w < w , w > >
  typealias w < w > = w < w < w > >
  typealias w < w , w > = w < w < w > , w >
  typealias w < w , w > = w < w < w , w > >
  typealias w < w , w , w > = w < w < w , w > , w > w
  fun < w > w ( w : w ) : w < w > = w ( w , null )
  fun < w > w ( w : w , w : w ) : w < w > = w ( w , w ( w , null ) )
  fun < w > w ( w : w , w : w , w : w ) : w < w > = w ( w , w ( w , w ( w , null ) ) )
  fun < w , w : w < w , w > , w > w . w ( ) : w < w > = w ( w , null )
  fun < w , w : w < w , w > , w > w . w ( ) : w < w > = w ( w , w ! ! . w ( ) )
  fun < w , w : w < w , w > , w > w . w ( ) : w < w , w > = w ! !
  fun < w , w : w < w , w > , w > w . w ( ) : w < w , w > = w ( ) . w ( )
  operator fun < w , w > w < w , w > . w ( w : w < w , w > ) : w < w , w > = w ( )
  val w : w < w > get ( ) = w ( this as w )
  abstract fun w ( ) : w < w , * >
  override fun w ( ) = this :: class . w ( ) + w . w ( )
  fun w ( ) : w = w ( w ( ) )
  override fun w ( ) : w < w > = w ( w )
  override fun w ( ) : w = w ( )
  object w : w < w , w > ( null ) { override fun w ( ) = w }
  typealias w < w > = w < w >
  typealias w < w > = w < w < w < w > > >
  typealias w < w > = w < w < w < w < w > > > >
  typealias w < w > = w < w < w < w < w < w > > > > >
  typealias w < w > = w < w < w < w < w < w < w > > > > > >
  typealias w = w < w < w > >
  typealias w = w < w < w < w > > >
  val w : w = w . w . w
  val w get ( ) = w ( this as w )
  open class w < w > ( override val w : w ? = null ) : w < w , w < w > > ( w ) { companion object : w < w > ( ) }
  override fun w ( w : w ? ) = if ( w is w < * > ) if ( w == null && w . w == null ) true else w == w . w else false
  override fun w ( ) = w . w ( ) . w ( )
  object w : w < w > ( null )
  fun w < * > . w ( w : w = w ) : w = ( w as? w < * > ) ? . w ( w + w ) ? : w
  operator fun w . w ( w : w < * > ) : w = w ( ) + w . w ( )
  operator fun w . w ( w : w < * > ) : w = w ( ) - w . w ( )
  operator fun w . w ( w : w < * > ) : w = w ( ) * w . w ( )
  operator fun w . w ( w : w < * > ) : w = w ( ) / w . w ( )
  operator fun w < * > . w ( w : w ) : w = w ( ) + w . w ( )
  operator fun w < * > . w ( w : w ) : w = w ( ) - w . w ( )
  operator fun w < * > . w ( w : w ) : w = w ( ) * w . w ( )
  operator fun w < * > . w ( w : w ) : w = w ( ) / w . w ( )
  fun < w : w < * > , w : w < w > > w . w ( ) : w = w ( this ) as w
  fun < w : w < * > , w : w < w > > w . w ( ) : w = w as w
  fun < w : w < * > , w : w < w > > w . w ( ) : w = w ( ) . w ( )
  fun < w > w < w > . w ( w : w ) : w = w ( w ( w = w . w ( ) ) )
  fun < w > w < w > . w ( w : w < w > ) : w = w . w { w , w -> w + w }
  fun < w > w < w > . w ( w : w < w > ) : w = w . w { w , w -> ( w * w ) }
  val w : w get ( ) = w . w ( )
  operator fun w . w ( w : w ) = w ( this , w )
  infix fun w . w ( w : w ) = w ( this , w )
  override fun w . w ( ) : w = w ( )
  override fun w . w ( ) : w = this + w
  override fun w . w ( w : w ) : w
  val w : w . ( w , w ) -> w
  override fun w . w ( w : w ) = w ( this , w )
  tailrec fun < w > w < w > . w ( w : w , w : w , w : w = w , w : w = w ) : w = w ( )
  operator fun w . w ( w : w ) : w = w ( this , w )
  fun w ( w : ( w ) -> w ) = w ( w . w { w ( it ) } )
  infix fun w . w ( w : w < w > ) : w < w > = w . w { w . w ( it , this ) }
  class w < w , w : w < w > > ( override val w : w ) : w < w , w >
  operator fun w ( w : w ) : w = w ( w + w . w , w + w . w )
  operator fun w ( w : w ) = w ( w * w . w , w * w . w )
  operator fun w ( w : w ) = w ( w * w . w + w . w * w , w * w . w )
  operator fun w ( w : w ) = w ( w * w . w - w . w * w , w * w . w )
  override fun w ( ) = w ( ) . w ( )
  fun w ( w : w , w : w ) = w / w . w ( w ) w w / w . w ( w )
  fun w ( ) = w ( w ( ) )
  fun w ( vararg w : w ) : w = w ( w . w ( ) )
  fun w ( vararg w : w ) : w = w ( w . w { it . w } )
  interface w < w , w , w > : w < w , w , w > , w < w > , w
  operator fun get ( w : ( w ) -> w ) : w < w > = w . w ( w )
  val w : w < w , w , w > get ( ) = w ( w )
  operator fun get ( w : w ) : w = w [ w ]
  val w : w < w > = set . w ( )
  val w : w < w , w > = w . w { w , w -> w w w } . w ( )
  operator fun get ( it : w ) : w = w [ it ]
  operator fun get ( w : w ) : w = w [ w ] ? : - w
  operator fun w ( w : w ) : w = w ( w - w . w )
  var w = w ( w . w , w , w . w { it in w } )
  fun w ( w : w = w . w ) = w ( w , this as w )
  val w = w < w , w > ( )
  fun w ( ) = w ( ) . w ( ) . w ( ) [ w ] . w ( )
  val w = if ( w is w < * , * , * > ) w . w else w . w ( )
  override fun w ( ) = w ( this ) { it . w } . w ( )
  interface w < w , w , w > : w < w , w , w >
  interface w < w , w , w > : w < w , w , w > , w
  val w : ( w ) -> w < w > w
  private fun w < w > . w ( ) : w < w > = w { it . w ( ) } . w ( )
  fun w ( ) : w = w ( w ( w ) . w ( ) )
  abstract class w < w , w , w > : w < w , w , w >
  override val w : w = w . w ( )
  override fun w ( w : w ? ) = ( w as? w ) ? . w { w ( ) == w . w ( ) } ? : false
  override fun w ( w : w ? ) = ( w as? w < * , * , * > ) ? . w { w == it . w } ? : false
  interface w { fun w ( ) : w }
  abstract class w : w < w , w , w >
  typealias w < w > = w < w ? > ? w
  typealias w < w > = w < w > w
  fun w . w ( ) = w ( w ) { this [ it ] }
  fun w < w < w > > . w ( ) = w ( w ( ) )
  fun w ( w : w , w : w , w : w ) = w w
  val w = w ( w , w , w , w , w , w , w , w , w , w )
  val w = w ( w , w , w , w )
  interface w : w < w , w , w >
  fun < w , w , w : w < w > , w : w > w ( w : ( w ) -> w , w : w ) : w = w ( )
  val w = w ( w [ w ] . w ( ) )
  private fun w ( w : w ) = w [ w ] ? : w
  inline fun < w > w < w > . w ( w : w ) = w ( w ) w
  inline fun < w > w < w > . w ( ) = w ( ) ! ! w
  fun < w > w < w > . w ( ) : w = this [ w ] ! !
  open class w ( override val w : w < w > = w ( w ( ) ) )
  override val w : ( w ) -> w < w >
  operator fun w ( w : w ? , w : w < * > ) : w = w ( w . w )
  override val w : ( w : w < w > ) -> w
  override val w : ( w : w , w : w ) -> w
  override val w : ( w : w , w : ( w ) -> w < w > ) -> w
  val w by w ( ) ; val w by w ( ) ; val w by w ( ) ; val w by w ( )
  class w ( val w : w , val w : w )
  operator fun w . w ( w : w ) = w + w ( w , w , w )
  sealed class w < w , w > ( val w : w , val w : w ) { abstract val w : w }
  infix fun < w > w < w , * > . w ( w : w < w , * > ) = w
  infix fun < w , w > w < w , * > . w ( w : w < w , * > ) = w
  infix fun < w > w < w , * > . w ( w : w ) = w
  infix fun w < * , * > . w ( w : w ) = w
  fun < w > w < * , w > . w ( ) = w
  class w < w , w > ( val w : w , val w : w ) : w ( w , w )
  class w < w , w , w > ( val w : w , val w : w , val w : w ) : w ( w , w , w )
  fun < w , w > w < w , w > . w ( ) = w ( w , w )
  fun < w , w , w > w < w , w , w > . w ( ) = w ( w , w , w )
  fun < w , w , w , w > w < w , w , w , w > . w ( ) = w ( w , w , w , w )
  val w = w ( w , w ) . w ( )
  fun w ( w : w , w : w ) : w = w w w
  val w = w ( w , w , w , w , w )
  private val w : w < w , w > = w ( w , w )
  operator fun get ( w : w ) = w [ w ]
  val w : w ? = w . w ( w , w )
  val w = w . w . w ( ) . w ( )
  operator fun w ( w : w ) = w in w
  val w = w { it . w . w ( w ) . w ( ) . w ( ) }
  val w = w . w ( w .. w ) . w ( )
  val w = w . w ( w ) { w , w -> w w ( w w w ) } w w
  private fun w < w > . w ( ) : w < w > = w { w ( it ) }
  class w ( val w : w < w > ) : w < w > by w
  var ( w , w , w ) = w w w w w
  val w = w ( w ) { w }
  val w = ( w + w ) . w ( w ) - w . w ( w )
  val w = w ( ) . w ( w / w ) . w ( )
  val w = w ( w / w , w )
  val w : w = w [ w ] . w
  val w = w ( w [ w ] . w )
  val w = w [ w ] [ w ]
  val w = w + w * ( w + w * w ) + w * w
  val w = w + w * ( ( w - w ) + w * w ) + w * w + w
  var w = this [ w ] . w ( ) w w w w w w
  var w by w ( ) ; var w by w ( ) ; var w by w ( ) ; var w by w ( )
  operator fun w . w ( w : w ) = w ( this , w ) { w , w -> w + w }
  operator fun w . w ( w : w ) = w ( this , w ) { w , w -> w - w }
  operator fun w . w ( w : w ) = w ( this , w ) { w , w -> w * w }
  operator fun w . w ( w : w ) = w ( this , w ) { w , w -> w / w }
  object w : w ( ) , w , w
  fun w ( w : w ) : w = if ( w is w ) w else w ( w . w ( ) )
  fun w ( w : w , w : w , w : ( w , w ) -> w ) : w = w ( w ( w ) , w ( w ) )
  fun w ( vararg w : w ) : w < w > = w . w { w ( it ) } . w ( )
  override fun w ( ) = if ( w == w . w ) w else w . w ( )
  operator fun w ( w : w ) = w ( w . w , this , w ( w ) )
  infix fun w ( w : w ) = w ( w . w , this , w ( w ) )
  operator fun w ( ) = w ( w . w , this )
  fun w ( ) = w ( w . w , this )
  fun w ( w : w ) = w ( w . w , this , w ( w ) )
  class w ( val w : w = w ( ) ) : w ( w )
  val w : ( w < out w > ) -> w = { w ( w ) }
  fun w ( vararg w : w ) = w . w == w . w
  val w by w ( ) ; val w by w ( ) ; val w by w ( )
  val w by w ( ) ; val w by w ( )
  operator fun w . w ( w : w ) : w = this - w ( w )
  operator fun w . w ( w : w ) : w = w ( this ) - w
  operator fun w . w ( w : w ) : w = w ( this ) - w ( w )
  override fun w ( out : w < w > ) = w ( w = w ( ) , out = out )
  fun w ( ) = w ( w . w , w ) { w , w -> this [ w ] . w }
  val ( w , w ) = w . w { it . w }
  val w = w . w ( w . w { it . w } . w ( ) )
  override val w : ( w : w < w < w > > ) -> w < w >
  override val w : ( w : w < w > , w : w < w > ) -> w < w >
  override val w : ( w : w < w > , w : ( w < w > ) -> w < w < w > > ) -> w < w >
  val w : w < w , w > = w ( )
  operator fun w . w ( w : w ) = w * this w
  val w . w : w get ( ) = w . w ( this , this @ w )
  fun w ( w : w ) = w ( w . w , this , w . w ( w ) )
  fun w ( ) : w = w ( ) . w ( ) . w
  abstract class w { override fun w ( ) = this :: class . w ! ! }
  object set : w . w ( ) , w
  object get : w . w ( ) , w
  object w : w . w ( ) , w
  operator fun get ( w : w ) = w ( w . get , w ( ) , w ( ) , w ( w ) )
  operator fun w ( w : w ) = w ( w . w , w ( ) , w ( ) , w ( w ) )
  operator fun w ( w : w ? , w : w < * > ) : w = w ( w . w , w as? w )
  fun w ( ) = ( if ( w != null ) w else w . w . w { it . w != null } ! ! . w ) ! !
  fun w ( ) = w ( ) . w [ this ] ? : this
  class w ( val w : w = w ( ) , override val w : w ) : w ( w , w )
  override val w : w < w < w , w , w > , w > get ( ) = w ( )
  fun w ( w : w ) : w = w ( )
  fun w ( w : w ) = w ( w ) { w , w -> w == w }
  operator fun w ( w : w ) : w = this + - w * w
  fun w ( w : w ) = w ( w ) { w , w -> if ( w == w ) w else w }
  operator fun w . w ( w : w ) : w = w * this
  operator fun w . w ( w : w ) : w = w ( ) * w
  operator fun w . w ( w : w ) : w = w ( ) + w
  operator fun w . w ( w : w ) : w = this - w . w ( )
  operator fun get ( w : w ) = w . w ( w ) { w }
  operator fun set ( w : w , w : w ) { w [ w ] = w }
  val w = w ( w , w , w ) . w ( ) . w
  val w . w : w by w { w ( this ) }
  val w . w : w by w { w { ( w , w ) -> w w w ( w ) } . w ( ) }
  val w . w : w by w { w ( w ) }
  infix fun w . w ( w : w ) = w ( w , w . w )
  val w : w < w > = w . w ( ) . w ( )
  val w by w { w . w { it . w } }
  val w : w < w > by w { w { w } }
  private fun < w > w ( w : w . ( ) -> w < w > ) : w < w > = w . w { it . w ( ) } . w ( )
  operator fun w . w ( w : w ) : w = w in w . w . w
  operator fun w . w ( w : w ) : w = w . w ( w ) != null
  operator fun w . w ( w : w ) : w = w . w { w in it }
  infix fun w . w ( w : w ) : w = w w this
  infix fun w . w ( w : w ) : w = w ( this , w )
  infix fun w . w ( w : w ) : w = w ( * w , w )
  infix fun w . w ( w : w ) : w = w ( this , * w . w )
  infix fun w . w ( w : w ) : w = w ( * w , * w . w )
  val w = w < w , w < w > > ( )
  val w = w . w . w ( this )
  val w = w . w { w [ it ] . w { it [ w ] } . w ( ) }
  val ( w , w ) = w . w . w { it . w . w } ! !
  fun w . w ( w : w ) : w < w > = w ( w )
  fun w . w ( ) = w . w == w && w [ w ] in w
  val w = w { it . w == w } . w { it . w } . w ( )
  val w = w { it . w . w > w } ? : return this
  val w = w w w . w . w ( w )
  val w = w . w w ( w . w . w ( w ) + w )
  val w = this - w + w + w
  val w = w . w { it . w . w { it !in w } } ? : return this
  val w = w . w . w { it !in w }
  val w = w . w . w ( ) . w { it [ w ] = w }
  val w = w w w ( w . w [ w ] )
  val w = this - w + ( w . w w w ) + w
  typealias w = ( w , w < w > ) -> w < w >
  val w . w : w < w > by w { w ( ) }
  val w = ( if ( ! w ) w . w else w ) . w ( )
  val ( w , w ) = it . w ( )
  val w = w . w { w in it . w }
  val ( w , w ) = it w w . w ( )
  val w . w by w { w < w > ( ) }
  val w . w by w { w ( w ) }
  val w = w . w . w { w [ it ] == w } . w ( )
  typealias w = w < w , w < w > >
  val w . w : w get ( ) = w
  fun w . w ( ) : w = w ( this )
  val w . w : w < w > by w { w { it . w } . w ( ) }
  val w . w : w < w > by w { w + w { it . w } }
  val w . w : w < w > by w { w - w }
  val w . w : w < w > by w { w { it . w . w == w } }
  val w . w : w < w > by w { w { it !in w } }
  val w . w : w by w { w ( ) }
  val w . w : w by w { w [ this ] ! ! [ w ] }
  val w . w by w { w < w , w < w > > ( ) }
  val w . w : w < w , w > by w { w ( ) }
  val w : w < w > = w . w . w ( )
  val w : w < w , w > = w . w ( w . w ) . w ( )
  val w = w . w ( { it . w } , { it . w } ) . w { it . w . w ( ) }
  operator fun get ( w : w < w > ) : w < w > = w [ w ] ? : w ( )
  operator fun get ( w : w ) : w < w < w > > = w [ w ] ? : w ( )
  fun w . w ( w : w ) = w ( w ) . w ( )
  fun w . w ( w : w ) = w . w ( ) . w { w ( it ) } . w ( )
  fun w . w ( vararg w : w ) : w < w > = w . w [ w . w ( ) ] ? : w ( )
  fun w . w ( ) = w { w { w -> w . w . w { w -> w . w - w } } }
  fun w . w ( ) = w { it . w ( ) } . w ( )
  fun w . w ( w : w ) : w = w ( w . w ( ) . w ( ) )
  fun w . w ( w : w ) : w = w . w ( w ( ) )
  fun w . w ( w : w ) : w = w . w . w { w ( it ) }
  fun w . w ( w : w ) : w ? = w ( ) . w ( w )
  fun w . w ( w : w ) : w = w ( w . w ( ) ) [ w ] . w ( )
  fun w . w ( w : w ) : w = w ( w . w ( ) )
  fun w . w ( w : w , w : w ) : w = w ( w , w , w )
  fun w . w ( ) : w = w in this
  fun w . w ( w : w ) = w ( ) || w ( w )
  fun w . w ( w : w ) : w = w ( ) && w ( w ) . w ( w ) in w . w
  fun w . w ( w : w ) : w = w . w . w { w ( it ) } . w { it }
  val w . w : w < w > by w { w { it . w ( ) } . w ( w ) }
  fun w . w ( ) : w = w . w ( )
  val init = this * w ( w , w , w ( w ) { w } )
  val w : ( w ) -> w = { it . w { w ( it ) } }
  val w : ( w ) -> w = { it . w ( ) }
  tailrec fun w ( w : w , w : w = w ) : w = if ( w == w ) w else w ( w - w , w * w )
  fun w < w > . w ( ) = w ( w , this [ w ] . w ) { w , w -> this [ w ] [ w ] }
  fun w ( w : w , w : w ) = if ( w == w ) w else w
  infix fun w < w > . w ( w : w ) = w { it w w }
  fun w . w ( ) = w ( )
  fun w . w ( ) = w ( ) [ w ]
  val w = w { it . w ( ) }
  val ( w , w ) = w w w ( w . w . w ( ) / w ) . w ( )
  val ( w , w ) = w [ w , w ] . w ( ) . w { it [ w ] w it [ w ] }
  private fun < w > w < w < w > > . w ( w : w ) = w { it [ w ] }
  val w = w . w { it . w }
  val w = ( w w w ) . w { w . w ( it ) . w { w . w ( it ) ! ! . w . w } }
  fun w ( w : w , w : w ) : w = w ( w . w ( ) , w . w ( ) )
  var w = w ( w . w + w )
  val w = w ( w . w + w )
  val w = w [ w - w ] + w
  val w = w [ w - w ] + if ( w [ w - w ] == w [ w - w ] ) w else w
  val w : w = ( - w ) . w ( ) w w . w ( )
  val w = w { w { it } }
  val w = w { w -> w { w { w { w } } } }
  val w = w ( w , w , w )
  val w = w w w . w ( )
  val w = w w w w w ( w ) w w ( w )
  val w : w < w > = w ( w , w , w , w )
  val w = w ( w ( w , w , w ) , w ( w , w , w ) , w ( w , w , w ) )
  val w = w ( w ( w , w ) , w ( w , w ) , w ( w , w ) )
  val w = w ( w ( w , w ) , w ( w , w ) )
  val w : w < w > = w ( w ( w , w , w ) , w ( w , w , w ) , w ( w , w , w ) )
  val w : w < w > = w ( w ( w , w ) , w ( w , w ) , w ( w , w ) )
  val w : w < w > = w ( w ( w , w ) , w ( w , w ) )
  val w = w . w . w . w
  val w : w < w < w > > = w ( w ( w ) )
  val w : w < w < w > > = w ( w ( w ( w ) ) ) . w { it - w } . w { it + w }
  val w : w < w < w > > = w . w { it - w } . w { it + w }
  val w : w < w < w > > = w . w { it - w } . w { it + w } . w { it + w }
  val w = w ( w ) . w ( )
  val ( w , w ) = w w w
  val w = w . w . w ( ) . w ( w ) . w ( )
  fun w < w > . w ( ) = w { it . w ( ) }
  val w = w ( w ( w , w , w ) )
  val w = w ( w ( w , w , w , w , w , w ) , w ( w , w , w , w , w , w ) )
  val w = w ( w ( w ) , w ( w ) , w ( w ) )
  val w = w { w - w - w - w - w }
  val w = w { w - w - w - w - w - w - w }
  fun w ( ) = w ( w , w . w ( ) )
  val w = w . w ( w , - w , w , - w , - w ) . w ( )
  val ( w , w ) = w . w ( )
  val w = w . w ( ) . w
  val w = w . w ( ) . w ( ) . w
  val w = w ( w , w , w , w , w , w , w , w , w )
  val w = w { w } . w ( w = w )
  val w = w . w . w { it + w . w ( it . w ) }
  val w = w . w { it * it }
  val w = w . w ( w . w , w )
  val w = w ( w ( w * w ) { it } )
  val w = w ( w , w ( w * w ) { it } )
  val ( w , w ) = w ( w ) w w ( w )
  val w = w ( w w w , w )
  val w = w ( w , w ) w w ( w , w )
  val w = w ( w ( w , w ) , w )
  val w = w ( w ( w , w ) , w ) w w ( w , w ( w , w ) )
  fun w ( ) = w . w . w ( ) . w ( )
  val ( w , w ) = w ( w , w ) . w { it w w ( w , it - w ) }
  fun w ( w : w ) = w - w . w ( w - w )
  override val w : w = ( w - w ) . w ( w ) / w
  override val w : w = w ( w , w )
  override fun w ( w : w ) : w = w ( w , w , w )
  override val w : w = w * w / ( ( w + w ) . w ( w ) * ( w + w + w ) )
  operator fun w ( w : w ) : w = w ( )
  infix fun w ( w : w ) : w = w ( )
  override fun w ( w : w ) = w ( w )
  operator fun w ( w : w ) = w ( w = w * w , w = w * w * w )
  operator fun w ( w : w ) = w ( w = w + w , w = w )
  abstract fun w ( w : w ) : w
  override fun w ( w : w ) : w = w . w ( w )
  open fun w ( w : w ) = w . w ( w )
  operator fun w ( w : w < * > ) : w < * > = w ( )
  infix fun w ( w : w < * > ) : w < * > = w ( )
  fun w ( vararg w : w < w , w < w > > ) : w = w ( )
  open val w : w < w < * > >
  override val w : w by w { w { it . w } }
  override val w : w by w { w { it . w * it . w + it . w * it . w } + w * w }
  override fun w ( w : w ) = w { it . w . w ( w ) }
  override val w : w < w < * > >
  override fun w ( w : w ) = w < w > ( w , w )
  val w = w . w ( this @ w . w ( ) , w )
  val w = w ( w , w ( w , w ) )
  val w = ( w .. w ) . w { w . w ( ) w w . w ( ) }
  operator fun get ( w : w ) : w = w [ w ] ! !
  override fun get ( w : w ) : w = w [ w ] ! !
  val w : w by w ( w ) { w . w }
  val w = w . w { w [ it ] } . w ( )
  val w : w < w < w > , w > = w ( )
  val w = w . w ( w [ w ] ) . w ( ) . w ( ) . w ( )
  val w = w . w . w ( w ) . w ( )
  val w : w < w < w > > = w < w < w > > ( w )
  val w : w < w > = w . w { w , w -> w + w }
  private val w = w ( w . w ) w
  private val w = w ( w . w ) { it } w
  val ( w , w ) = w < w > ( ) w w < w > ( )
  fun w < * , * , * > . w ( ) = w ( ) . w ( w ) . w ( )
  fun w < * > . w ( ) : w = ( this as w < * , * , * > ) . w ( )
  fun w < * , * , * > . w ( ) : w = w ( )
  fun w < * , * , * > . w ( ) : w = w . w ( w ) . w ( w . w ( w ( ) ) )
  fun w . w ( ) = w ( w , w ) . w ( )
  fun w . w ( ) = w ( w , w ( ) ) . w ( )
  operator fun w . w ( w : w ) : w = w ( w ) . w ( ) . w ( ) ! !
  val w = if ( w ) w else w
  fun w . w ( ) : w = w { if ( it == w ) w else w } . w ( )
  fun w ( w : w . ( ) -> w ) = w ( )
  fun w ( ) : w = w ( this )
  fun w ( w : w = w ( ) ) : w = w ( this , w . w ( w ) )
  fun w ( w : w ) : w = w ( this , w . w ( w ) )
  fun w ( w : w ) = w ( this , w ( w ) )
  fun w ( w : w ) = w ( this , w ( w . w ( ) ) )
  val w : w < w , w > = w . w { ( w , w ) -> w . w ( ) w w } . w ( )
  val w : w < w ? > = w . w
  operator fun get ( w : w ) : w ? = w [ w . w ( ) ]
  fun w ( w : w ) : w < w > = w ( w . w )
  operator fun w . w ( ) : w = w ( w ( this ) )
  infix fun w . w ( w : w ) : w = w ( w ( this ) , w ( w ) )
  infix fun w . w ( w : w ) : w = w ( w ) . w ( )
  fun w . w ( ) : w = w . w ( w ( this ) )
  infix fun w . w ( w : w ) : w = w . w ( w ( this ) , w ( w ) )
  fun w . w ( w : w ) : w = w ( ) . w ( ) . w ( w ) . w ( )
  private fun w ( w : w . ( ) -> w ) : w = w ( w , w . w ( w . w ( ) ) )
  override val w : w by w { w { false } }
  override val w : w by w { w { true } }
  override fun w . w ( w : w ) : w = w { w w w . w }
  infix fun w ( w : w ) : w = w { w w w }
  fun w ( ) : w = w { w . w ( ) }
  fun w ( ) = w . w ( ) . w ( )
  private fun w ( w : w . ( ) -> w ) = w ( w , w . w ( w . w ( ) ) )
  override val w : w by w { w { w } }
  override fun w . w ( w : w ) : w = w { this @ w w w }
  fun w ( ) = w ( ) . w ( )
  fun w ( w : w ) : w = w . w ( w )
  fun w ( w : w ) : w = w . w { w . w ( it ) as w } . w ( )
  val w = w . w ( w ) . w { w ( this @ w ) ; w ( w ) } . w ( )
  fun w . w ( w : w ) : w = w ( )
  val w = w { w ( w ) ; w ( w ) ; w ( ) }
  val w = if ( w . w ( ) == w . w ) w else w . w
  fun w . w ( ) : w < w , w < w , w > ? > = w ( w ( w ( ) ) . w ( ) )
  infix fun w . w ( w : w ) : w = w . w ( this , w )
  infix fun w . w ( w : w ) : w = w ( w )
  val w . w by w { w . w ( ) }
  fun w . w ( ) : w = w { w ( it ) } . w ( )
  val w . w : w < w , w < w < w > > > by w { w ( ) }
  val ( w , w ) = w . w { it in w }
  val w = w . w ( w ) . w ( w )
  val w = w . w [ w ( w ) ]
  val w = w . w { w . w [ it ] . w } ! !
  val ( w , w ) = w . w { it . w ( w ) } . w ( )
  val w = w . w ( w ) { w , w -> w w w } w w ( w )
  val init = w ( ) . w ( w )
  val w = ( w * w ) + w
  val ( w , w ) = ( it . w ( ) as w ) . w ( w )
  val w = w . w ( w , w ) . w { ! it . w ( w ) } . w ( )
  val w : w < w > = w . w . w { w [ it ] . w ( w = w ) } . w ( )
  val w = w . w ( w ) w w . w
  val ( w , w ) = w ( w )
  val w = w . w . w ( w < w > ( ) ) { w , w -> w + w }
  var ( w , w ) = w . w ( w = w )
  val w = w . w { ( w , w ) -> w in w && w } . w ( )
  fun w . w ( w : w ) : w = w ( w ) . w ( ) . w ( )
  fun w . w ( w : w ) = w . w ( ) . w ( ) [ w . w [ w ] ]
  val w = w [ { it . w == w } ] . w ( )
  fun w ( w : w ) : w < w >
  val ( w , w , w ) = w ( w [ w ] , w [ w ] , w [ w ] )
  val w = w . w { ! it . w ( ) }
  val w = w ( ) . w { w ( ) }
  val w = w ( w . w ( ) , w )
  val w : w = w . w ( w . w ( ) )
  var w : w < w > ? = null
  var w : w = w . w ( )
  val w = w < w ? > ( )
  override fun w ( ) : w < w > = this
  override fun w ( ) : w = w < w . w
  val w = w . w ( w - w , w )
  val w = w ( w , w , w , w . w )
  fun w ( ) : w = w ! ! . w . w
  val w : w get ( ) = w . w + w . w
  val w = w . w . w ( w )
  val w = w . w . w + w
  val w = w + w . w . w
  val w = if ( w . w ) w else w
  private val w : w < w , w >
  fun w ( w : w < w > ) : w < w > = w . w { this [ it ] }
  val w : w = if ( w < w ) w [ w ] else w . w ( )
  val w = w . w ( w . w )
  val w : w < w , w > = w ( w )
  val w = w . w ( w ) ! !
  val w = w . w ( ) . w [ w ]
  override fun w ( ) : w ? = null
  val w = w . w ( w :: class . w )
  val ( w , w ) = w . w ( w ) . w { it w w ( it ) }
  val w = w ( ) . w < w > ( it )
  val w by w { w . w { it . w ( ) } }
  val w = w ( w , w , null , null , null )
  val w : w = w ( w . w , w )
  val w : w < w > = w ( w , w , w )
  val w = w ( ) . w ( w . w , w )
  val ( w , w ) = w ( )
  val w = w . w { it . w ( w = w ) } . w ( )
  val w = w ( w , w . w { it . w . w ( ) } )
  val w = { w . w ( ) . w < w }
  val ( w , w ) = w . w ( ) . w { w , it -> w w it }
  val w = w ( ) . w < w < w , w < w , w > > > ( w )
  val w : w < w , w < w > > = w ( )
  val w = if ( w == w ) w else w
  val w = w . w ( w + w . w , w = w )
  fun w . w ( ) = ( w ( ( this + w ) . w ( ) / w ) * w ) . w ( )
  typealias w = w < w , w < w , w > >
  val w : w = w . w { ( w w w ) }
  fun w . w ( ) = w ( ) . w ( )
  val w = w . w ( ) . w { it !in w } . w ( )
  fun w . w ( ) = w == w && this in w
  val w : w < w , w > = w . w ( ) . w ( w ) . w ( )
  val w = ( w + w ) . w ( )
  val w = w . w { ( w . w ( it ) ) }
  val ( w , w ) = w w w ( w )
  fun w . w ( ) = w ( ) . w { w ( it ) }
  val w = w { ! it . w ( ) } . w . w
  val w = w . w ( ) / w . w ( )
  override fun w ( ) = w . w ( ) + w . w . w ( )
  fun w ( ) = w ( w ( w , w ) )
  val w = w ( :: w , :: w )
  val w = w < w < w , w > > ( )
  val w : w < w , w < w > > = w . w ( ) . w ( w ) . w ( )
  val ( w , w ) = w ( w , w )
  val w = w . w { w . w ( it ) }
  val w = w [ w ] . w ( )
  val w = w ( it ) . w ( )
  val ( w , w ) = w . w { ( w , w ) -> w . w w w . w }
  val ( w , w ) = w . w . w w w . w . w
  val w = it . w ( ) . w ( )
  fun w . w ( ) : w = w ( ) . w
  fun w ( w : w ) = w ( w ) . w ( )
  val w = w . w { it . w . w ( ) } + w
  val w = ( w . w ( ) + w . w ( ) ) . w { w , w -> w w w } . w ( )
  val w : w < w > = w . w { w : w -> w ( w , w ) }
  fun w . w ( ) = w ( ) . w { w -> w { it / w } . w ( ) }
  val w = w . w * w . w
  val ( w , w ) = w ( w ( w , w ) , w ( w , w ) )
  val w = w . w ( w ( ) ) . w { w . w ( it ) }
  val w = w ( w ( w ) )
  fun w . w ( w : w ) : w = w ( w ) . w . w ( w )
  fun w ( w : w ) = w ( w ) . w ( w )
  fun w . w ( w : w , w : w = w ) = w ( w ( w ) , w )
  val w = w . w { it . w } . w ( w . w { it . w } . w ( ) )
  val w = w ( w ( w ) , w )
  val w = w . w { w w it }
  typealias w = w < w , w , w , w >
  override fun w ( ) : w = w . w
  override fun w ( ) = w . w ( w )
  fun w ( w : w ) : w = w . w == w || w . w == w
  val w = w ( if ( w ) { w . w ( w ) } else w )
  val w : w < w ? > = w ( )
  val w : w by w { w ( w ( w ) , w ( w ) ) }
  val ( w , w ) = w ( ) . w { it w it :: class . w }
  val w = w . w ( w ) . w ( ) . w ( w )
  fun w ( w : w < w > ) = w ( ) . w ( w )
  val w = w . w ( w ) ? : return @ w
  val w = w . w ( w ) . w { it . w [ w ] ! ! . w } . w ( )
  override fun w ( w : w ? ) = ( w as? w ) ? . w == w
  val w = w ? : w . w ( )
""".trimIndent()

val commonKotlinKeywords: Set<Σᐩ> = coarsenedKotlinLines
  .tokenizeByWhitespace().filterNot { it.isBlank() }
  .distinct().toSet()

val permissiveKotlinCFG = """
  START -> START START
  START -> ${commonKotlinKeywords.joinToString(" | ") { it } }
""".parseCFG().apply { blocked.add("w") }


val ignoredKeywords =
  setOf("import", "package", "//", "/*", "\"", "\'", "\\`", "data", "_")


val allBuiltinTypes = setOf(
  "Any", "Boolean", "Byte", "Char", "Double", "Float", "Int", "Long", "Nothing", "Short", "String",
  "Unit", "Array", "BooleanArray", "ByteArray", "CharArray", "DoubleArray", "FloatArray",
  "IntArray", "LongArray", "ShortArray", "List", "Map", "MutableList", "MutableMap", "MutableSet",
  "Set", "Sequence", "StringBuffer", "StringBuilder", "Triple", "Pair", "Exception", "Throwable",
  "Regex", "RegexOption", "MatchGroup", "MatchGroupCollection", "MatchResult", "MatchResult.Destructured",
)

val allBuiltinNames = setOf(
  "println", "print", "readLine", "readText", "mutableMapOf", "mapOf", "mutableListOf", "listOf",
  "mutableSetOf", "setOf", "arrayOf", "arrayOfNulls", "sequenceOf", "emptySequence", "emptyList",
  "listOfNotNull", "emptyMap", "mapOfNotNull", "emptySet", "setOfNotNull", "error", "require",
  "requireNotNull", "check", "checkNotNull", "assert", "assertNotNull", "generateSequence",
)

val allNames = officialKotlinKeywords + allBuiltinNames + allBuiltinTypes + commonKotlinKeywords