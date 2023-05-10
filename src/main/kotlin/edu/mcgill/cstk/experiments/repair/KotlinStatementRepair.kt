package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import bijectiveRepair
import edu.mcgill.cstk.utils.*
import org.intellij.lang.annotations.Language
import org.jetbrains.kotlin.spec.grammar.tools.*
import java.io.File
import kotlin.time.*

/*
./gradlew kotlinStatementRepair 2>&1 | grep -v "Parser error:"
 */

@OptIn(ExperimentalTime::class)
fun main() {
//  fetchKotlinExamples()
  MAX_SAMPLE = 200
  originalKotlinLines.lines().map {
    val original = it.lexAsKotlin().joinToString(" ").trim()
    val prompt = original.constructPromptByDeletingRandomSyntax(
      eligibleTokensForDeletion = officialKotlinKeywords,
      tokensToDelete = 1,
      tokenizer = Σᐩ::lexAsKotlin
    )
    original to prompt
  }
  .filter { !it.second.isValidKotlin() }
  .forEach { (original, prompt) ->
    println("Original:  $original\nCorrupted: ${prettyDiffNoFrills(original, prompt)}")
    val startTime = System.currentTimeMillis()
    parallelRepairKotlinStatement(prompt).also {
//    repairKotlinStatement(prompt).also {
      val contained = original in it
      val elapsed = System.currentTimeMillis() - startTime

      it.take(100).forEach {
        println("Δ=${levenshtein(prompt, it) - 1} repair: ${prettyDiffNoFrills(prompt, it)}")
      }

      println("Found ${it.size} valid repairs in ${elapsed}ms, or roughly ${it.size / (elapsed/1000.0)} repairs per second.")
      println("Original string was ${if (contained) "#${it.indexOf(original)}" else "NOT"} in repair proposals!\n")
    }
  }
}

// Get top level directory and all Kotlin files in all subdirectories
fun fetchKotlinExamples() =
  File(File("").absolutePath)
  .walkTopDown().filter { it.extension == "kt" }
  .flatMap { it.readLines() }
  .filter { it.isValidKotlin() } .toList()
  .filter {
    it.coarsenAsKotlin().let { str ->
      dropKeywords.none { it in str } && str.split(" ").size in 10..40
    }
  }.map { it.trim() }.distinct().forEach { println(it) }

fun Σᐩ.coarsenAsKotlin(): Σᐩ =
  lexAsKotlin().joinToString(" ") {
    when {
      it.isBracket() -> it
      it.none { it.isLetterOrDigit() } -> it
      it in officialKotlinKeywords -> it
      else -> "w"
    }
  }

fun Σᐩ.uncoarsenAsKotlin(prompt: Σᐩ): Σᐩ {
  val words = prompt.tokenizeByWhitespace()
    .filter { it !in officialKotlinKeywords && it.any { it.isLetterOrDigit() } }.toMutableList()
  val uncoarsed = tokenizeByWhitespace().joinToString(" ") { token ->
    when {
      token.isBracket() -> token
      token.none { it.isLetterOrDigit() } -> token
      token == "w" -> words.removeFirst()
      token in officialKotlinKeywords -> token
      else -> throw Exception("Unknown token: $token")
    }
  } + words.joinToString(" ", " ")

//  println("After uncoarsening: $uncoarsed")
  return uncoarsed
}

@OptIn(ExperimentalTime::class)
fun parallelRepairKotlinStatement(
  prompt: Σᐩ,
  clock: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow()
): List<Σᐩ> =
  bijectiveRepair(
    prompt = prompt,
    fillers = commonKotlinKeywords + "ε" - "w",
//    coarsen = Σᐩ::coarsenAsKotlin,
//    uncoarsen = Σᐩ::uncoarsenAsKotlin,
    takeMoreWhile = { clock.elapsedNow().inWholeMilliseconds < TIMEOUT_MS },
    //  updateProgress = { println(it) },
    filter = { isValidKotlin() },
//    diagnostic = { println("Δ=${levenshtein(prompt, it) - 1} repair: ${prettyDiffNoFrills(prompt, it)}") },
  )

@OptIn(ExperimentalTime::class)
fun repairKotlinStatement(
  prompt: Σᐩ,
  clock: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow()
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
    filter = { isValidKotlin() },
  )

@OptIn(ExperimentalTime::class)
private fun bruteForceKotlinRepair(clock: TimeSource.Monotonic.ValueTimeMark): CFG.(List<Σᐩ>) -> Sequence<Σᐩ> =
  { a: List<Σᐩ> ->
    try {
      a.genCandidates(setOf(), commonKotlinKeywords + "ε" - "w" )
      a.solve(this, takeMoreWhile = { clock.elapsedNow().inWholeMilliseconds < TIMEOUT_MS })
        //  .also { println("Solving: ${it.joinToString(" ")}") }
    } catch (e: Exception) { e.printStackTrace(); emptySequence()}
  }

val dropKeywords =
  setOf("import", "package", "//", "/*", "\"", "\'", "data", "_")

val officialKotlinKeywords = setOf(
  "as", "as?", "break", "class", "continue", "do", "else", "false", "for", "fun", "if", "in",
  "!in", "interface", "is", "!is", "null", "object", "package", "return", "super", "this",
  "throw", "true", "try", "typealias", "val", "var", "when", "while", "by", "catch", "constructor",
  "delegate", "dynamic", "field", "file", "finally", "get", "import", "init", "param", "property",
  "receiver", "set", "setparam", "where", "actual", "abstract", "annotation", "companion",
  "const", "crossinline", "data", "enum", "expect", "external", "final", "infix", "inline",
  "inner", "internal", "lateinit", "noinline", "open", "operator", "out", "override", "private",
  "protected", "public", "reified", "sealed", "suspend", "tailrec", "vararg", "field", "it"
)

fun Σᐩ.isValidKotlin(): Boolean =
  try { parseKotlinCode(tokenizeKotlinCode(this)).let { true } }
  catch (_: Throwable) { false }

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