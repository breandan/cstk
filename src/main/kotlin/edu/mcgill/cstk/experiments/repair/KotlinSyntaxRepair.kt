package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.*
import edu.mcgill.cstk.utils.*
import org.jetbrains.kotlin.spec.grammar.tools.*
import java.io.File

/*
./gradlew kotlinSyntaxRepair
 */

fun main() {
  """StringUtils.splitByCharacterTypeCamelCase(token).joinToString(" ") { old ->""".trimIndent()
      .lexAsKotlin().joinToString(" ").let { println(it) }
  """listOf(getDirectHyponyms(sense), getDirectHypernyms(sense))) &&&"""
    .lexAsKotlin().joinToString(" ").let { println(it) }
//  fetchKotlinExamples()
  coarsenedKotlinLines.lines().forEach { if (kotlinCFG.parse(it) != null) println(it) }
}

// Get top level directory and all Kotlin files in all subdirectories
fun fetchKotlinExamples() =
  File(File("").absolutePath)
  .walkTopDown().filter { it.extension == "kt" }
  .flatMap { it.readLines() }
  .filter { it.isValidKotlin() }
  .map { it.coarsenAsKotlin() }.toList()
  .filter { str -> dropKeywords.none { it in str } && str.split(" ").size in 10..40 }
  .distinct()
  .forEach { println(it) }

val kotlinCFG = """
  START -> statement

  statement -> declaration
  statement -> expression
  statement -> control_flow
  statement -> function_call
  statement -> assignment
  statement -> return_statement

  declaration -> val_declaration | var_declaration | function_declaration

  val_declaration -> val w : type | val w : type = expression | val w = expression
  var_declaration -> var w : type | var w : type = expression | var w = expression

  type -> basic_type | nullable_type
  basic_type -> Int | Float | Double | String | Boolean | Char | Long | Short | Any | Unit
  nullable_type -> basic_type ?

  function_declaration -> fun w ( parameters ) : type = expression
  parameters -> parameter | parameter , parameters
  parameter -> w : type

  expression -> binary_expr | unary_expr | paren_expr | w | literal | lambda_expr
  binary_expr -> expression binary_op expression
  binary_op -> + | - | * | / | % | && | || | == | != | > | < | >= | <= | ?: | . | ?. | ?:? | .? | ..
  unary_expr -> unary_op expression
  unary_op -> ! | -

  paren_expr -> ( expression )

  literal -> int_literal | float_literal | double_literal | string_literal | boolean_literal | char_literal
  int_literal -> int
  float_literal -> float
  double_literal -> double
  string_literal -> string
  boolean_literal -> true | false
  char_literal -> char

  lambda_expr -> { lambda_parameters -> lambda_body }
  lambda_parameters -> w | w , lambda_parameters
  lambda_body -> expression

  control_flow -> if_expr | when_expr
  if_expr -> if paren_expr expression | if paren_expr expression else expression
  when_expr -> when paren_expr { when_cases }
  when_cases -> when_case | when_case when_cases
  when_case -> expression -> expression | else -> expression

  function_call -> w ( function_args )
  function_args -> expression | expression , function_args

  assignment -> w assignment_op expression
  assignment_op -> = | += | -= | *= | /= | %=

  return_statement -> return expression
""".trimIndent().parseCFG()

val dropKeywords = setOf("import", "package", "//", "\"", "data", "_")

val kotlinKeywords = setOf(
  "as", "as?", "break", "class", "continue", "do", "else", "false", "for", "fun", "if", "in",
  "!in", "interface", "is", "!is", "null", "object", "package", "return", "super", "this",
  "throw", "true", "try", "typealias", "val", "var", "when", "while", "by", "catch", "constructor",
  "delegate", "dynamic", "field", "file", "finally", "get", "import", "init", "param", "property",
  "receiver", "set", "setparam", "where", "actual", "abstract", "annotation", "companion",
  "const", "crossinline", "data", "enum", "expect", "external", "final", "infix", "inline",
  "inner", "internal", "lateinit", "noinline", "open", "operator", "out", "override", "private",
  "protected", "public", "reified", "sealed", "suspend", "tailrec", "vararg", "field", "it"
)

fun String.coarsenAsKotlin(): String =
  lexAsKotlin().joinToString(" ") {
    if (it in kotlinKeywords || it.none { it.isLetterOrDigit() }) it else "w"
  }.trim()

fun String.isValidKotlin(): Boolean =
  try { parseKotlinCode(tokenizeKotlinCode(this)).let { true } }
  catch (_: Throwable) { false }

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
  val w = w < w && w ! ! [ w ] == '-'
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
  class w : w < w , w , w > { /*...*/ }
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
  val w = w ( '|' , '*' , '^' )
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
  val w = ( w ( w , w ) + w ( w , w ) + w ( w , w ) + '+' + '/' ) . w ( )
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
  fun w . w ( ) = w ( ) == '<' && w ( ) == '>'
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
  val w = w ( '(' , '{' , '[' )
  val w = w ( ')' , '}' , ']' )
  fun w . w ( ) : w = w ( ) . w
  fun w ( w : w ) = w ( w ) . w ( )
  val w = w . w { it . w . w ( ) } + w
  fun w . w ( w : w ) = w ( w + w - w ( ) - w , ' ' )
  val w = ( w . w ( ) + w . w ( ) ) . w { w , w -> w w w } . w ( )
  val w : w < w > = w . w { w : w -> w ( w , w ) }
  fun w . w ( ) = w ( ) . w { w -> w { it / w } . w ( ) }
  val w = w . w * w . w
  val ( w , w ) = w ( w ( w , w ) , w ( w , w ) )
  val w = w . w ( w ( ) ) . w { w . w ( it ) }
  val w = w ( w ( w ) )
  fun w . w ( ) = w ( ) . w ( '.' )
  fun w . w ( ) = w ( ) . w ( '/' )
  fun w . w ( w : w ) : w = w ( w ) . w . w ( w )
  fun w ( w : w ) = w ( w ) . w ( w )
  fun w ( ) = w ( ) . w ( ':' )
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