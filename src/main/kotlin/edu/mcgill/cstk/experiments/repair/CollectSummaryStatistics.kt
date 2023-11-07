package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.tokenizeByWhitespace
import edu.mcgill.cstk.utils.*

/*
./gradlew collectSummaryStats
 */
fun main() {
//  stackOverflowSnips().computeLengthDistributionStats()
//  stackOverflowSnips().computeRawTokenFrequencies()
//  seq2ParseSnips().computeBigramFrequencies()
//  computeErrorSizeFreq()
//  computePatchStats()
  computePatchTrigramStats()
}

/*
Percentage of snippets with lexical token length <= n

| n  | StackOverflow | Seq2Parse |
|:--:|:-------------:|:---------:|
| 10 |      9%       |    29%    |
| 20 |      16%      |    47%    |
| 30 |      25%      |    62%    |
| 40 |      34%      |    73%    |
| 50 |      41%      |    79%    |
 */

// Returns the frequency of tokenized snippet lengths for buckets of size 10
fun Sequence<List<String>>.computeLengthDistributionStats(
  brokeSnippets: Sequence<String> = readContents("parse_errors.json"),
) =
    map { it.size }.take(10000).groupBy { it / 10 }.mapValues { it.value.size }
    .toList().sortedBy { it.first }.map { it.first * 10 to it.second }
    .runningFold(0 to 0) { (_, prevCount), (n, count) -> n to (prevCount + count) }
    .let { it.map { (n, count) -> n to count.toDouble() / it.last().second } }
    .joinToString("\n") { "<=${it.first}, ${it.second}" }.also { println(it) }

fun Sequence<List<String>>.computeRawTokenFrequencies() =
  take(10000).flatten().groupingBy { it }.eachCount()
    // Now normalize by total number of tokens
    .let { val total = it.values.sum(); it.mapValues { (_, count) -> count.toDouble() / total } }
    // Use precision of 4 decimal places
    .mapValues { (_, count) -> "%.4f".format(count) }
    .toList().sortedByDescending { it.second }
    .joinToString("\n") { "${it.first}, ${it.second}" }.also { println("Token, Frequency\n$it") }

/*
Top Bigrams:

Bigram, Frequency
"_NAME_ (", 0.0551
") _NEWLINE_", 0.0424
"( _NAME_", 0.0330
"_NEWLINE_ _DEDENT_", 0.0321
"_NEWLINE_ _INDENT_", 0.0258
"_NEWLINE_ _NAME_", 0.0255
": _NEWLINE_", 0.0238
"_NAME_ )", 0.0237
"_NAME_ =", 0.0232
"_NEWLINE_ _NEWLINE_", 0.0224
"Colon Newline", 0.0197
"_INDENT_ _NAME_", 0.0190
"Newline Indent", 0.0167
"( _STRING_", 0.0149
"= _NAME_", 0.0146
"_NEWLINE_ _ENDMARKER_", 0.0146
"_STRING_ )", 0.0136
"_DEDENT_ _DEDENT_", 0.0128
"_NUMBER_ ,", 0.0126
", _NUMBER_", 0.0116
"_NAME_ _NEWLINE_", 0.0113
"_DEDENT_ _NAME_", 0.0112
", _NAME_", 0.0102
"_NUMBER_ _NEWLINE_", 0.0101
"_NUMBER_ )", 0.0097
"_ENDMARKER_ Stmts_Or_Newlines", 0.0095
". _NAME_", 0.0092
"Indent Stmts_Or_Newlines", 0.0090
"_DEDENT_ _ENDMARKER_", 0.0085
"_NAME_ .", 0.0084
"Def_Keyword Simple_Name", 0.0081
 */

fun Sequence<List<String>>.computeBigramFrequencies() =
    take(10000).flatten().zipWithNext().groupingBy { it }.eachCount()
    // Now normalize by total number of tokens
    .let { val total = it.values.sum(); it.mapValues { (_, count) -> count.toDouble() / total } }
    // Use precision of 4 decimal places
    .mapValues { (_, count) -> "%.4f".format(count) }
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "\"${it.first.first} ${it.first.second}\", ${it.second}" }.also { println("Bigram, Frequency\n$it") }

fun stackOverflowSnips() =
  readContents("parse_errors.json").map { it.lexToStrTypesAsPython() }

fun seq2ParseSnips() =
  brokenPythonSnippets.map { it.substringBefore("<||>").tokenizeByWhitespace() }

/*
Number of edits, Frequency
0, 0.0
1, 0.512
2, 0.834
3, 1.0
 */
fun computeErrorSizeFreq() =
  preprocessStackOverflow().map { (broke, humfix, minfix) ->
    val brokeLex = broke.lexToStrTypesAsPython()
    val minfixLex = minfix.lexToStrTypesAsPython()
    val minpatch = extractPatch(brokeLex, minfixLex)
//    println(prettyDiffs(listOf(brokeLex.joinToString(" "), minfixLex.joinToString(" ")), listOf("broken", "minimized fix")))
    minpatch.changes().size
  }.take(1000).groupBy { it }.mapValues { it.value.size }
    .toList().sortedBy { it.first }.map { it.first to it.second }
    .runningFold(0 to 0) { (_, prevCount), (n, count) -> n to (prevCount + count) }
    .let { it.map { (n, count) -> n to count.toDouble() / it.last().second } }
    .joinToString("\n") { "${it.first}, ${it.second}" }
      .also { println("Number of edits, Frequency\n$it") }

/*
Insertion, Frequency, Deletion     , Frequency, Substitution            , Frequency
')'      , 98       , NAME         , 219      , (UNKNOWN_CHAR -> STRING), 53
','      , 87       , UNKNOWN_CHAR , 160      , (NAME -> ',')           , 22
98       , 59       , '.'          , 50       , (':' -> ',')            , 12
99       , 58       , NUMBER       , 41       , (NEWLINE -> ':')        , 12
'}'      , 46       , '>'          , 30       , ('=' -> ':')            , 11
']'      , 36       , NEWLINE      , 28       , (STRING -> ',')         , 8
':'      , 35       , ')'          , 26       , ('.' -> ',')            , 8
'('      , 33       , '**'         , 25       , (NAME -> STRING)        , 8
NEWLINE  , 28       , ':'          , 23       , ('pass' -> NAME)        , 8
NAME     , 19       , '>>'         , 20       , ('=' -> '==')           , 7
STRING   , 17       , STRING       , 19       , ('class' -> NAME)       , 6
'{'      , 13       , 98           , 13       , ('[' -> '{')            , 6
'='      , 12       , '...'        , 12       , ('break' -> NAME)       , 6
'['      , 10       , '('          , 10       , (UNKNOWN_CHAR -> ',')   , 6
'class'  , 8        , '{'          , 10       , (',' -> ':')            , 6
'def'    , 7        , 99           , 10       , ('(' -> '[')            , 6
'import' , 6        , '['          , 9        , (']' -> '}')            , 5
'.'      , 6        , ']'          , 7        , (UNKNOWN_CHAR -> '.')   , 4
'from'   , 5        , '*'          , 6        , (NAME -> ']')           , 4
'return' , 3        , '/'          , 5        , ('...' -> ',')          , 4
'in'     , 2        , '//'         , 5        , ('in' -> NAME)          , 4
'...'    , 2        , ','          , 4        , ('<' -> '&')            , 3
'except' , 2        , '}'          , 4        , ('>' -> ';')            , 3
*/

fun computePatchStats() =
  preprocessStackOverflowInParallel(take = 100_000).map { (broke, _, minfix) ->
    val brokeLexed = broke.lexToStrTypesAsPython()
    val minfixLexed = minfix.lexToStrTypesAsPython()
    val patch = extractPatch(brokeLexed, minfixLexed)
    patch.changes().map {
      if (patch[it].old == "") "INS, ${patch[it].new}"
      else if (patch[it].new == "") "DEL, ${patch[it].old}"
      else "SUB, ${patch[it].old} -> ${patch[it].new}"
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "${it.first}, ${it.second}" }
    .also { println("Type, Edit, Frequency\n$it") }

/*
Type, Edit, Frequency
INS, ')' [')'] END, 456
INS, NAME ['('] NAME, 455
INS, NAME [')'] END, 196
INS, NAME ['('] STRING, 174
DEL, NAME [NAME] NEWLINE, 150
DEL, START ['>>'] '>', 142
INS, NAME [')'] NEWLINE, 130
DEL, NAME [NAME] END, 120
DEL, ')' [UNKNOWN_CHAR] END, 109
DEL, '>>' ['>'] NAME, 101
INS, ']' [')'] END, 95
INS, NEWLINE [98] NAME, 91
DEL, NAME [NAME] NAME, 87
INS, ')' [')'] NEWLINE, 85
INS,  [NEWLINE] NAME, 82
DEL, NAME [UNKNOWN_CHAR] ']', 80
INS, NAME ['='] NAME, 80
INS, 99 [99] END, 79
DEL, NAME [NAME] ':', 78
INS, ')' [':'] NEWLINE, 77
INS, STRING [')'] END, 76
INS, ':' [NAME] END, 74
DEL, ')' [')'] END, 69
DEL, NAME [NAME] '.', 64
DEL, NAME [UNKNOWN_CHAR] ')', 63
INS, STRING [')'] NEWLINE, 60
DEL, START [UNKNOWN_CHAR] NAME, 60
SUB, '(' [UNKNOWN_CHAR -> STRING] NAME, 57
INS, ')' [')'] , 56
INS, NUMBER [','] NUMBER, 55
SUB, 98 ['pass' -> NAME] NEWLINE, 55
SUB, ',' [UNKNOWN_CHAR -> STRING] NAME, 53
INS, NEWLINE [99] NAME, 53
DEL, NAME [NAME] '(', 51
DEL, UNKNOWN_CHAR [NAME] ']', 51
DEL, UNKNOWN_CHAR [NAME] ')', 47
INS, ']' [')'] NEWLINE, 47
INS, START ['import'] NAME, 45
INS, '}' ['}'] END, 45
DEL, STRING [NAME] UNKNOWN_CHAR, 42
INS, ')' [NEWLINE] NAME, 42
INS,  [')'] END, 42
INS, NEWLINE [98] 'def', 41
DEL, ']' [UNKNOWN_CHAR] END, 39
INS, NAME [','] NAME, 39
DEL, STRING [NAME] STRING, 38
SUB, STRING [NAME -> ','] UNKNOWN_CHAR, 37
DEL, NEWLINE [98] NAME, 36
INS, ')' [']'] END, 36
DEL, NAME [UNKNOWN_CHAR] END, 35
INS, NAME ['.'] NAME, 34
INS, ']' [']'] END, 34
DEL, NAME [NAME] '=', 34
DEL, ')' ['.'] END, 33
SUB, '=' [UNKNOWN_CHAR -> STRING] NAME, 33
DEL, NAME [':'] NEWLINE, 33
DEL, NAME ['('] NAME, 33
DEL, 99 [99] END, 32
INS, ']' ['}'] END, 32
SUB, '[' [UNKNOWN_CHAR -> STRING] NAME, 32
INS, STRING [','] NAME, 31
INS, STRING [','] STRING, 31
INS, STRING [STRING] NEWLINE, 29
INS, NAME ['in'] NAME, 29
INS, ',' [']'] END, 28
INS, START ['def'] NAME, 27
INS, START ['{'] STRING, 27
INS, NAME [':'] NEWLINE, 27
DEL, START ['**'] NAME, 27
INS, NEWLINE [98] 'if', 27
DEL, ']' ['.'] END, 26
INS, NUMBER [','] STRING, 26
INS, ',' ['}'] END, 26
DEL, ')' [':'] END, 26
DEL, UNKNOWN_CHAR [NAME] UNKNOWN_CHAR, 25
INS, STRING [']'] ')', 25
INS, NAME ['('] , 25
INS, ':' ['('] , 25
DEL, NAME [NAME] ',', 25
INS, ')' [')'] 'for', 25
INS, START ['from'] NAME, 24
INS, START ['class'] NAME, 24
INS, STRING ['}'] END, 23
DEL, ')' ['**'] END, 23
INS, NAME [','] STRING, 23
SUB, STRING ['=' -> ':'] STRING, 23
INS, '}' [')'] END, 23
INS, ':' ['...'] END, 23
DEL, NAME ['.'] END, 22
DEL, NAME [NAME] ')', 22
DEL, START [NEWLINE] 98, 22
DEL, NAME ['>'] NEWLINE, 21
DEL, NAME [UNKNOWN_CHAR] ',', 21
INS, ']' [']'] ')', 21
DEL, UNKNOWN_CHAR [NAME] '.', 21
INS, NAME ['('] '[', 21
DEL, ',' [NEWLINE] STRING, 20
INS, NEWLINE [98] 'for', 20
SUB, 98 ['break' -> NAME] NEWLINE, 20
DEL, ')' [')'] NEWLINE, 20
*/

fun computePatchTrigramStats() =
  preprocessStackOverflowInParallel(take = 100_000).map { (broke, _, minfix) ->
    val brokeLexed = listOf("START") + broke.lexToStrTypesAsPython() + listOf("END")
    val minfixLexed = listOf("START") + minfix.lexToStrTypesAsPython() + listOf("END")
    val patch = extractPatch(brokeLexed, minfixLexed)
    patch.changes().map {
      (
        if (patch[it].old == "") "INS, ${patch[it-1].old} [${patch[it].new}] ${patch[it+1].old}"
        else if (patch[it].new == "") "DEL, ${patch[it-1].old} [${patch[it].old}] ${patch[it+1].old}"
        else "SUB, ${patch[it-1].old} [${patch[it].old} -> ${patch[it].new}] ${patch[it+1].old}"
      )
    }
  }.toList().flatten().groupingBy { it }.eachCount()
    .toList().sortedByDescending { it.second }.take(100)
    .joinToString("\n") { "${it.first}, ${it.second}" }
    .also { println("Type, Edit, Frequency\n$it") }