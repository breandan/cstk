package edu.mcgill.cstk.disk

import com.google.common.base.Ascii
import com.google.common.collect.Iterables
import java.lang.Character.*

/** To check whether a char is whitespace/control/punctuation.  */
internal object CharChecker {
  /** To judge whether it's an empty or unknown character.  */
  fun isInvalid(ch: Char): Boolean = ch.code == 0 || ch.code == 0xfffd

  /** To judge whether it's a control character(exclude whitespace).  */
  fun isControl(ch: Char): Boolean {
    if (isWhitespace(ch)) return false
    val type = getType(ch)
    return type == CONTROL.toInt() || type == FORMAT.toInt()
  }

  /** To judge whether it can be regarded as a whitespace.  */
  fun isWhitespace(ch: Char): Boolean {
    if (ch.isWhitespace()) return true
    val type = getType(ch)
    return type == SPACE_SEPARATOR.toInt() ||
      type == LINE_SEPARATOR.toInt() ||
      type == PARAGRAPH_SEPARATOR.toInt()
  }

  /** To judge whether it's a punctuation.  */
  fun isPunctuation(ch: Char) =
    getType(ch) in setOf(
      CONNECTOR_PUNCTUATION.toInt(),
      DASH_PUNCTUATION.toInt(),
      START_PUNCTUATION.toInt(),
      END_PUNCTUATION.toInt(),
      INITIAL_QUOTE_PUNCTUATION.toInt(),
      FINAL_QUOTE_PUNCTUATION.toInt(),
      OTHER_PUNCTUATION.toInt()
    )
}

/** Basic tokenization (punctuation splitting, lower casing, etc.)  */
class BasicTokenizer(private val doLowerCase: Boolean) {
  fun tokenize(text: String?): List<String> {
    val cleanedText = cleanText(text)
    val origTokens = whitespaceTokenize(cleanedText)
    val stringBuilder = StringBuilder()
    for (token in origTokens) {
      val list = runSplitOnPunc(if (doLowerCase) { Ascii.toLowerCase(token) } else token)
      for (subToken in list) stringBuilder.append(subToken).append(" ")
    }
    return whitespaceTokenize(stringBuilder.toString())
  }

  companion object {
    /* Performs invalid character removal and whitespace cleanup on text. */
    fun cleanText(text: String?): String {
      if (text == null) throw NullPointerException("The input String is null.")
      val stringBuilder = StringBuilder("")
      for (element in text) {
        // Skip the characters that cannot be used.
        if (CharChecker.isInvalid(element) || CharChecker.isControl(element)) {
          continue
        }
        stringBuilder.append(if (CharChecker.isWhitespace(element)) " " else element)
      }
      return stringBuilder.toString()
    }

    /* Runs basic whitespace cleaning and splitting on a piece of text. */
    fun whitespaceTokenize(text: String?): List<String> {
      if (text == null) throw NullPointerException("The input String is null.")
      return listOf(*text.split(' ').toTypedArray())
    }

    /* Splits punctuation on a piece of text. */
    fun runSplitOnPunc(text: String?): List<String> {
      if (text == null) throw NullPointerException("The input String is null.")
      val tokens = mutableListOf<String>()
      var startNewWord = true
      for (element in text) {
        if (CharChecker.isPunctuation(element)) {
          tokens.add(element.toString())
          startNewWord = true
        } else {
          if (startNewWord) {
            tokens.add("")
            startNewWord = false
          }
          tokens[tokens.size - 1] = Iterables.getLast(tokens) + element
        }
      }
      return tokens
    }
  }
}

class FullTokenizer(private val dic: Map<String, Int>, doLowerCase: Boolean = false) {
  private val basicTokenizer: BasicTokenizer
  private val wordpieceTokenizer: WordpieceTokenizer
  fun tokenize(text: String?): List<String> {
    val splitTokens: MutableList<String> = ArrayList()
    for (token in basicTokenizer.tokenize(text))
      splitTokens.addAll(wordpieceTokenizer.tokenize(token))
    return splitTokens
  }

  fun convertTokensToIds(tokens: List<String>): List<Int?> {
    val outputIds: MutableList<Int?> = ArrayList()
    for (token in tokens) outputIds.add(dic[token])
    return outputIds
  }

  init {
    basicTokenizer = BasicTokenizer(doLowerCase)
    wordpieceTokenizer = WordpieceTokenizer(dic)
  }
}

/** Word piece tokenization to split a piece of text into its word pieces.  */
class WordpieceTokenizer(private val dic: Map<String, Int>) {
  /**
   * Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first
   * algorithm to perform tokenization using the given vocabulary. For example: input = "unaffable",
   * output = ["un", "##aff", "##able"].
   *
   * @param text: A single token or whitespace separated tokens. This should have already been
   * passed through `BasicTokenizer.
   * @return A list of wordpiece tokens.
   */
  fun tokenize(text: String?): List<String> {
    if (text == null) throw NullPointerException("The input String is null.")
    val outputTokens: MutableList<String> = ArrayList()
    for (token in BasicTokenizer.whitespaceTokenize(text)) {
      if (token.length > MAX_INPUTCHARS_PER_WORD) {
        outputTokens.add(UNKNOWN_TOKEN)
        continue
      }
      var isBad =
        false // Mark if a word cannot be tokenized into known subwords.
      var start = 0
      val subTokens: MutableList<String> = ArrayList()
      while (start < token.length) {
        var curSubStr = ""
        var end = token.length // Longer substring matches first.
        while (start < end) {
          val subStr = if (start == 0) token.substring(
            start,
            end
          ) else "##" + token.substring(start, end)
          if (dic.containsKey(subStr)) {
            curSubStr = subStr
            break
          }
          end--
        }

        // The word doesn't contain any known subwords.
        if ("" == curSubStr) {
          isBad = true
          break
        }

        // curSubStr is the longeset subword that can be found.
        subTokens.add(curSubStr)

        // Proceed to tokenize the resident string.
        start = end
      }
      if (isBad) {
        outputTokens.add(UNKNOWN_TOKEN)
      } else {
        outputTokens.addAll(subTokens)
      }
    }
    return outputTokens
  }

  companion object {
    private const val UNKNOWN_TOKEN = "[UNK]" // For unknown words.
    private const val MAX_INPUTCHARS_PER_WORD = 200
  }
}

// TODO pack the URL query with 512 tokens
fun main() {
//  val vocab = SimpleVocabulary.builder()
//    .optMinFrequency(1)
//    .add(VOCAB.readText().lines())
//    .optUnknownToken("[UNK]")
//    .build()
//
//  File("vocab.txt").let {
//    if (!it.exists())
//      it.writeText(
//        URL("https://huggingface.co/microsoft/codebert-base/resolve/main/vocab.json")
//          .readText()
//          .removePrefix("{\"")
//          .substringBeforeLast("\"")
////        .replace("Ġ", " ") //https://github.com/huggingface/transformers/issues/3867#issuecomment-616956437
//          .split(Regex("\": [0-9]*, \""))
//          .joinToString("\n")
//      )
//  }

//  println(
//     FullTokenizer().tokenize("System.out.println(\"hello world!\")")
//      .joinToString("\n"))
//
//
//  println(
//    BasicTokenizer(false).tokenize("System.out.println(\"hello world!\")")
//      .joinToString("\n"))
}