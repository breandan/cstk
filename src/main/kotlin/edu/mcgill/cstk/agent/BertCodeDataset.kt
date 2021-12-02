package edu.mcgill.cstk.agent

import ai.djl.modality.nlp.preprocess.UnicodeNormalizer
import ai.djl.ndarray.*
import ai.djl.training.dataset.*
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import edu.mcgill.cstk.disk.*
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.*
import java.util.Map.Entry.comparingByValue
import java.util.concurrent.ConcurrentHashMap
import kotlin.io.path.toPath
import kotlin.streams.*


class BertCodeDataset(
  var batchSize: Int = BATCH_SIZE,
  var epochLimit: Long = 1000L
): Dataset {
  var parsedFiles: Sequence<ParsedFile>? = null
  var dictionary: Dictionary? = null
  var rand: Random = Random(89724308)
  var manager: NDManager = NDManager.newBaseManager()

  override fun getData(manager: NDManager): Iterable<Batch> =
    object: Iterable<Batch>, Iterator<Batch> {
      var maskedInstances: List<MaskedInstance> = createEpochData()
      var idx: Int = batchSize

      private fun createEpochData(): List<MaskedInstance> {
        // turn data into sentence pairs containing consecutive lines
        val sentencePairs = ArrayList<SentencePair?>()
        parsedFiles!!.forEach { parsedFile: ParsedFile ->
          parsedFile.addToSentencePairs(sentencePairs)
        }
        sentencePairs.shuffle(rand)
        // swap sentences with 50% probability for next sentence task
        var idx = 1
        while (idx < sentencePairs.size) {
          sentencePairs[idx - 1]!!.maybeSwap(rand, sentencePairs[idx])
          idx += 2
        }
        // Create masked instances for training
        return sentencePairs
          .take(epochLimit.toInt())
          .map { MaskedInstance(rand, dictionary!!, it!!) }
      }

      override fun iterator(): Iterator<Batch> = this

      override fun hasNext(): Boolean = idx < maskedInstances.size

      override fun next(): Batch {
        val ndManager = manager.newSubManager()
        val batchData = maskedInstances.subList(idx - batchSize, idx)
        val batch = createBatch(ndManager, batchData, idx, maskedInstances.size)
        idx++
        return batch
      }
    }

  override fun prepare(progress: Progress?) {
    // get all applicable files
    parsedFiles = TEST_DIR.allFilesRecursively()
      .map { it.toPath() }
      // read & tokenize them
      .map { parseFile(it) }
    // determine dictionary
    dictionary = buildDictionary(countTokens(parsedFiles))
  }

  fun getDictionarySize(): Int = dictionary!!.tokens.size

  class ParsedFile constructor(val tokenizedLines: List<List<String>>) {
    fun addToSentencePairs(sentencePairs: MutableList<SentencePair?>) {
      var idx = 1
      while (idx < tokenizedLines.size) {
        sentencePairs.add(
          SentencePair(
            ArrayList(tokenizedLines[idx - 1]),
            ArrayList(tokenizedLines[idx])
          )
        )
        idx += 2
      }
    }
  }

  /** Helper class to preprocess data for the next sentence prediction task.  */
  class SentencePair constructor(
    var sentenceA: MutableList<String>,
    var sentenceB: MutableList<String>
  ) {
    var consecutive = true
    fun maybeSwap(rand: Random?, other: SentencePair?) {
      if (rand!!.nextBoolean()) {
        val otherA = other!!.sentenceA
        other.sentenceA = sentenceA
        sentenceA = otherA
        consecutive = false
        other.consecutive = false
      }
    }

    val totalLength: Int get() = sentenceA.size + sentenceB.size

    fun truncateToTotalLength(totalLength: Int) {
      var count = 0
      while (totalLength > totalLength) {
        if (count % 2 == 0 && sentenceA.isNotEmpty()) {
          sentenceA.removeAt(sentenceA.size - 1)
        } else if (sentenceB.isNotEmpty()) {
          sentenceB.removeAt(sentenceB.size - 1)
        }
        count++
      }
    }
  }

  /** A single bert pretraining instance. Applies masking to a given sentence pair.  */
  private class MaskedInstance constructor(
    rand: Random,
    val dictionary: Dictionary,
    val originalSentencePair: SentencePair,
  ) {
    val label = arrayOf(
      CLS,
      *originalSentencePair.sentenceA.toTypedArray(),
      SEP,
      *originalSentencePair.sentenceB.toTypedArray(),
      SEP,
    )

    // Randomly pick 20% of indices to mask
    val maskedIndices = label.indices.shuffled(DEFAULT_RAND)
      .take((label.size / 5).coerceAtMost(MAX_MASKING_PER_INSTANCE)).sorted()

    val masked = label.copyOf().also { masked ->
      // Perform masking of these indices
      maskedIndices.forEach {
        val r = rand.nextFloat()
        masked[it] = when {
          r < 0.8f -> MSK
          r < 0.9f -> dictionary.getRandomToken(rand)
          else -> masked[it]
        }
      }
    }

    // create type tokens (0 = sentence a, 1 = sentence b)
    val typeIds = IntArray(MAX_SEQUENCE_LENGTH) {
      val startIdx = originalSentencePair.sentenceA.size + 2
      val endIdx = startIdx + originalSentencePair.sentenceB.size
      if(it in startIdx..endIdx) 1 else 0
    }

    val tokenIds = IntArray(MAX_SEQUENCE_LENGTH)
      .apply { masked.forEachIndexed { i, it -> this[i] = dictionary[it] } }

    val inputMask = IntArray(MAX_SEQUENCE_LENGTH)
      .apply { fill(1, toIndex = label.size - 1) }

    val maskedPositions = IntArray(MAX_MASKING_PER_INSTANCE)
      .apply { maskedIndices.forEachIndexed { i, it -> this[i] = it } }

    val nextSentenceLabel = if (originalSentencePair.consecutive) 1 else 0

    val maskedIds: IntArray = IntArray(MAX_MASKING_PER_INSTANCE).apply {
      maskedIndices.forEachIndexed { i, it -> this[i] = dictionary[label[it]] }
    }

    val labelMask: IntArray = IntArray(MAX_MASKING_PER_INSTANCE)
      .apply { fill(1, toIndex = maskedIndices.size - 1) }
  }

  /** Helper class to create a token to id mapping.  */
  class Dictionary constructor(val tokens: List<String>) {
    private val tokenToId: MutableMap<String, Int>
    operator fun get(id: Int): String =
      if (id >= 0 && id < tokens.size) tokens[id] else UNK

    operator fun get(token: String): Int =
      tokenToId.getOrDefault(token, UNK_ID)

    fun toIds(tokens: List<String>): List<Int> = tokens.map { this[it] }

    fun toTokens(ids: List<Int>): List<String> = ids.map { this[it] }

    fun getRandomToken(rand: Random?): String =
      tokens[rand!!.nextInt(tokens.size)]

    init {
      tokenToId = HashMap(tokens.size)
      for (idx in tokens.indices) tokenToId[tokens[idx]] = idx
    }
  }

  companion object {
    private fun createBatch(
      ndManager: NDManager,
      instances: List<MaskedInstance>,
      idx: Int,
      dataSize: Int
    ): Batch {
      val inputs = NDList(
        batchFromList(ndManager, instances) { obj: MaskedInstance -> obj.tokenIds },
        batchFromList(ndManager, instances) { obj: MaskedInstance -> obj.typeIds },
        batchFromList(ndManager, instances) { obj: MaskedInstance -> obj.inputMask },
        batchFromList(ndManager, instances) { obj: MaskedInstance -> obj.maskedPositions }
      )
      val labels = NDList(
        nextSentenceLabelsFromList(ndManager, instances),
        batchFromList(ndManager, instances) { obj: MaskedInstance -> obj.maskedIds },
        batchFromList(ndManager, instances) { obj: MaskedInstance -> obj.labelMask }
      )
      return Batch(
        ndManager,
        inputs,
        labels,
        instances.size,
        Batchifier.STACK,
        Batchifier.STACK,
        idx.toLong(),
        dataSize.toLong()
      )
    }

    private fun batchFromList(
      ndManager: NDManager,
      batchData: List<IntArray>
    ) = ndManager.create(batchData.toTypedArray())

    private fun batchFromList(
      ndManager: NDManager,
      instances: List<MaskedInstance>,
      f: (MaskedInstance) -> IntArray
    ): NDArray = batchFromList(ndManager, instances.map { f(it) })

    private fun nextSentenceLabelsFromList(
      ndManager: NDManager, instances: List<MaskedInstance>
    ) = ndManager.create(instances.map { it.nextSentenceLabel }.toIntArray())

    private fun normalizeLine(line: String): String {
      if (line.isEmpty()) return line
      // in source code, preceding whitespace is relevant, trailing ws is not
      // so we get the index of the last non ws char
      val unicodeNormalized = UnicodeNormalizer.normalizeDefault(line)
      var endIdx = line.length - 1
      while (endIdx >= 0 && unicodeNormalized[endIdx].isWhitespace()) endIdx--
      return line.substring(0, endIdx + 1)
    }

    private fun fileToLines(file: Path): List<String> = try {
      Files.lines(file, StandardCharsets.UTF_8)
        .map { line: String -> normalizeLine(line) }
        .filter { line: String -> line.trim { it <= ' ' }.isNotEmpty() }
        .asSequence().toList()
    } catch (ioe: IOException) {
      throw IllegalStateException("Could not read file $file", ioe)
    }

    private fun tokenizeLine(normalizedLine: String): List<String> {
      if (normalizedLine.isEmpty()) return emptyList()
      if (normalizedLine.length == 1) return listOf(normalizedLine)
      val result: MutableList<String> = ArrayList()
      val length = normalizedLine.length
      val currentToken = StringBuilder()
      for (idx in 0..length) {
        val c: Char = if (idx < length) normalizedLine[idx] else 0.toChar()
        val isAlphabetic = Character.isAlphabetic(c.code)
        val isUpperCase = Character.isUpperCase(c)
        if (c.code == 0 || !isAlphabetic || isUpperCase) {
          // we have reached the end of the string, encountered something other than a letter
          // or reached a new part of a camel-cased word - emit a new token
          if (currentToken.isNotEmpty()) {
            result.add(currentToken.toString().lowercase())
            currentToken.setLength(0)
          }
          // if we haven't reached the end, we need to use the char
          if (c.code != 0) {
            // the char is not alphabetic, turn it into a separate token
            if (!isAlphabetic) result.add(c.toString())
            else currentToken.append(c)
          }
        } else {
          // we have a new char to append to the current token
          currentToken.append(c)
        }
      }
      return result
    }

    private fun countTokens(parsedFiles: Sequence<ParsedFile>?): Map<String, Long> {
      val result: MutableMap<String, Long> = ConcurrentHashMap(50000)
      parsedFiles!!.forEach { parsedFile: ParsedFile ->
        parsedFile.tokenizedLines.forEach { tokens: List<String> ->
          tokens.forEach { token ->
            val count = result.getOrDefault(token, 0L)
            result[token] = count + 1
          }
        }
      }
      return result
    }

    private fun parseFile(file: Path) = ParsedFile(
      fileToLines(file)
        .map { line: String -> normalizeLine(line) }
        .filter { line: String -> line.isNotEmpty() }
        .map { normalizedLine: String -> tokenizeLine(normalizedLine) }
    )

    private fun buildDictionary(
      countedTokens: Map<String, Long>,
      maxSize: Int = 35000
    ): Dictionary {
      if (maxSize < RESERVED_TOKENS.size)
        throw IllegalArgumentException(
          "Dictionary must be at least ${RESERVED_TOKENS.size} long."
        )

      val result = ArrayList<String>(maxSize)
      result.addAll(RESERVED_TOKENS)
      val sortedByFrequency = countedTokens
        .entries
        .sortedWith(comparingByValue(Comparator.reverseOrder()))
        .map { it.key }

      var idx = 0
      while (result.size < maxSize && idx < sortedByFrequency.size) {
        result.add(sortedByFrequency[idx])
        idx++
      }
      return Dictionary(result)
    }
  }
}