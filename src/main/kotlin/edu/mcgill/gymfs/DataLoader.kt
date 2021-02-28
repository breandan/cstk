package edu.mcgill.gymfs

import kotlin.random.Random


/** A single bert pretraining instance. Applies masking to a given sentence pair.  */
class MaskedInstance constructor(
  rand: Random,
  val dictionary: Dictionary,
  val originalSentencePair: SentencePair,
) {
  val label: ArrayList<String> =
    ArrayList<String>(originalSentencePair.totalLength + 3).also { label ->
      label.add(CLS)
      label.addAll(originalSentencePair.sentenceA)
      label.add(SEP)
      label.addAll(originalSentencePair.sentenceB)
      label.add(SEP)
    }

  // Randomly pick 20% of indices to mask
  val maskedIndices = label.indices.shuffled(rand)
    .take((label.size / 5).coerceAtMost(MAX_MASKING_PER_INSTANCE)).sorted()

  val masked = ArrayList(label).also { masked ->
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

  // create type tokens (0 = sentence a, 1, sentence b)
  val typeIds: ArrayList<Int> = ArrayList<Int>(label.size).also { typeIds ->
    var typeId = 0
    for (idx in label.indices) {
      typeIds.add(typeId)
      if (label[idx] === SEP) typeId++
    }
  }

  val tokenIds = IntArray(MAX_SEQUENCE_LENGTH)
    .apply { masked.forEachIndexed { i, it -> this[i] = dictionary[it] } }

  fun getTypeIds() = IntArray(MAX_SEQUENCE_LENGTH)
    .apply { typeIds.forEachIndexed { i, it -> this[i] = it } }

  val inputMask = IntArray(MAX_SEQUENCE_LENGTH)
    .apply { fill(1, toIndex = typeIds.size - 1) }

  val maskedPositions = IntArray(MAX_MASKING_PER_INSTANCE)
    .apply { maskedIndices.forEachIndexed { i, it -> this[i] = it } }

  val nextSentenceLabel = if (originalSentencePair.consecutive) 1 else 0

  val maskedIds: IntArray = IntArray(MAX_MASKING_PER_INSTANCE)
    .apply { maskedIndices.forEachIndexed { i, it -> this[i] = dictionary[label[it]] } }

  val labelMask: IntArray = IntArray(MAX_MASKING_PER_INSTANCE)
    .apply { fill(1, toIndex = maskedIndices.size - 1) }
}

/** Helper class to create a token to id mapping.  */
class Dictionary(frequencyList: List<String>) {
  val tokens = (RESERVED_TOKENS + frequencyList).take(MAX_VOCAB)

  val tokenToId: MutableMap<String, Int> = HashMap<String, Int>(tokens.size)
    .apply { tokens.forEachIndexed { i, it -> this[it] = i } }

  fun getToken(id: Int): String =
    if (id >= 0 && id < tokens.size) tokens[id] else UNK

  operator fun get(token: String): Int = tokenToId.getOrDefault(token, UNK_ID)

  fun getRandomToken(rand: Random): String = tokens.random(rand)
}

class SentencePair(
  var sentenceA: ArrayList<String>, var sentenceB: ArrayList<String>
) {
  var consecutive = true
  fun maybeSwap(rand: Random, other: SentencePair?) {
    if (rand.nextBoolean()) {
      val otherA = other!!.sentenceA
      other.sentenceA = sentenceA
      sentenceA = otherA
      consecutive = false
      other.consecutive = false
    }
  }

  val totalLength = sentenceA.size + sentenceB.size

  fun truncateToTotalLength(totalLength: Int) {
    var count = 0
    while (this.totalLength > totalLength) {
      if (count % 2 == 0 && sentenceA.isNotEmpty())
        sentenceA.removeAt(sentenceA.size - 1)
      else if (sentenceB.isNotEmpty())
        sentenceB.removeAt(sentenceB.size - 1)
      count++
    }
  }
}
