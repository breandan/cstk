package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.automata.DFSM
import ai.hypergraph.kaliningraph.parsing.*
import java.util.*
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.stream.IntStream

data class DFASize(val states: Long, val transitions: Long)

/** Computes minimal partial-DFA sizes for L(this) ∩ Σⁿ without expanding the finite parse forest. */
fun CFG.minimalSliceDFASizes(
  maxLength: Int,
  onSlice: (Int, DFASize) -> Unit = { _, _ -> }
): List<DFASize> = buildMinimalSlices(maxLength, onSlice).sizes

/** Constructs the actual minimal partial DFA for L(this) ∩ Σ^[length]. */
fun CFG.minimalSliceDFA(
  length: Int,
  onSlice: (Int, DFASize) -> Unit = { _, _ -> }
): PackedDFA = buildMinimalSlices(length, onSlice).let { build ->
  build.arena.pack(build.root, tmLst, build.sizes.last())
}

/** Primitive CSR representation of a deterministic finite automaton. */
class PackedDFA internal constructor(
  val terminals: List<String>,
  val startState: Int,
  val finalState: Int,
  private val offsets: IntArray,
  private val edges: IntArray,
  private val targetBits: Int
) {
  val stateCount: Int get() = offsets.size - 1
  val transitionCount: Int get() = edges.size
  val width: Int get() = terminals.size
  val size: DFASize get() = DFASize(stateCount.toLong(), transitionCount.toLong())
  fun summarize() = "(states=$stateCount, transitions=$transitionCount)"
  private val targetMask = if (targetBits == 0) 0 else -1 ushr (Int.SIZE_BITS - targetBits)
  private val terminalIds by lazy(LazyThreadSafetyMode.PUBLICATION) {
    terminals.withIndex().associate { (i, terminal) -> terminal to i }
  }

  fun isFinal(state: Int): Boolean = finalState >= 0 && state == finalState
  fun outBegin(state: Int): Int = offsets[state]
  fun outEnd(state: Int): Int = offsets[state + 1]
  fun labelAt(edge: Int): Int = edges[edge] ushr targetBits
  fun targetAt(edge: Int): Int = edges[edge] and targetMask

  /** Returns the target state, or -1 when the partial DFA has no such transition. */
  fun transition(state: Int, label: Int): Int {
    var low = outBegin(state)
    var high = outEnd(state) - 1
    while (low <= high) {
      val middle = (low + high) ushr 1
      val found = labelAt(middle)
      when {
        found < label -> low = middle + 1
        label < found -> high = middle - 1
        else -> return targetAt(middle)
      }
    }
    return -1
  }

  fun recognizes(labels: IntArray): Boolean {
    var state = startState
    for (label in labels) {
      state = transition(state, label)
      if (state < 0) return false
    }
    return isFinal(state)
  }

  fun recognizes(tokens: Iterable<String>): Boolean {
    var state = startState
    for (token in tokens) {
      val label = terminalIds[token] ?: return false
      state = transition(state, label)
      if (state < 0) return false
    }
    return isFinal(state)
  }

  inline fun forEachTransition(state: Int, action: (label: Int, target: Int) -> Unit) {
    for (edge in outBegin(state) until outEnd(state)) action(labelAt(edge), targetAt(edge))
  }

  /** Compatibility adapter. The legacy type boxes and duplicates its transition graph. */
  fun toDFSM(maxStates: Int = 100_000, maxTransitions: Int = 1_000_000): DFSM {
    require(stateCount <= maxStates && transitionCount <= maxTransitions) {
      "Refusing to inflate $stateCount states and $transitionCount transitions into legacy DFSM"
    }
    val names = Array(stateCount) { "q$it" }
    val delta = LinkedHashMap<String, Map<Int, String>>(stateCount)
    for (state in names.indices) {
      val row = LinkedHashMap<Int, String>(outEnd(state) - outBegin(state))
      forEachTransition(state) { label, target -> row[label] = names[target] }
      delta[names[state]] = row
    }
    return DFSM(
      Q = names.toSet(),
      deltaMap = delta,
      q_alpha = names[startState],
      F = if (finalState < 0) emptySet() else setOf(names[finalState]),
      width = width
    )
  }
}

private data class MinimalSliceBuild(
  val arena: AcyclicDFAArena,
  val root: Int,
  val sizes: List<DFASize>
)

private fun CFG.buildMinimalSlices(
  maxLength: Int,
  onSlice: (Int, DFASize) -> Unit
): MinimalSliceBuild {
  require(maxLength >= 1)
  val width = nonterminals.size
  val binary = Array(width) { IntArray(0) }
  val terminals = Array(width) { IntArray(0) }

  forEach { (lhs, rhs) ->
    require(
      rhs.size == 2 && rhs.all { it in nonterminals } ||
        rhs.size == 1 && rhs[0] !in nonterminals
    ) { "Expected a CFG in binary normal form, found $lhs -> ${rhs.joinToString(" ")}" }
  }

  vindex.forEachIndexed { a, rules -> binary[a] = rules }
  terminalLists.forEachIndexed { a, labels ->
    terminals[a] = labels.map(tmMap::getValue).distinct().sorted().toIntArray()
  }

  val arena = AcyclicDFAArena()
  val table = Array(maxLength + 1) { IntArray(width) { AcyclicDFAArena.EMPTY } }

  IntStream.range(0, width).forEach { a ->
    val labels = terminals[a]
    if (labels.isNotEmpty()) {
      table[1][a] = arena.intern(IntArray(labels.size * 2) { i ->
        if (i and 1 == 0) labels[i / 2] else AcyclicDFAArena.FINAL
      })
    }
  }

  val start = bindex[START_SYMBOL]
  val result = ArrayList<DFASize>(maxLength)
  result += arena.sizeOf(table[1][start]).also { onSlice(1, it) }

  for (length in 2..maxLength) {
    val n = length
    // Only START is needed in the last layer; every child has a shorter length.
    val cells = if (n == maxLength) IntStream.of(start) else IntStream.range(0, width)
    // Cells in one layer read only completed, shorter layers.
    (if (n < 8 || n == maxLength) cells else cells.parallel()).forEach { a ->
      val rules = binary[a]
      val products = LongArray((rules.size / 2) * (n - 1))
      var size = 0
      for (r in rules.indices step 2) {
        val b = rules[r]
        val c = rules[r + 1]
        for (split in 1 until n) {
          val left = table[split][b]
          val right = table[n - split][c]
          if (left != AcyclicDFAArena.EMPTY && right != AcyclicDFAArena.EMPTY)
            products[size++] = AcyclicDFAArena.product(left, right)
        }
      }
      table[n][a] = arena.unionProducts(products.copyOf(size))
    }
    result += arena.sizeOf(table[n][start]).also { onSlice(n, it) }
    arena.clearProductMemo()
  }
  return MinimalSliceBuild(arena, table[maxLength][start], result)
}

private class AcyclicDFAArena {
  // EMPTY=-1; FINAL=0; every other id names one canonical [label, target, ...] row.
  private class LongBuffer {
    private var values = LongArray(4)
    private var size = 0
    fun add(value: Long) {
      if (size == values.size) values = values.copyOf(size * 2)
      values[size++] = value
    }
    fun toLongArray() = values.copyOf(size)
  }

  private data class IntArrayKey(val values: IntArray) {
    override fun hashCode() = values.contentHashCode()
    override fun equals(other: Any?) = other is IntArrayKey && values.contentEquals(other.values)
  }

  private data class LongArrayKey(val values: LongArray) {
    override fun hashCode() = values.contentHashCode()
    override fun equals(other: Any?) = other is LongArrayKey && values.contentEquals(other.values)
  }

  private val nextId = AtomicInteger(1)
  private val rows = ConcurrentHashMap<Int, IntArray>().apply { put(FINAL, IntArray(0)) }
  private val rowIds = ConcurrentHashMap<IntArrayKey, Int>().apply { put(IntArrayKey(IntArray(0)), FINAL) }
  @Volatile private var productMemo = ConcurrentHashMap<LongArrayKey, CompletableFuture<Int>>()

  fun intern(row: IntArray): Int {
    require(row.isNotEmpty())
    return rowIds.computeIfAbsent(IntArrayKey(row)) {
      nextId.getAndIncrement().also { rows[it] = row }
    }
  }

  fun unionProducts(raw: LongArray): Int {
    var size = 0
    for (packed in raw) {
      var prefix = left(packed)
      var suffix = right(packed)
      if (prefix == EMPTY || suffix == EMPTY) continue
      if (prefix == FINAL) prefix = suffix.also { suffix = FINAL }
      raw[size++] = product(prefix, suffix)
    }
    if (size == 0) return EMPTY
    Arrays.sort(raw, 0, size)
    var unique = 1
    for (i in 1 until size)
      if (raw[i] != raw[unique - 1]) raw[unique++] = raw[i]
    if (unique == 1 && right(raw[0]) == FINAL) return left(raw[0])

    val frozen = raw.copyOf(unique)
    val key = LongArrayKey(frozen)
    val mine = CompletableFuture<Int>()
    val future = productMemo.putIfAbsent(key, mine)
    if (future != null) return future.join()

    try {
      val byLabel = HashMap<Int, LongBuffer>()
      frozen.forEach { packed ->
        val prefix = left(packed)
        val suffix = right(packed)
        require(prefix != FINAL)
        val row = rows.getValue(prefix)
        for (i in row.indices step 2)
          byLabel.getOrPut(row[i]) { LongBuffer() }.add(product(row[i + 1], suffix))
      }

      val result = IntArray(byLabel.size * 2)
      var i = 0
      byLabel.keys.sorted().forEach { label ->
        result[i++] = label
        result[i++] = unionProducts(byLabel.getValue(label).toLongArray())
      }
      return intern(result).also(mine::complete)
    } catch (t: Throwable) {
      mine.completeExceptionally(t)
      productMemo.remove(key, mine)
      throw t
    }
  }

  fun sizeOf(root: Int): DFASize {
    if (root == EMPTY) return DFASize(1, 0)
    val seen = BitSet(nextId.get())
    var queue = IntArray(1024)
    var head = 0
    var tail = 1
    queue[0] = root
    seen[root] = true
    var states = 0L
    var transitions = 0L

    while (head < tail) {
      val row = rows.getValue(queue[head++])
      states++
      transitions += row.size / 2
      for (i in 1 until row.size step 2) {
        val target = row[i]
        if (!seen[target]) {
          seen[target] = true
          if (tail == queue.size) queue = queue.copyOf(queue.size * 2)
          queue[tail++] = target
        }
      }
    }
    return DFASize(states, transitions)
  }

  fun pack(root: Int, terminals: List<String>, expected: DFASize): PackedDFA {
    productMemo.clear()
    rowIds.clear()
    if (root == EMPTY)
      return PackedDFA(terminals.toList(), 0, EMPTY, intArrayOf(0, 0), IntArray(0), 0)

    require(expected.states < Int.MAX_VALUE) { "Packed DFA has too many states: ${expected.states}" }
    require(expected.transitions <= Int.MAX_VALUE) { "Packed DFA has too many transitions: ${expected.transitions}" }
    val stateCount = expected.states.toInt()
    val transitionCount = expected.transitions.toInt()
    val arenaToDense = IntArray(nextId.get()) { EMPTY }
    val denseToArena = IntArray(stateCount)
    val offsets = IntArray(stateCount + 1)
    arenaToDense[root] = 0
    denseToArena[0] = root
    var head = 0
    var tail = 1

    while (head < tail) {
      val row = rows.getValue(denseToArena[head])
      offsets[head + 1] = offsets[head] + row.size / 2
      for (i in 1 until row.size step 2) {
        val target = row[i]
        if (arenaToDense[target] == EMPTY) {
          arenaToDense[target] = tail
          denseToArena[tail++] = target
        }
      }
      head++
    }
    check(tail == stateCount && offsets.last() == transitionCount)

    val targetBits = (stateCount - 1).bitLength()
    val labelBits = (terminals.size - 1).bitLength()
    require(targetBits + labelBits <= Int.SIZE_BITS) {
      "Cannot pack $stateCount states and ${terminals.size} labels into 32-bit transitions"
    }
    val edges = IntArray(transitionCount)
    IntStream.range(0, stateCount).parallel().forEach { state ->
      val row = rows.getValue(denseToArena[state])
      var edge = offsets[state]
      for (i in row.indices step 2) {
        val label = row[i]
        val target = arenaToDense[row[i + 1]]
        check(target >= 0)
        edges[edge++] = (label shl targetBits) or target
      }
    }

    val finalState = arenaToDense[FINAL]
    check(finalState >= 0)
    rows.clear()
    return PackedDFA(
      terminals = terminals.toList(),
      startState = 0,
      finalState = finalState,
      offsets = offsets,
      edges = edges,
      targetBits = targetBits
    )
  }

  fun clearProductMemo() { productMemo = ConcurrentHashMap() }

  companion object {
    const val EMPTY = -1
    const val FINAL = 0
    fun product(left: Int, right: Int) = (left.toLong() shl 32) or (right.toLong() and 0xffffffffL)
    private fun left(product: Long) = (product shr 32).toInt()
    private fun right(product: Long) = product.toInt()
  }
}

private fun Int.bitLength() = if (this <= 0) 0 else Int.SIZE_BITS - countLeadingZeroBits()
