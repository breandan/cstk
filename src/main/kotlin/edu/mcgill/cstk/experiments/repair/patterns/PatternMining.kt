package edu.mcgill.cstk.experiments.repair.patterns

import ai.hypergraph.kaliningraph.automata.latestLangEditDistance
import ai.hypergraph.kaliningraph.parsing.levenshtein
import ai.hypergraph.kaliningraph.parsing.tmLst
import ai.hypergraph.kaliningraph.repair.CFG_THRESH
import ai.hypergraph.kaliningraph.repair.LED_BUFFER
import ai.hypergraph.kaliningraph.repair.LangCache
import ai.hypergraph.kaliningraph.repair.MAX_RADIUS
import ai.hypergraph.kaliningraph.repair.MAX_TOKENS
import ai.hypergraph.kaliningraph.repair.MIN_TOKENS
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.repair.s2pg
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import edu.mcgill.cstk.experiments.repair.evaluateRegexRepairOnStackOverflow
import edu.mcgill.cstk.experiments.repair.sendCPU
import edu.mcgill.cstk.experiments.repair.sizeAndDistBalancedRepairsUnminimized
import java.io.OutputStream
import java.io.PrintStream
import java.util.Locale

fun main(args: Array<String>) {
    val patternConfig = parseRepairPatternCollectorArgs(args) ?: return
    if (patternConfig.selfTest) {
        testRepairPatternClassifier()
        return
    }
    collectUnrecognizedRepairPatterns(
        limit = patternConfig.limit,
        examplesPerPattern = patternConfig.examplesPerPattern,
        maxBlockSize = patternConfig.maxBlockSize,
        verboseDfa = patternConfig.verboseDfa
    )
}

private const val DEFAULT_MAX_PATTERN_BLOCK = 4
private const val ALIGNMENT_INFINITY = Int.MAX_VALUE / 4

private data class RepairPatternCollectorConfig(
    val limit: Int? = null,
    val examplesPerPattern: Int = 3,
    val maxBlockSize: Int = DEFAULT_MAX_PATTERN_BLOCK,
    val verboseDfa: Boolean = false,
    val selfTest: Boolean = false
)

/**
 * The order is also the specificity order used to choose a primary label when
 * several augmented edit families cover the same repair.
 *
 * A weighted one-symbol substitution is deliberately absent: it is already a
 * vanilla Levenshtein transition at unit cost, and deciding whether a cheaper
 * transition applies requires a concrete confusion relation and weight.
 */
private enum class RepairPattern(
    val displayName: String,
    val automatonComplexity: String,
    val requiresRuleInventory: Boolean = false
) {
    ADJACENT_TRANSPOSITION(
        "Adjacent transposition",
        "Theta(nd)"
    ),
    CHARACTER_REPETITION(
        "Token-symbol repetition or unduplication",
        "Theta(nd), fixed maximum repetition"
    ),
    TANDEM_DUPLICATION(
        "Tandem block duplication or contraction",
        "O(ndB) macro arcs"
    ),
    BOUNDED_REVERSAL(
        "Bounded block reversal",
        "O(ndB) macro arcs"
    ),
    ADJACENT_BLOCK_SWAP(
        "Bounded adjacent-block swap",
        "O(ndB^2) macro arcs"
    ),
    ARBITRARY_SYMBOL_SWAP(
        "One arbitrary non-adjacent symbol swap",
        "O(nq^2), O(n^2) on a fixed alphabet"
    ),
    AFFINE_GAP_RUN(
        "Affine gap run (open=1, extension=0)",
        "Theta(nd), constant gap modes"
    ),
    WRAP_OR_UNWRAP(
        "Wrapper-pair rule candidate",
        "Theta(nd) after fixing the wrapper-pair inventory",
        requiresRuleInventory = true
    ),
    MULTI_SYMBOL_MERGE_OR_SPLIT(
        "Bounded 1-to-many rule candidate",
        "O(nd rho B) after fixing rho merge/split rules",
        requiresRuleInventory = true
    )
}

private data class PatternMacroUse(
    val sourceStart: Int,
    val sourceEnd: Int,
    val targetStart: Int,
    val targetEnd: Int,
    val sourceSymbols: List<String>,
    val targetSymbols: List<String>,
    val shape: String
) {
    fun render(): String {
        fun List<String>.renderSide() = if (isEmpty()) "epsilon" else joinToString(" ")
        return "${sourceSymbols.renderSide()} -> ${targetSymbols.renderSide()} ($shape)"
    }
}

private data class InventoryRuleKey(
    val direction: String,
    val source: List<String>,
    val target: List<String>
)

private fun PatternMacroUse.inventoryRuleKey(pattern: RepairPattern): InventoryRuleKey =
    when (pattern) {
        RepairPattern.WRAP_OR_UNWRAP -> {
            val wrapper =
                if (shape.startsWith("wrap ")) targetSymbols else sourceSymbols
            InventoryRuleKey(
                direction = shape.substringBefore(' '),
                source = listOfNotNull(wrapper.firstOrNull(), wrapper.lastOrNull()),
                target = emptyList()
            )
        }
        RepairPattern.MULTI_SYMBOL_MERGE_OR_SPLIT ->
            InventoryRuleKey(shape.substringBefore(' '), sourceSymbols, targetSymbols)
        else -> InventoryRuleKey(shape, sourceSymbols, targetSymbols)
    }

private data class PatternAlignment(
    val distance: Int,
    val macroUses: List<PatternMacroUse>
) {
    fun isBetterThan(other: PatternAlignment?): Boolean =
        other == null ||
                distance < other.distance ||
                (distance == other.distance && macroUses.size < other.macroUses.size)
}

private data class RepairPatternMatch(
    val pattern: RepairPattern,
    val augmentedDistance: Int,
    val macroUses: List<PatternMacroUse>
)

private data class CataloguedRepair(
    val id: Int,
    val brokenTokens: String,
    val fixedTokens: String,
    val brokenCode: String,
    val fixedCode: String,
    val vanillaDistance: Int,
    val radius: Int
)

private data class PatternExample(
    val repair: CataloguedRepair,
    val match: RepairPatternMatch
)

private fun parseRepairPatternCollectorArgs(args: Array<String>): RepairPatternCollectorConfig? {
    var config = RepairPatternCollectorConfig()

    fun nonNegative(name: String, value: String): Int =
        value.toIntOrNull()?.takeIf { it >= 0 }
            ?: error("$name must be a non-negative integer, got '$value'")

    for (arg in args) {
        config = when {
            arg == "--help" || arg == "-h" -> {
                println(
                    """
          Collect repair patterns outside the Python Bar-Hillel DFA.

          Usage:
            ./gradlew collectSummaryStats
            ./gradlew collectSummaryStats --args='--limit=25 --examples=2 --max-block=4'
            ./gradlew collectSummaryStats --args='--self-test'

          Options:
            --limit=N       Process at most N balanced repair instances.
            --examples=N    Print at most N examples per pattern (default: 3).
            --max-block=N   Maximum bounded block/repetition size (default: 4).
            --verbose-dfa   Keep per-instance DFA construction logs.
            --self-test     Run fast classifier checks without loading Python data.
          """.trimIndent()
                )
                return null
            }
            arg == "--self-test" -> config.copy(selfTest = true)
            arg == "--verbose-dfa" -> config.copy(verboseDfa = true)
            arg.startsWith("--limit=") ->
                config.copy(limit = nonNegative("--limit", arg.substringAfter('=')))
            arg.startsWith("--examples=") ->
                config.copy(examplesPerPattern = nonNegative("--examples", arg.substringAfter('=')))
            arg.startsWith("--max-block=") -> {
                val size = nonNegative("--max-block", arg.substringAfter('='))
                require(size >= 2) { "--max-block must be at least 2" }
                config.copy(maxBlockSize = size)
            }
            else -> error("Unknown argument '$arg'; use --help for usage")
        }
    }

    return config
}

private fun <T> suppressStandardOutput(action: () -> T): T {
    val previousOutput = System.out
    val sink = PrintStream(OutputStream.nullOutputStream())
    return try {
        System.setOut(sink)
        action()
    } finally {
        System.setOut(previousOutput)
        sink.close()
    }
}

/**
 * Catalogues only instances satisfying the exact predicate used in
 * [evaluateRegexRepairOnStackOverflow]:
 *
 *     dfa?.recognizes(fixedTokens, s2pg.tmLst) ?: false
 *
 * Classification is performed over the same abstract token alphabet as the
 * Levenshtein DFA. For each edit family, a two-layer Wagner-Fischer table
 * permits exactly one atomic macro transition plus ordinary edits before and
 * after it. This is deliberately conservative for edits that would overlap the
 * interior of a copied/swapped block. A family covers an instance only when it
 * strictly lowers the token edit distance into the radius used to construct
 * the sparse-GRE DFA.
 */
fun collectUnrecognizedRepairPatterns(
    limit: Int? = null,
    examplesPerPattern: Int = 3,
    maxBlockSize: Int = DEFAULT_MAX_PATTERN_BLOCK,
    verboseDfa: Boolean = false
) {
    require(limit == null || limit >= 0)
    require(examplesPerPattern >= 0)
    require(maxBlockSize >= 2)

    // Match PythonBarHillelRepair.main before forcing the lazy balanced dataset.
    LangCache.prepopPythonLangCache()
    TIMEOUT_MS = 30_000
    MIN_TOKENS = 3
    MAX_TOKENS = 80
    MAX_RADIUS = 3
    LED_BUFFER = 1
    CFG_THRESH = 10_000

    val sampledDataset =
        limit?.let { sizeAndDistBalancedRepairsUnminimized.take(it).toList() }
            ?: sizeAndDistBalancedRepairsUnminimized.toList()
    val dataset = sampledDataset
        .sortedWith(
            compareBy(
                { it.π1 },
                { it.π2 },
                { it.π3 },
                { it.π4 }
            )
        )

    val matchesByPattern =
        RepairPattern.entries.associateWith { mutableListOf<PatternExample>() }
    val primaryCounts = RepairPattern.entries.associateWith { 0 }.toMutableMap()
    val unmatched = mutableListOf<CataloguedRepair>()

    var processed = 0
    var recognized = 0
    var rejected = 0
    var dfaNull = 0
    var inRadiusRejections = 0
    var classifiable = 0
    var coveredByRuleFreePattern = 0
    var coveredIncludingRuleCandidates = 0
    var failures = 0

    fun reportProgress() {
        if (processed % 25 == 0 || processed == dataset.size) {
            println(
                "Pattern collector: $processed/${dataset.size} processed, " +
                        "$rejected rejected, $coveredIncludingRuleCandidates/$classifiable covered"
            )
        }
    }

    dataset.forEachIndexed { index, (brokenString, fixedString, brokenCode, fixedCode) ->
        processed++
        val brokenTokens = brokenString.tokenizeByWhitespace()
        val fixedTokens = fixedString.tokenizeByWhitespace()
        val vanillaDistance = levenshtein(brokenTokens, fixedTokens)

        val dfa = try {
            if (verboseDfa) sendCPU(brokenString)
            else suppressStandardOutput { sendCPU(brokenString) }
        } catch (e: Exception) {
            failures++
            System.err.println(
                "Pattern collector failed to construct instance ${index + 1}: " +
                        "${e::class.simpleName}: ${e.message}"
            )
            reportProgress()
            return@forEachIndexed
        }

        // repairWithSparseGRE writes this global; capture it before the next item.
        val languageEditDistance = latestLangEditDistance
        val radius = sparseGRESearchRadius(languageEditDistance)
        val dfaRecognized = try {
            dfa?.recognizes(fixedTokens, s2pg.tmLst) ?: false
        } catch (e: Exception) {
            failures++
            System.err.println(
                "Pattern collector failed to recognize instance ${index + 1}: " +
                        "${e::class.simpleName}: ${e.message}"
            )
            reportProgress()
            return@forEachIndexed
        }

        if (dfaRecognized) {
            recognized++
            reportProgress()
            return@forEachIndexed
        }

        // From here down, every item matches `!dfaRecognized`.
        rejected++
        if (dfa == null) dfaNull++
        if (vanillaDistance <= radius) {
            // This is a construction/encoding diagnostic, not a missing edit family.
            inRadiusRejections++
            reportProgress()
            return@forEachIndexed
        }

        classifiable++
        val repair = CataloguedRepair(
            id = index + 1,
            brokenTokens = brokenString,
            fixedTokens = fixedString,
            brokenCode = brokenCode,
            fixedCode = fixedCode,
            vanillaDistance = vanillaDistance,
            radius = radius
        )
        val matches = RepairPattern.entries.mapNotNull { pattern ->
            patternAwareAlignment(
                source = brokenTokens,
                target = fixedTokens,
                pattern = pattern,
                maxBlockSize = maxBlockSize
            ).takeIf {
                it.distance < vanillaDistance &&
                        it.distance <= radius &&
                        it.macroUses.isNotEmpty()
            }?.let {
                RepairPatternMatch(pattern, it.distance, it.macroUses)
            }
        }

        if (matches.isEmpty()) {
            unmatched.add(repair)
        } else {
            coveredIncludingRuleCandidates++
            if (matches.any { !it.pattern.requiresRuleInventory })
                coveredByRuleFreePattern++
            matches.forEach { match ->
                matchesByPattern.getValue(match.pattern).add(PatternExample(repair, match))
            }
            val primary = matches.minWith(
                compareBy<RepairPatternMatch>(
                    { it.pattern.ordinal },
                    { it.augmentedDistance },
                    { it.macroUses.size }
                )
            )
            primaryCounts[primary.pattern] = primaryCounts.getValue(primary.pattern) + 1
        }

        reportProgress()
    }

    printRepairPatternReport(
        processed = processed,
        recognized = recognized,
        rejected = rejected,
        dfaNull = dfaNull,
        inRadiusRejections = inRadiusRejections,
        classifiable = classifiable,
        coveredByRuleFreePattern = coveredByRuleFreePattern,
        coveredIncludingRuleCandidates = coveredIncludingRuleCandidates,
        failures = failures,
        matchesByPattern = matchesByPattern,
        primaryCounts = primaryCounts,
        unmatched = unmatched,
        examplesPerPattern = examplesPerPattern,
        maxBlockSize = maxBlockSize
    )
}

/**
 * Keep this synchronized with repairWithSparseGRE. PythonBarHillelRepair's
 * human-readable log currently caps at MAX_RADIUS, while the actual sparse GRE
 * is constructed with a MAX_RADIUS + LED_BUFFER cap.
 */
private fun sparseGRESearchRadius(languageEditDistance: Int): Int =
    (languageEditDistance + LED_BUFFER).coerceAtMost(MAX_RADIUS + LED_BUFFER)

private fun patternAwareAlignment(
    source: List<String>,
    target: List<String>,
    pattern: RepairPattern,
    maxBlockSize: Int
): PatternAlignment {
    val rows = source.size + 1
    val columns = target.size + 1
    val withoutMacro = Array(rows) {
        Array(columns) { PatternAlignment(ALIGNMENT_INFINITY, emptyList()) }
    }
    val withMacro = Array(rows) {
        Array(columns) { PatternAlignment(ALIGNMENT_INFINITY, emptyList()) }
    }
    withoutMacro[0][0] = PatternAlignment(0, emptyList())

    for (i in 0..source.size) {
        for (j in 0..target.size) {
            if (i == 0 && j == 0) continue
            var bestWithoutMacro: PatternAlignment? = null
            var bestWithMacro: PatternAlignment? = null

            fun consider(previousI: Int, previousJ: Int, cost: Int) {
                val previousWithoutMacro = withoutMacro[previousI][previousJ]
                if (previousWithoutMacro.distance < ALIGNMENT_INFINITY) {
                    val candidate = PatternAlignment(
                        previousWithoutMacro.distance + cost,
                        previousWithoutMacro.macroUses
                    )
                    if (candidate.isBetterThan(bestWithoutMacro)) bestWithoutMacro = candidate
                }

                val previousWithMacro = withMacro[previousI][previousJ]
                if (previousWithMacro.distance < ALIGNMENT_INFINITY) {
                    val candidate = PatternAlignment(
                        previousWithMacro.distance + cost,
                        previousWithMacro.macroUses
                    )
                    if (candidate.isBetterThan(bestWithMacro)) bestWithMacro = candidate
                }
            }

            fun considerMacro(previousI: Int, previousJ: Int, shape: String) {
                // A catalogue entry is explained by exactly one atomic macro. Keeping
                // unused/used layers avoids the delete-all + insert-all degeneracy of
                // repeated zero-extension gap runs and implements the one-swap model.
                val previous = withoutMacro[previousI][previousJ]
                if (previous.distance >= ALIGNMENT_INFINITY) return
                val candidateDistance = previous.distance + 1
                if (
                    bestWithMacro != null &&
                    candidateDistance > bestWithMacro!!.distance
                ) return

                val macro = PatternMacroUse(
                    sourceStart = previousI,
                    sourceEnd = i,
                    targetStart = previousJ,
                    targetEnd = j,
                    sourceSymbols = source.subList(previousI, i).toList(),
                    targetSymbols = target.subList(previousJ, j).toList(),
                    shape = shape
                )
                val candidate =
                    PatternAlignment(candidateDistance, previous.macroUses + macro)
                if (candidate.isBetterThan(bestWithMacro)) bestWithMacro = candidate
            }

            if (i > 0) consider(i - 1, j, 1)
            if (j > 0) consider(i, j - 1, 1)
            if (i > 0 && j > 0)
                consider(i - 1, j - 1, if (source[i - 1] == target[j - 1]) 0 else 1)

            when (pattern) {
                RepairPattern.ADJACENT_TRANSPOSITION -> {
                    if (
                        i >= 2 && j >= 2 &&
                        source[i - 2] != source[i - 1] &&
                        source[i - 2] == target[j - 1] &&
                        source[i - 1] == target[j - 2]
                    ) considerMacro(i - 2, j - 2, "adjacent symbols")
                }

                RepairPattern.CHARACTER_REPETITION -> {
                    if (i >= 1) {
                        for (copies in 2..minOf(maxBlockSize, j)) {
                            if (target.regionIs(j - copies, copies, source[i - 1]))
                                considerMacro(i - 1, j - copies, "repeat 1->$copies")
                        }
                    }
                    if (j >= 1) {
                        for (copies in 2..minOf(maxBlockSize, i)) {
                            if (source.regionIs(i - copies, copies, target[j - 1]))
                                considerMacro(i - copies, j - 1, "unduplicate $copies->1")
                        }
                    }
                }

                RepairPattern.TANDEM_DUPLICATION -> {
                    for (blockSize in 2..maxBlockSize) {
                        if (
                            i >= blockSize && j >= 2 * blockSize &&
                            regionsEqual(source, i - blockSize, target, j - 2 * blockSize, blockSize) &&
                            regionsEqual(source, i - blockSize, target, j - blockSize, blockSize)
                        ) considerMacro(
                            i - blockSize,
                            j - 2 * blockSize,
                            "duplicate block=$blockSize"
                        )

                        if (
                            i >= 2 * blockSize && j >= blockSize &&
                            regionsEqual(source, i - 2 * blockSize, target, j - blockSize, blockSize) &&
                            regionsEqual(source, i - blockSize, target, j - blockSize, blockSize)
                        ) considerMacro(
                            i - 2 * blockSize,
                            j - blockSize,
                            "contract tandem block=$blockSize"
                        )
                    }
                }

                RepairPattern.BOUNDED_REVERSAL -> {
                    for (blockSize in 3..minOf(maxBlockSize, i, j)) {
                        val sourceStart = i - blockSize
                        val targetStart = j - blockSize
                        if (
                            !regionsEqual(source, sourceStart, target, targetStart, blockSize) &&
                            reversedRegionsEqual(source, sourceStart, target, targetStart, blockSize)
                        ) considerMacro(sourceStart, targetStart, "reverse block=$blockSize")
                    }
                }

                RepairPattern.ADJACENT_BLOCK_SWAP -> {
                    for (leftSize in 1..maxBlockSize) {
                        for (rightSize in 1..maxBlockSize) {
                            val totalSize = leftSize + rightSize
                            if (totalSize < 3 || i < totalSize || j < totalSize) continue
                            val sourceStart = i - totalSize
                            val targetStart = j - totalSize
                            if (
                                !regionsEqual(source, sourceStart, target, targetStart, totalSize) &&
                                regionsEqual(
                                    source, sourceStart,
                                    target, targetStart + rightSize,
                                    leftSize
                                ) &&
                                regionsEqual(
                                    source, sourceStart + leftSize,
                                    target, targetStart,
                                    rightSize
                                )
                            ) considerMacro(
                                sourceStart,
                                targetStart,
                                "swap adjacent blocks=$leftSize+$rightSize"
                            )
                        }
                    }
                }

                RepairPattern.ARBITRARY_SYMBOL_SWAP -> {
                    for (span in 3..minOf(i, j)) {
                        val sourceStart = i - span
                        val targetStart = j - span
                        if (
                            source[sourceStart] != source[i - 1] &&
                            source[sourceStart] == target[j - 1] &&
                            source[i - 1] == target[targetStart] &&
                            regionsEqual(
                                source, sourceStart + 1,
                                target, targetStart + 1,
                                span - 2
                            )
                        ) considerMacro(
                            sourceStart,
                            targetStart,
                            "swap distance=${span - 1}"
                        )
                    }
                }

                RepairPattern.WRAP_OR_UNWRAP -> {
                    // source body -> left + body + right
                    for (bodySize in 1..minOf(i, j - 2)) {
                        val sourceStart = i - bodySize
                        val targetStart = j - bodySize - 2
                        if (
                            targetStart >= 0 &&
                            regionsEqual(source, sourceStart, target, targetStart + 1, bodySize)
                        ) considerMacro(
                            sourceStart,
                            targetStart,
                            "wrap body=$bodySize"
                        )
                    }

                    // left + body + right -> target body
                    for (bodySize in 1..minOf(j, i - 2)) {
                        val sourceStart = i - bodySize - 2
                        val targetStart = j - bodySize
                        if (
                            sourceStart >= 0 &&
                            regionsEqual(source, sourceStart + 1, target, targetStart, bodySize)
                        ) considerMacro(
                            sourceStart,
                            targetStart,
                            "unwrap body=$bodySize"
                        )
                    }
                }

                RepairPattern.MULTI_SYMBOL_MERGE_OR_SPLIT -> {
                    if (i >= 1) {
                        for (width in 2..minOf(maxBlockSize, j)) {
                            val sourceSymbol = source[i - 1]
                            if ((j - width until j).none { target[it] == sourceSymbol })
                                considerMacro(i - 1, j - width, "split 1->$width")
                        }
                    }
                    if (j >= 1) {
                        for (width in 2..minOf(maxBlockSize, i)) {
                            val targetSymbol = target[j - 1]
                            if ((i - width until i).none { source[it] == targetSymbol })
                                considerMacro(i - width, j - 1, "merge $width->1")
                        }
                    }
                }

                RepairPattern.AFFINE_GAP_RUN -> {
                    for (runLength in 2..j)
                        considerMacro(i, j - runLength, "insert run=$runLength")
                    for (runLength in 2..i)
                        considerMacro(i - runLength, j, "delete run=$runLength")
                }
            }

            withoutMacro[i][j] =
                bestWithoutMacro ?: error("No vanilla alignment path to ($i, $j)")
            bestWithMacro?.let { withMacro[i][j] = it }
        }
    }

    val vanilla = withoutMacro[source.size][target.size]
    val augmented = withMacro[source.size][target.size]
    return if (augmented.distance < vanilla.distance) augmented else vanilla
}

private fun List<String>.regionIs(start: Int, length: Int, symbol: String): Boolean {
    if (start < 0 || start + length > size) return false
    for (offset in 0 until length)
        if (this[start + offset] != symbol) return false
    return true
}

private fun regionsEqual(
    first: List<String>,
    firstStart: Int,
    second: List<String>,
    secondStart: Int,
    length: Int
): Boolean {
    if (
        firstStart < 0 || secondStart < 0 ||
        firstStart + length > first.size ||
        secondStart + length > second.size
    ) return false

    for (offset in 0 until length)
        if (first[firstStart + offset] != second[secondStart + offset]) return false
    return true
}

private fun reversedRegionsEqual(
    first: List<String>,
    firstStart: Int,
    second: List<String>,
    secondStart: Int,
    length: Int
): Boolean {
    if (
        firstStart < 0 || secondStart < 0 ||
        firstStart + length > first.size ||
        secondStart + length > second.size
    ) return false

    for (offset in 0 until length)
        if (first[firstStart + offset] != second[secondStart + length - offset - 1])
            return false
    return true
}

private fun printRepairPatternReport(
    processed: Int,
    recognized: Int,
    rejected: Int,
    dfaNull: Int,
    inRadiusRejections: Int,
    classifiable: Int,
    coveredByRuleFreePattern: Int,
    coveredIncludingRuleCandidates: Int,
    failures: Int,
    matchesByPattern: Map<RepairPattern, List<PatternExample>>,
    primaryCounts: Map<RepairPattern, Int>,
    unmatched: List<CataloguedRepair>,
    examplesPerPattern: Int,
    maxBlockSize: Int
) {
    fun percent(numerator: Int, denominator: Int): String =
        if (denominator == 0) "0.0%"
        else String.format(Locale.ROOT, "%.1f%%", 100.0 * numerator / denominator)

    println()
    println("Unrecognized repair-pattern catalogue")
    println("====================================")
    println("Processed: $processed")
    println("DFA recognized: $recognized")
    println("DFA rejected (!dfaRecognized): $rejected")
    println("  Null DFA results (classifiable when outside-radius): $dfaNull")
    println("  In-radius rejection diagnostics: $inRadiusRejections")
    println("  Outside-radius, classifiable repairs: $classifiable")
    println(
        "  Covered by a rule-free structural family: $coveredByRuleFreePattern " +
                "(${percent(coveredByRuleFreePattern, classifiable)} classifiable, " +
                "${percent(coveredByRuleFreePattern, rejected)} of !dfaRecognized)"
    )
    println(
        "  Covered including rule-inventory candidates: $coveredIncludingRuleCandidates " +
                "(${percent(coveredIncludingRuleCandidates, classifiable)} classifiable, " +
                "${percent(coveredIncludingRuleCandidates, rejected)} of !dfaRecognized)"
    )
    println(
        "  Still unmatched: ${unmatched.size} " +
                "(${percent(unmatched.size, classifiable)} classifiable, " +
                "${percent(unmatched.size, rejected)} of !dfaRecognized)"
    )
    println("Processing failures (not counted as rejection): $failures")
    println("Bounded-pattern parameter: B=$maxBlockSize")
    println("Every augmented distance admits exactly one unit-cost atomic macro.")
    println("Any-match columns overlap; Primary uses the enum's specificity order.")
    println(
        "Rule-inventory candidates assume each observed mapping is admitted; " +
                "Observed rules estimates rho."
    )
    println("Weighted 1-to-1 substitutions need a confusion relation/cost and are not inferred.")
    println("The balanced input sampler is randomized, so counts can vary between runs.")
    println()
    println(
        "| Pattern | Evidence | Any-match repairs | % !dfaRecognized | Primary repairs | " +
                "Observed rules | Automaton complexity |"
    )
    println("| --- | --- | ---: | ---: | ---: | ---: | --- |")

    val orderedPatterns = RepairPattern.entries.sortedWith(
        compareByDescending<RepairPattern> { matchesByPattern.getValue(it).size }
            .thenBy { it.ordinal }
    )
    orderedPatterns.forEach { pattern ->
        val examples = matchesByPattern.getValue(pattern)
        val count = examples.size
        val evidence =
            if (pattern.requiresRuleInventory) "rule-inventory candidate" else "rule-free structural"
        val observedRules =
            if (pattern.requiresRuleInventory)
                examples.asSequence()
                    .flatMap { it.match.macroUses.asSequence() }
                    .map { it.inventoryRuleKey(pattern) }
                    .distinct()
                    .count()
                    .toString()
            else "-"
        println(
            "| ${pattern.displayName} | $evidence | $count | ${percent(count, rejected)} | " +
                    "${primaryCounts.getValue(pattern)} | $observedRules | " +
                    "${pattern.automatonComplexity} |"
        )
    }

    println()
    println("Shape counts")
    println("------------")
    println("(Counts below are macro uses on the selected one-macro alignments.)")
    orderedPatterns.forEach { pattern ->
        val examples = matchesByPattern.getValue(pattern)
        if (examples.isEmpty()) return@forEach
        val shapes = examples.asSequence()
            .flatMap { it.match.macroUses.asSequence() }
            .groupingBy { it.shape }
            .eachCount()
            .entries
            .sortedWith(compareByDescending<Map.Entry<String, Int>> { it.value }.thenBy { it.key })
            .take(8)
            .joinToString(", ") { "${it.key}: ${it.value}" }
        println("${pattern.displayName}: $shapes")
    }

    if (examplesPerPattern == 0) return

    println()
    println("Token-level examples")
    println("--------------------")
    orderedPatterns.forEach { pattern ->
        val examples = matchesByPattern.getValue(pattern)
            .distinctBy {
                it.match.macroUses.joinToString("|") { use -> use.shape + ":" + use.render() }
            }
            .take(examplesPerPattern)
        if (examples.isEmpty()) return@forEach

        println()
        println(pattern.displayName)
        examples.forEach { example ->
            val repair = example.repair
            println(
                "  #${repair.id}: d=${repair.vanillaDistance}->" +
                        "${example.match.augmentedDistance}, radius=${repair.radius}"
            )
            println(
                "    macro: ${
                    compactForReport(example.match.macroUses.joinToString("; ") { it.render() })
                }"
            )
            println("    tokens: ${compactForReport(repair.brokenTokens)}")
            println("         -> ${compactForReport(repair.fixedTokens)}")
            println("    source: ${compactForReport(repair.brokenCode)}")
            println("    repair: ${compactForReport(repair.fixedCode)}")
        }
    }

    if (unmatched.isNotEmpty()) {
        println()
        println("Unmatched examples")
        unmatched.take(examplesPerPattern).forEach { repair ->
            println("  #${repair.id}: d=${repair.vanillaDistance}, radius=${repair.radius}")
            println("    tokens: ${compactForReport(repair.brokenTokens)}")
            println("         -> ${compactForReport(repair.fixedTokens)}")
        }
    }
}

private fun compactForReport(text: String, maxLength: Int = 220): String {
    val compact = text.replace(Regex("\\s+"), " ").trim()
    return if (compact.length <= maxLength) compact else compact.take(maxLength - 3) + "..."
}

private fun testRepairPatternClassifier() {
    fun assertCovered(
        pattern: RepairPattern,
        source: List<String>,
        target: List<String>,
        expectedDistance: Int = 1
    ) {
        val vanillaDistance = levenshtein(source, target)
        val alignment = patternAwareAlignment(
            source,
            target,
            pattern,
            DEFAULT_MAX_PATTERN_BLOCK
        )
        check(alignment.distance == expectedDistance) {
            "$pattern: expected augmented distance $expectedDistance, got " +
                    "${alignment.distance} for $source -> $target"
        }
        check(alignment.distance < vanillaDistance) {
            "$pattern did not improve vanilla distance $vanillaDistance"
        }
        check(alignment.macroUses.isNotEmpty()) {
            "$pattern improved the distance without recording a macro"
        }
        check(alignment.macroUses.size == 1) {
            "$pattern used ${alignment.macroUses.size} macros; the catalogue permits exactly one"
        }
    }

    fun assertNotImproved(
        pattern: RepairPattern,
        source: List<String>,
        target: List<String>
    ) {
        val vanillaDistance = levenshtein(source, target)
        val alignment = patternAwareAlignment(
            source,
            target,
            pattern,
            DEFAULT_MAX_PATTERN_BLOCK
        )
        check(alignment.distance == vanillaDistance && alignment.macroUses.isEmpty()) {
            "$pattern unexpectedly improved $source -> $target from " +
                    "$vanillaDistance to ${alignment.distance}: ${alignment.macroUses}"
        }
    }

    assertCovered(
        RepairPattern.ADJACENT_TRANSPOSITION,
        listOf("a", "b"),
        listOf("b", "a")
    )
    assertCovered(
        RepairPattern.ADJACENT_TRANSPOSITION,
        listOf("a", "b", "x"),
        listOf("b", "a", "y"),
        expectedDistance = 2
    )
    assertCovered(
        RepairPattern.CHARACTER_REPETITION,
        listOf("x"),
        listOf("x", "x", "x")
    )
    assertCovered(
        RepairPattern.TANDEM_DUPLICATION,
        listOf("a", "b"),
        listOf("a", "b", "a", "b")
    )
    assertCovered(
        RepairPattern.BOUNDED_REVERSAL,
        listOf("a", "b", "c", "d"),
        listOf("d", "c", "b", "a")
    )
    assertCovered(
        RepairPattern.ADJACENT_BLOCK_SWAP,
        listOf("a", "b", "c"),
        listOf("b", "c", "a")
    )
    assertCovered(
        RepairPattern.ARBITRARY_SYMBOL_SWAP,
        listOf("a", "x", "y", "b"),
        listOf("b", "x", "y", "a")
    )
    assertCovered(
        RepairPattern.WRAP_OR_UNWRAP,
        listOf("x", "y"),
        listOf("left", "x", "y", "right")
    )
    assertCovered(
        RepairPattern.MULTI_SYMBOL_MERGE_OR_SPLIT,
        listOf("x"),
        listOf("y", "z")
    )
    assertCovered(
        RepairPattern.AFFINE_GAP_RUN,
        listOf("a"),
        listOf("a", "x", "y")
    )
    assertNotImproved(
        RepairPattern.AFFINE_GAP_RUN,
        listOf("a", "b", "c", "d"),
        listOf("w", "x", "y", "z")
    )
    assertNotImproved(
        RepairPattern.CHARACTER_REPETITION,
        listOf("x"),
        listOf("x", "x")
    )
    assertNotImproved(
        RepairPattern.MULTI_SYMBOL_MERGE_OR_SPLIT,
        listOf("x"),
        listOf("x", "y", "z")
    )

    val twoSwaps = patternAwareAlignment(
        listOf("a", "b", "c", "d", "e", "f"),
        listOf("c", "b", "a", "f", "e", "d"),
        RepairPattern.ARBITRARY_SYMBOL_SWAP,
        DEFAULT_MAX_PATTERN_BLOCK
    )
    check(twoSwaps.distance == 3 && twoSwaps.macroUses.size == 1) {
        "The one-swap family admitted multiple swaps: $twoSwaps"
    }

    println(
        "Repair-pattern classifier self-test passed " +
                "(${RepairPattern.entries.size} families plus negative guards)."
    )
}
