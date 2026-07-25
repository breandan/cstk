package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.parsing.levenshteinAlign
import ai.hypergraph.kaliningraph.parsing.paintANSIColors
import ai.hypergraph.kaliningraph.parsing.patchSize
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import com.beust.klaxon.Klaxon
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit
import kotlin.streams.asStream
import kotlin.time.TimeSource


private fun envInt(name: String, default: Int): Int =
    System.getenv(name)?.toIntOrNull() ?: default

private class OpenAIRequestException(val statusCode: Int, message: String): RuntimeException(message)
private class OpenAIEmptyResponseException(message: String): RuntimeException(message)

private fun envBool(name: String): Boolean =
    System.getenv(name)?.trim()?.lowercase() in setOf("1", "true", "yes", "y")

private fun envOptionalDouble(name: String): String? =
    System.getenv(name)?.trim()?.takeIf { it.isNotEmpty() }?.also {
        require(it.toDoubleOrNull() != null) { "$name must be a number, got: $it" }
    }

private fun jsonEscape(s: String): String =
    buildString(s.length + 16) {
        s.forEach {
            when (it) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (it.code < 0x20) append("\\u%04x".format(it.code)) else append(it)
            }
        }
    }

private fun chatGPTRepairPrompt(brokenTokens: String) = """
You are repairing a Python syntax error in tokenized form.

The input and output are whitespace-separated lexer tokens, not source code.
Identifiers and literals are abstracted as NAME, NUMBER, and STRING.
Structural tokens such as NEWLINE, INDENT, DEDENT, and ENDMARKER must remain tokens.

Return exactly one repaired token sequence in the same whitespace-separated token format.
Return the complete repaired sequence, including all unchanged tokens from the input.
Do not return only the changed token, only the edit, or a patch.
Make the smallest syntactic repair that is most likely to be the author's intended code.
Do not return markdown, XML tags, explanations, alternatives, or source code.

Broken token sequence:
$brokenTokens
""".trimIndent()

private fun normalizeChatGPTRepair(raw: String): String {
    val tagged = Regex("""(?is)<repair>(.*?)</repair>""").find(raw)?.groupValues?.get(1)
    val cleaned = (tagged ?: raw)
        .lines()
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("```") }
        .joinToString(" ")
        .removePrefix("Repair:")
        .removePrefix("repair:")
        .trim()
        .trim('"', '\'', '`')

    val normalized = cleaned.replace(Regex("\\s+"), " ").trim()
    if (normalized.isEmpty()) throw OpenAIEmptyResponseException("OpenAI response contained no visible repaired tokens.")

    return normalized.addNewLineIfMissing()
}

private fun extractOpenAIText(json: String): String {
    val parsed = Klaxon().parseJsonObject(json.reader())
    parsed.string("output_text")?.let { return it }

    val output = parsed.array<Any>("output") ?: return ""
    return output.joinToString("\n") { item ->
        val itemMap = item as? Map<*, *> ?: return@joinToString ""
        val content = itemMap["content"] as? List<*> ?: return@joinToString ""
        content.joinToString("\n") { contentItem ->
            val contentMap = contentItem as? Map<*, *> ?: return@joinToString ""
            (contentMap["text"] ?: contentMap["content"] ?: "").toString()
        }
    }.trim()
}

private fun openAIResponseSummary(json: String): String = try {
    val parsed = Klaxon().parseJsonObject(json.reader())
    val status = parsed.string("status")
    val incompleteReason = parsed.obj("incomplete_details")?.string("reason")
    listOfNotNull(
        status?.let { "status=$it" },
        incompleteReason?.let { "incomplete_reason=$it" },
    ).joinToString(", ").let { if (it.isEmpty()) "" else " ($it)" }
} catch (_: Exception) {
    ""
}

private fun requestChatGPTRepair(
    brokenTokens: String,
    client: OkHttpClient,
    apiKey: String,
    model: String,
    maxOutputTokens: Int,
    reasoningEffort: String,
    temperature: String?,
    debugResponse: Boolean,
): String {
    val reasoning =
        if (reasoningEffort.equals("none", ignoreCase = true)) ""
        else ""","reasoning":{"effort":"${jsonEscape(reasoningEffort)}"}"""
    val temperatureParam = temperature?.let { ""","temperature":$it""" } ?: ""

    val body = """
{
  "model": "${jsonEscape(model)}",
  "input": [
    {
      "role": "user",
      "content": "${jsonEscape(chatGPTRepairPrompt(brokenTokens))}"
    }
  ],
  "max_output_tokens": $maxOutputTokens,
  "store": false$temperatureParam$reasoning
}
""".trimIndent()

    val request = Request.Builder()
        .url("https://api.openai.com/v1/responses")
        .header("Authorization", "Bearer $apiKey")
        .header("Content-Type", "application/json")
        .post(body.toRequestBody("application/json".toMediaType()))
        .build()

    client.newCall(request).execute().use { response ->
        val responseBody = response.body.string()
        if (!response.isSuccessful) throw OpenAIRequestException(response.code, "OpenAI API ${response.code}: $responseBody")
        val extractedText = extractOpenAIText(responseBody)
        if (extractedText.isBlank()) {
            val rawSuffix = if (debugResponse) "\nRaw response:\n$responseBody" else " Set OPENAI_DEBUG_RESPONSE=true to print the raw response."
            throw OpenAIEmptyResponseException(
                "OpenAI response had no visible output text${openAIResponseSummary(responseBody)}. " +
                        "Try increasing OPENAI_MAX_OUTPUT_TOKENS or setting OPENAI_REASONING_EFFORT=none.$rawSuffix"
            )
        }
        return normalizeChatGPTRepair(extractedText)
    }
}

fun evaluateChatGPTRepairPrecision() {
    val apiKey = System.getenv("OPENAI_API_KEY") ?: error("Set OPENAI_API_KEY before running the ChatGPT repair harness.")
    val model = System.getenv("OPENAI_MODEL") ?: "gpt-5.4-mini"
    val reasoningEffort = System.getenv("OPENAI_REASONING_EFFORT") ?: "none"
    val limit = envInt("CHATGPT_REPAIR_LIMIT", Int.MAX_VALUE)
    val maxOutputTokens = envInt("OPENAI_MAX_OUTPUT_TOKENS", 1024)
    val timeoutSeconds = envInt("OPENAI_TIMEOUT_SECONDS", 60).toLong()
    val temperature = envOptionalDouble("OPENAI_TEMPERATURE")
    val debugResponse = envBool("OPENAI_DEBUG_RESPONSE")
    val dataset = sizeAndDistBalancedRepairsUnminimized
    val p1ByLenAndDist = mutableMapOf<Pair<Int, Int>, S2PMetrics>()
    var seen = 0
    var total = 0
    var correct = 0
    var failed = 0

    val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
        .callTimeout(timeoutSeconds + 5, TimeUnit.SECONDS)
        .build()

    println("Running ChatGPT tokenized Python repair harness")
    println("Model: $model, reasoning_effort: $reasoningEffort, temperature: ${temperature ?: "model default"}, max_output_tokens: $maxOutputTokens, limit: ${if (limit == Int.MAX_VALUE) "all" else limit}")
    println("Set OPENAI_MODEL, OPENAI_REASONING_EFFORT=none|low|medium, OPENAI_TEMPERATURE, OPENAI_MAX_OUTPUT_TOKENS, CHATGPT_REPAIR_LIMIT, OPENAI_TIMEOUT_SECONDS to override defaults.")

    fun reportRunningStats() {
        println()
        println("Precision@1\n===========")
        println(p1ByLenAndDist.summarizeLenAndDist())
        println("Overall Precision@1: $correct / $total = ${if (total == 0) 0.0 else correct.toDouble() / total}")
        if (failed > 0) println("Skipped failed requests: $failed / $seen")
        println()
    }

    dataset.asStream().limit(limit.toLong()).forEach { (brokeStr, fixedStr) ->
        seen++
        val start = TimeSource.Monotonic.markNow()
        val brokeToks = brokeStr.tokenizeByWhitespace()
        val fixedToks = fixedStr.tokenizeByWhitespace()
        val trueLevDist = levenshteinAlign(brokeToks, fixedToks).patchSize()
        val lenBucket = (brokeToks.size / LEN_BUCKET_INTERVAL) * LEN_BUCKET_INTERVAL

        val predicted = try {
            requestChatGPTRepair(
                brokenTokens = brokeStr,
                client = client,
                apiKey = apiKey,
                model = model,
                maxOutputTokens = maxOutputTokens,
                reasoningEffort = reasoningEffort,
                temperature = temperature,
                debugResponse = debugResponse,
            )
        } catch (t: Exception) {
            failed++
            println("[$seen] | len=$lenBucket | Δ=$trueLevDist | request_failed | ${start.elapsedNow().ms3()}")
            println("OpenAI request failed; skipping Precision@1 denominator: ${t.message}")
            if (t is OpenAIRequestException && t.statusCode in 400..499 && t.statusCode != 429) {
                error("Stopping because OpenAI rejected the request shape. Fix the request/model configuration before continuing.")
            }
            reportRunningStats()
            return@forEach
        }

        val bucket = p1ByLenAndDist.getOrPut(lenBucket to trueLevDist) { S2PMetrics() }
        total++
        bucket.total++

        val matched = predicted == fixedStr.addNewLineIfMissing()
        if (matched) {
            correct++
            bucket.top1++
        }

        println("[$seen] | evaluated=$total | len=$lenBucket | Δ=$trueLevDist | match=$matched | ${start.elapsedNow().ms3()}")
        println("Broken:    $brokeStr")
        println("Predicted: $predicted")
        println("Expected:  $fixedStr")
        if (!matched) println("Diff:      ${levenshteinAlign(brokeStr.tokenizeByWhitespace(), predicted.tokenizeByWhitespace()).paintANSIColors()}")
        reportRunningStats()
    }

    reportRunningStats()
}