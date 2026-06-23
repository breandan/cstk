package edu.mcgill.cstk.experiments.repair

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import edu.mcgill.cstk.experiments.probing.charify
import edu.mcgill.cstk.experiments.probing.uncharify
import java.awt.Desktop
import java.io.File
import java.net.InetSocketAddress
import java.net.URI
import java.net.URLDecoder
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.concurrent.CompletableFuture
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

private const val PORT = 8000
private const val RERANKER_MODEL_VERSION = "1100"

private val webRoot = "scripts/reranker_v3/"
private val streams = LinkedBlockingQueue<HttpExchange>()

private data class PendingRequest(
  val id: Long,
  val future: CompletableFuture<List<String>>
)

@Volatile private var pending: PendingRequest? = null
private val nextRequestId = AtomicLong(1)

private lateinit var server: HttpServer

private fun HttpExchange.sendBytes(code: Int, mime: String, body: ByteArray) {
  responseHeaders.add("Content-Type", mime)
  sendResponseHeaders(code, body.size.toLong())
  responseBody.use { it.write(body) }
}

private fun HttpExchange.sendText(code: Int, mime: String, text: String) =
  sendBytes(code, mime, text.toByteArray(StandardCharsets.UTF_8))

private fun HttpExchange.sendFile(file: File, mime: String) {
  if (!file.isFile) {
    sendText(404, "text/plain; charset=utf-8", "Missing file: ${file.absolutePath}")
    return
  }

  responseHeaders.add("Content-Type", mime)
  responseHeaders.add("Cache-Control", "no-store")
  sendResponseHeaders(200, file.length())
  file.inputStream().use { input ->
    responseBody.use { output -> input.copyTo(output) }
  }
}

private fun enc(s: String): String =
  URLEncoder.encode(s, StandardCharsets.UTF_8.name())

private fun dec(s: String): String =
  URLDecoder.decode(s, StandardCharsets.UTF_8.name())

private fun parseForm(body: String): Map<String, List<String>> {
  if (body.isBlank()) return emptyMap()

  val out = linkedMapOf<String, MutableList<String>>()
  for (part in body.split('&')) {
    if (part.isEmpty()) continue
    val eq = part.indexOf('=')
    val k = if (eq >= 0) dec(part.substring(0, eq)) else dec(part)
    val v = if (eq >= 0) dec(part.substring(eq + 1)) else ""
    out.getOrPut(k) { mutableListOf() }.add(v)
  }
  return out
}

private fun buildJobBody(id: Long, query: String, docs: List<String>): String =
  buildString {
    append("id=").append(enc(id.toString()))
    append("&q=").append(enc(query))
    for (doc in docs) append("&d=").append(enc(doc))
  }

fun startRerankerServer() {
  if (::server.isInitialized) return

  server = HttpServer.create(InetSocketAddress(PORT), 0).apply {
    createContext("/") { ex ->
      ex.sendFile(File(webRoot, "reranker_server.html"), "text/html; charset=utf-8")
    }

    createContext("/reranker_${RERANKER_MODEL_VERSION}.js") { ex ->
      ex.sendFile(File(webRoot, "reranker_${RERANKER_MODEL_VERSION}.js"), "text/javascript; charset=utf-8")
    }

    createContext("/reranker_${RERANKER_MODEL_VERSION}.safetensors") { ex ->
      ex.sendFile(File(webRoot, "reranker_${RERANKER_MODEL_VERSION}.safetensors"), "application/octet-stream")
    }

    // Browser opens EventSource("/stream").
    // We hold the exchange until a JVM caller has work to send.
    createContext("/stream") { ex ->
      streams.put(ex)
    }

    createContext("/result") { ex ->
      val body = ex.requestBody.readAllBytes().toString(StandardCharsets.UTF_8)
      val form = parseForm(body)

      val id = form["id"]?.firstOrNull()?.toLongOrNull()
      val docs = form["d"].orEmpty()

      val p = pending
      if (p != null && id == p.id) {
        p.future.complete(docs)
        ex.sendResponseHeaders(204, -1)
      } else {
        ex.sendText(409, "text/plain; charset=utf-8", "No matching pending request")
      }
    }

    createContext("/error") { ex ->
      val body = ex.requestBody.readAllBytes().toString(StandardCharsets.UTF_8)
      val form = parseForm(body)

      val id = form["id"]?.firstOrNull()?.toLongOrNull()
      val message = form["message"]?.firstOrNull() ?: "Unknown browser-side reranker error"

      val p = pending
      if (p != null && id == p.id) {
        p.future.completeExceptionally(IllegalStateException(message))
        ex.sendResponseHeaders(204, -1)
      } else {
        ex.sendText(409, "text/plain; charset=utf-8", "No matching pending request")
      }
    }

    executor = null
    start()
  }

  println("WebGPU reranker page: http://localhost:$PORT/")
  runCatching {
    if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
      Desktop.getDesktop().browse(URI("http://localhost:$PORT/"))
    }
  }.onFailure { println("Could not open browser automatically: ${it.message}") }
}

/**
 * Rerank docs for a query. Returns the docs sorted descending by model score.
 */
@Synchronized
fun rerankWGPU(query: String, docs: List<String>, timeoutSec: Long = 120): List<String> {
  if (docs.isEmpty()) return emptyList()
  startRerankerServer()

  val query = query.charify()
  val docs = docs
    .map { it.addNewLineIfMissing() }
    .map { it.charify() }

  val ex = streams.poll(timeoutSec, TimeUnit.SECONDS)
    ?: error("Browser did not connect to /stream in time (page not open, or model not loaded yet)")

  val id = nextRequestId.getAndIncrement()
  val future = CompletableFuture<List<String>>()
  pending = PendingRequest(id, future)

  try {
    val body = buildJobBody(id, query, docs)

    ex.responseHeaders.add("Content-Type", "text/event-stream; charset=utf-8")
    ex.responseHeaders.add("Cache-Control", "no-cache")
    ex.responseHeaders.add("Connection", "keep-alive")
    ex.sendResponseHeaders(200, 0)

    // One SSE event, then close. EventSource reconnects to provide the next long-poll slot.
    ex.responseBody.use { os ->
      val msg = "retry: 50\ndata: $body\n\n"
      os.write(msg.toByteArray(StandardCharsets.UTF_8))
      os.flush()
    }

    return future.get(timeoutSec, TimeUnit.SECONDS).map { it.uncharify() }
  } finally { pending = null }
}

fun rerankWGPU(query: String, docs: String, timeoutSec: Long = 120): List<String> =
  rerankWGPU(query, docs.lines().filter { it.isNotBlank() }, timeoutSec)

fun rerankGPUNew(query: String, docs: List<String>, timeoutSec: Long = 120): List<String> =
  rerankWGPU(query, docs, timeoutSec)
