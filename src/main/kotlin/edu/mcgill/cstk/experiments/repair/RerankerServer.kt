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
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.net.http.HttpTimeoutException
import java.nio.charset.StandardCharsets
import java.time.Duration
import java.util.concurrent.CompletableFuture
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

private const val PORT = 8000

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

val rerankerVersion = "7700"

fun startRerankerServer() {
  if (::server.isInitialized) return

  server = HttpServer.create(InetSocketAddress(PORT), 0).apply {
    createContext("/") { ex ->
      ex.sendFile(File(webRoot, "reranker_$rerankerVersion.html"), "text/html; charset=utf-8")
    }

    createContext("/reranker_$rerankerVersion.js") { ex ->
      ex.sendFile(File(webRoot, "reranker_$rerankerVersion.js"), "text/javascript; charset=utf-8")
    }

    createContext("/reranker_$rerankerVersion.safetensors") { ex ->
      ex.sendFile(File(webRoot, "reranker_$rerankerVersion.safetensors"), "application/octet-stream")
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

    executor = null
    start()
  }

  if (Desktop.isDesktopSupported()) {
    Desktop.getDesktop().browse(URI("http://localhost:$PORT/"))
  }
}

/**
 * Rerank docs for a query. Returns the docs sorted descending by model score.
 */
@Synchronized
fun rerankGPUNew(query: String, docs: List<String>, timeoutSec: Long = 30): List<String> {
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

    // One SSE event, then close. EventSource auto-reconnects because of retry: 0.
    ex.responseBody.use { os ->
      val msg = "retry: 0\ndata: $body\n\n"
      os.write(msg.toByteArray(StandardCharsets.UTF_8))
      os.flush()
    }

    return future.get(timeoutSec, TimeUnit.SECONDS).map { it.uncharify() }
  } finally { pending = null }
}