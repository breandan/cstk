package edu.mcgill.cstk.experiments.repair

import com.sun.net.httpserver.*
import java.awt.Desktop
import java.io.File
import java.net.*
import java.nio.charset.StandardCharsets
import java.util.concurrent.*
import kotlin.system.measureTimeMillis

/**
./gradlew -q wgpuBarHillelRepair
*/
fun main() {
  startWGPUServer()

  val list = """
    [ NAME for NAME in NAME ] ]
    [ NAME for NAME , NAME in NAME ( NAME , NUMBER ) if ( NAME & ( NAME - NUMBER ) ]
  """.lines().filter { it.isNotEmpty() }

  measureTimeMillis {
    for (snippet in list) {
      println(send(snippet).lines().take(10).joinToString("\n"))
      println()
    }
  }.also { println("Total time: ${it / 1000.0}s") }

  stopWGPUServer()
}

private const val PORT = 8000
private val page by lazy { File(System.getProperty("user.home"), "tidyparse.html").readText() }
private val streams = LinkedBlockingQueue<HttpExchange>()
@Volatile private var resultWaiter: CompletableFuture<String>? = null
private lateinit var server: HttpServer

private fun HttpExchange.send(code: Int, mime: String, body: ByteArray) {
  responseHeaders.add("Content-Type", mime)
  sendResponseHeaders(code, body.size.toLong())
  responseBody.use { it.write(body) }
}

fun startWGPUServer() {
  if (::server.isInitialized) return
  server = HttpServer.create(InetSocketAddress(PORT), 0).apply {
    createContext("/") { it.send(200, "text/html", page.toByteArray()) }
    createContext("/stream") { ex -> streams.put(ex) }
    createContext("/result") { ex ->
      val txt = ex.requestBody.readAllBytes().toString(StandardCharsets.UTF_8).trim()
      resultWaiter?.complete(txt)
      ex.sendResponseHeaders(204, -1)
    }
    executor = null
    start()
  }

  Desktop.getDesktop().browse(URI("http://localhost:$PORT/"))
}

fun send(query: String, timeoutSec: Long = 30): String {
  val ex = streams.poll(timeoutSec, TimeUnit.SECONDS) ?: error("browser did not open /stream in time")
  resultWaiter = CompletableFuture()
  ex.responseHeaders.add("Content-Type", "text/event-stream")
  ex.sendResponseHeaders(200, 0)
  ex.responseBody.use { os -> os.write("retry: 0\ndata: $query\n\n".toByteArray()) }
  return resultWaiter!!.get(timeoutSec, TimeUnit.SECONDS)
}

fun stopWGPUServer() = if (::server.isInitialized) server.stop(0) else Unit