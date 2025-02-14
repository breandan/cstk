package edu.mcgill.cstk.experiments.repair

import com.sun.jna.*
import org.intellij.lang.annotations.Language
import java.io.File
import java.security.MessageDigest
import kotlin.math.ceil
import kotlin.math.log
import kotlin.random.Random
import kotlin.system.measureTimeMillis
import kotlin.time.TimeSource

fun main() {
  val clock = TimeSource.Monotonic.markNow()
  val t = 1000
  var arr = IntArray(t * t) { Random.nextInt(100) }
  val dylib = File("libMetalBridge.dylib")

  @Language("c++") // close enough
  val mpsSrc = """
  #include <metal_stdlib>
  using namespace metal;
  kernel void mat_mul(
    const device int* A[[buffer(0)]],
      device int* O[[buffer(1)]],
      constant uint& n[[buffer(2)]],
      uint i[[thread_position_in_grid]]
  ) {
    if (i < n * n) {
      uint r = i / n, c = i % n; int s = 0;
      for (uint k = 0; k < n; k++) s += A[r * n + k] * A[k * n + c];
      O[i] = s;
    }
  }

  inline int getBit(int value, uint bitIndex) { return (value >> bitIndex) & 1; }
  """

  @Language("swift")
  val swiftSrc = """
import Foundation
import Metal
private var dvc: MTLDevice!, mtq: MTLCommandQueue!, cps: MTLComputePipelineState!

@_cdecl("initMetalStuff")
public func initMetalStuff() {
  let metalSrc = #${"\"\"\""}$mpsSrc${"\"\"\""}#
  dvc = MTLCreateSystemDefaultDevice()!
  mtq = dvc.makeCommandQueue()!
  let lib = try! dvc.makeLibrary(source: metalSrc, options:nil)
  cps = try! dvc.makeComputePipelineState(function:lib.makeFunction(name:"mat_mul")!)
}

@_cdecl("imm")
public func imm(_ A: UnsafePointer<CInt>?, _ n: CInt, _ out: UnsafeMutablePointer<CInt>?) {
  let nn = Int(n), sz = nn * nn * 4, reps = Int(ceil(log2(Double(nn))))
  let BA = dvc.makeBuffer(bytes: A!, length: sz, options: [])!,
      BO = dvc.makeBuffer(length: sz, options: [])!
  for _ in 0..<reps {
    let cb = mtq.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(cps)

    enc.setBuffer(BA, offset: 0, index: 0)
    enc.setBuffer(BO, offset: 0, index: 1)
    var cpn = n; enc.setBytes(&cpn, length: MemoryLayout<CInt>.size, index: 2)
    enc.dispatchThreads(MTLSizeMake(nn * nn, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    memcpy(BA.contents(), BO.contents(), sz)
  }
  memcpy(out, BA.contents(), sz)
}""".trimIndent()

  val hash = md5(swiftSrc)
  val hashFile = File(".swiftHash")
  fun needsRebuild() = !dylib.exists() || !hashFile.exists() || hashFile.readText() != hash

  if (needsRebuild()) {
    val clock = TimeSource.Monotonic.markNow()
    File("MetalBridge.swift").writeText(swiftSrc)
    val cmd = "xcrun swiftc -emit-library MetalBridge.swift -o ${dylib.absolutePath} -module-name M " +
        "-Xlinker -install_name -Xlinker @rpath/libMetalBridge.dylib"
    if (cmd.exec() != 0) error("Failed to build Swift bridging code!")
    hashFile.writeText(hash)
    println("Finished rebuild in ${clock.elapsedNow()}")
  }

  val lib = (Native.load(dylib.absolutePath, MetalBridge::class.java) as MetalBridge).also { it.initMetalStuff() }

  println("Linking took: ${clock.elapsedNow()}")

  val memA = Memory((arr.size * 4).toLong()).apply { write(0, arr, 0, arr.size) }
  val memOut = Memory((arr.size * 4).toLong())

  val gpuMs = measureTimeMillis { lib.imm(memA, t, memOut) }
  println("GPU took ${gpuMs}ms")

  val outCPU = IntArray(arr.size)
  val cpuMs = measureTimeMillis {
    for (e in 0..<ceil(log(t.toDouble(), 2.0)).toInt()) {
      val temp = arr.copyOf()
      for (r in 0 until t) for (c in 0 until t) {
        var s = 0
        for (k in 0 until t) s += temp[r*t + k] * temp[k*t + c]
        outCPU[r*t + c] = s
      }
      arr = outCPU.copyOf()
    }
  }
  println("CPU took ${cpuMs}ms")

  val outGPU = memOut.getIntArray(0, arr.size)
  listOf(0, t - 1, t * (t - 1), t * t - 1).forEach {
    if (outGPU[it] != outCPU[it]) error("Mismatch @ $it: GPU=${outGPU[it]}, CPU=${outCPU[it]}")
  }
  println("GPU=CPU")
}

interface MetalBridge : Library {
  fun initMetalStuff()
  fun imm(a: Pointer, n: Int, out: Pointer)
}

fun md5(s: String) = MessageDigest.getInstance("MD5").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }
fun String.exec() = ProcessBuilder(split(" ")).inheritIO().start().waitFor()