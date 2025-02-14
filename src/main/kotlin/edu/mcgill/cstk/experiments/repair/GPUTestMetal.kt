package edu.mcgill.cstk.experiments.repair

import com.sun.jna.*
import org.intellij.lang.annotations.Language
import java.io.File
import java.security.MessageDigest
import kotlin.random.Random
import kotlin.system.measureTimeMillis
import kotlin.time.TimeSource

fun main() {
  val clock = TimeSource.Monotonic.markNow()
  val t = 2000
  val arr = IntArray(t * t) { Random.nextInt(10) }
  val dylib = File("libMetalBridge.dylib")

  @Language("c++") // close enough
  val mpsSrc = """#include <metal_stdlib>
  using namespace metal;
  kernel void mat_mul(
    const device int* A[[buffer(0)]],
    const device int* B[[buffer(1)]],
    device int* O[[buffer(2)]],
    uint i[[thread_position_in_grid]]
  ) {
    uint n=$t;
    if(i<n*n){
      uint r=i/n, c=i%n; int s=0;
      for(uint k=0;k<n;k++) s+=A[r*n+k]*B[k*n+c];
      O[i]=s;
    }
  }"""

  val swiftSrc = """
import Foundation
import Metal
private var d: MTLDevice!, q: MTLCommandQueue!, p: MTLComputePipelineState!

@_cdecl("initMetalStuff")
public func initMetalStuff() {
  let metalSrc = #${"\"\"\""}
  $mpsSrc
  ${"\"\"\""}#
  d=MTLCreateSystemDefaultDevice()!
  q=d.makeCommandQueue()!
  let L=try! d.makeLibrary(source: metalSrc, options:nil)
  p=try! d.makeComputePipelineState(function:L.makeFunction(name:"mat_mul")!)
}

@_cdecl("imm")
public func imm(_ A: UnsafePointer<CInt>?, _ B: UnsafePointer<CInt>?, _ n: CInt, _ out: UnsafeMutablePointer<CInt>?) {
  let nn=Int(n), sz=nn*nn*4
  let BA=d.makeBuffer(bytes:A!, length:sz, options:[])!,
      BB=d.makeBuffer(bytes:B!, length:sz, options:[])!,
      BO=d.makeBuffer(length:sz, options:[])!,
      cb=q.makeCommandBuffer()!,
      enc=cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(p)
  enc.setBuffer(BA, offset:0, index:0)
  enc.setBuffer(BB, offset:0, index:1)
  enc.setBuffer(BO, offset:0, index:2)
  enc.dispatchThreads(MTLSizeMake(nn*nn,1,1), threadsPerThreadgroup:MTLSizeMake(1,1,1))
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); memcpy(out, BO.contents(), sz)
}""".trimIndent()

  val hash = md5(swiftSrc)
  val hashFile = File(".swiftHash")
  fun needsRebuild() = !dylib.exists() || !hashFile.exists() || hashFile.readText() != hash

  if (needsRebuild()) {
    File("MetalBridge.swift".also { println("Rebuilding: $it") }).writeText(swiftSrc)
    val cmd = "xcrun swiftc -emit-library MetalBridge.swift -o ${dylib.absolutePath} -module-name M " +
        "-Xlinker -install_name -Xlinker @rpath/libMetalBridge.dylib"
    if (cmd.exec() != 0) error("Failed to build Swift bridging code!")
    hashFile.writeText(hash)
  }

  val lib = (Native.load(dylib.absolutePath, MetalBridge::class.java) as MetalBridge).also { it.initMetalStuff() }

  println("Warmup took: ${clock.elapsedNow()}")

  val memA = Memory((arr.size * 4).toLong()).apply { write(0, arr, 0, arr.size) }
  val memB = Memory((arr.size * 4).toLong()).apply { write(0, arr, 0, arr.size) }
  val memOut = Memory((arr.size * 4).toLong())

  val gpuMs = measureTimeMillis { lib.imm(memA, memB, t, memOut) }
  println("GPU took ${gpuMs}ms")

  val outCPU = IntArray(arr.size)
  val cpuMs = measureTimeMillis {
    for (r in 0 until t)
      for (c in 0 until t) {
        var s = 0; for (k in 0 until t) s += arr[r * t + k] * arr[k * t + c]
        outCPU[r * t + c] = s
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
  fun imm(a: Pointer, b: Pointer, n: Int, out: Pointer)
}

fun md5(s: String) = MessageDigest.getInstance("MD5").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }
fun String.exec() = ProcessBuilder(split(" ")).inheritIO().start().waitFor()