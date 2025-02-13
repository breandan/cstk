package edu.mcgill.cstk.experiments.repair

import com.sun.jna.*
import java.io.File
import java.security.MessageDigest
import kotlin.random.Random
import kotlin.system.measureTimeMillis
import kotlin.time.TimeSource

fun main() {
  val clock = TimeSource.Monotonic.markNow()
  val t = 1400
  val arr = IntArray(t * t) { Random.nextInt(10) } // A and B are the same array for demo
  val (ml, dl) = File("mat_mul.metallib") to File("libMetalBridge.dylib")

  // 1) Metal shader: fill entire O[r*n + c]
  val ms = """
#include <metal_stdlib>
using namespace metal;
kernel void mat_mul(const device int* A[[buffer(0)]], const device int* B[[buffer(1)]],
                    device int* O[[buffer(2)]], uint i[[thread_position_in_grid]]) {
  uint n=$t;
  if(i<n*n){
    uint r=i/n, c=i % n;
    int s=0; for(uint k=0;k<n;k++) s+=A[r*n+k]*B[k*n+c];
    O[i]=s;
  }
}""".trim()

  // 2) Swift bridging: pass "out" pointer & copy entire NxN result back
  val bs = """
import Foundation
import Metal
private var d: MTLDevice!, q: MTLCommandQueue!, p: MTLComputePipelineState!
@_cdecl("initMetalStuff")
public func initMetalStuff(_ x: UnsafePointer<CChar>?) {
  d=MTLCreateSystemDefaultDevice()!; q=d.makeCommandQueue()!
  let L=try! d.makeLibrary(filepath:String(cString:x!))
  p=try! d.makeComputePipelineState(function:L.makeFunction(name:"mat_mul")!)
}
@_cdecl("metalMatMul")
public func metalMatMul(_ A: UnsafePointer<CInt>?, _ B: UnsafePointer<CInt>?, _ n: CInt, _ out: UnsafeMutablePointer<CInt>?) {
  let nn=Int(n), sz=nn*nn*4
  let BA=d.makeBuffer(bytes:A!,length:sz,options:[])!,
      BB=d.makeBuffer(bytes:B!,length:sz,options:[])!,
      BO=d.makeBuffer(length:sz,options:[])!,
      CB=q.makeCommandBuffer()!,
      CE=CB.makeComputeCommandEncoder()!
  CE.setComputePipelineState(p)
  CE.setBuffer(BA,offset:0,index:0)
  CE.setBuffer(BB,offset:0,index:1)
  CE.setBuffer(BO,offset:0,index:2)
  CE.dispatchThreads(MTLSizeMake(nn*nn,1,1),threadsPerThreadgroup:MTLSizeMake(1,1,1))
  CE.endEncoding(); CB.commit(); CB.waitUntilCompleted()
  memcpy(out,BO.contents(),sz)
}
""".trim()

  // 3) Check MD5 & rebuild if needed
  val (mh, bh) = md5(ms) to md5(bs)
  val (mhf, bhf) = File(".mh") to File(".bh")
  fun needsRebuild(f: File, hf: File, h: String) = !f.exists() || !hf.exists() || hf.readText() != h
  if (needsRebuild(ml, mhf, mh)) {
    File("m.metal").writeText(ms)
    "xcrun metal -c m.metal -o m.air".execOk()
    "xcrun metallib m.air -o ${ml.absolutePath}".execOk()
    mhf.writeText(mh)
  }
  if (needsRebuild(dl, bhf, bh)) {
    File("b.swift").writeText(bs)
    val cc = "xcrun swiftc -emit-library b.swift -o ${dl.absolutePath} -module-name M " +
        "-Xlinker -install_name -Xlinker @rpath/libMetalBridge.dylib"
    cc.execOk(); bhf.writeText(bh)
  }

  // 4) JNA
  val mb = Native.load(dl.absolutePath, MetalBridge::class.java) as MetalBridge
  mb.initMetalStuff(ml.absolutePath)

  println("Warmup took: ${clock.elapsedNow()}")

  // 5) Allocate GPU input + output
  val inA = Memory((arr.size * 4).toLong()).apply { write(0, arr, 0, arr.size) }
  val inB = Memory((arr.size * 4).toLong()).apply { write(0, arr, 0, arr.size) }
  val out = Memory((arr.size * 4).toLong())

  // 6) Run GPU multiplication
  val gpuMs = measureTimeMillis { mb.metalMatMul(inA, inB, t, out) }
  println("GPU took ${gpuMs}ms")

  // 7) [Optional] CPU reference check (very naive O(n^3))
  val cpuOut = IntArray(t * t)
  val cpuMs = measureTimeMillis {
    for (r in 0 until t)
      for (c in 0 until t) {
        var s = 0; for (k in 0 until t) s += arr[r * t + k] * arr[k * t + c]
        cpuOut[r * t + c] = s
      }
  }
  println("CPU took ${cpuMs}ms")

  // 8) Compare a few random spots
  val gpuArr = out.getIntArray(0, arr.size)
  listOf(0, t - 1, t * (t - 1), t * t - 1).forEach {
    if (gpuArr[it] != cpuOut[it]) error("Mismatch at $it: GPU=${gpuArr[it]} CPU=${cpuOut[it]}")
  }
  println("GPU=CPU")
}

// -- JNA and tiny helper funcs below --
interface MetalBridge : Library {
  fun initMetalStuff(p: String)
  fun metalMatMul(a: Pointer, b: Pointer, n: Int, out: Pointer)
}

fun String.execOk() { if (ProcessBuilder(split(" ")).inheritIO().start().waitFor() != 0) error("$this failed") }

fun md5(s: String) = MessageDigest.getInstance("MD5").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }