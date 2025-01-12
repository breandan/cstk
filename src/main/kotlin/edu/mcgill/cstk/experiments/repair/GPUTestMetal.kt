package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.tensor.DoubleMatrix
import com.sun.jna.*
import edu.mcgill.cstk.utils.execInheritIO
import org.intellij.lang.annotations.Language
import java.io.File
import java.security.MessageDigest
import kotlin.random.Random

// 1) JNA Interface to our bridging .dylib
//    Declares a single function: metalMatMul(int*, int*, int).
interface MetalBridge : Library {
  fun initMetalStuff(metallibPath: String)
  fun metalMatMul(arr1: Pointer, arr2: Pointer, count: Int): Int
}

fun main() {
  val allTime = System.currentTimeMillis()
  // Our input data
  val t = 400
  val arr1 = IntArray(t*t) { Random.nextInt(10) }
  val count = arr1.size

  // -------------------------------------------------------------------------
  // 2) Embedded Metal Shader
  //    We'll produce "mat_mul.metal" or skip if cached/unchanged.
  // -------------------------------------------------------------------------
  @Language("c++") // close enough
  val metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void mat_mul(const device int* array1 [[ buffer(0) ]],
                        const device int* array2 [[ buffer(1) ]],
                        device atomic_int* result [[ buffer(2) ]],
                        uint tid [[ thread_position_in_grid ]]) {
        // Number of total elements in one flattened matrix
        uint n = $t;

        // Derive row and col from tid
        uint row = tid / n;
        uint col = tid % n;

        // Make sure we're in range
        if (row < n && col < n) {
            // Naive matrix multiplication for the single entry [row, col]
            int sum = 0;
            for (uint k = 0; k < n; k++) sum += array1[row * n + k] * array2[k * n + col];

            // We only store the top-right corner (row=0, col=n-1)
            if (row == 0 && col == (n - 1)) atomic_fetch_add_explicit(result, sum, memory_order_relaxed);
        }
    }
  """.trimIndent()

  // Where we'll store the compiled .metallib
  val metallibFile = File("mat_mul.metallib")

  // -------------------------------------------------------------------------
  // 3) Embedded Objective-C bridging code
  //    We'll compile to "libMetalBridge.dylib".
  // -------------------------------------------------------------------------

  @Language("swift")
  val bridgingSource = """
import Foundation
import Metal

// Global references so they're persistent:
private var sDevice: MTLDevice!
private var sCmdQueue: MTLCommandQueue!
private var sPipeline: MTLComputePipelineState!

// Called once to initialize with path to .metallib
@_cdecl("initMetalStuff")
public func initMetalStuff(_ metallibPath: UnsafePointer<CChar>?) {
    guard let cString = metallibPath,
          let path = String(validatingUTF8: cString) else {
        print("[MetalBridge] No valid metallib path.")
        return
    }

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("[MetalBridge] Failed to create default MTLDevice!")
        return
    }
    sDevice = device
    sCmdQueue = device.makeCommandQueue()

    // Load the precompiled .metallib
    do {
        let lib = try device.makeLibrary(filepath: path)
        // Look for function "mat_mul"
        guard let fn = lib.makeFunction(name: "mat_mul") else {
            print("[MetalBridge] Function 'mat_mul' not found in \\(path)")
            return
        }
        sPipeline = try device.makeComputePipelineState(function: fn)
    } catch {
        print("[MetalBridge] Error loading pipeline: \\(error)")
    }
}

// Does a GPU matMul of arr1 x arr2 (both T×T), storing top-right corner
@_cdecl("metalMatMul")
public func metalMatMul(
    _ arr1: UnsafePointer<CInt>?,
    _ arr2: UnsafePointer<CInt>?,
    _ count: CInt
) -> CInt {
    guard let device = sDevice,
          let queue = sCmdQueue,
          let pipeline = sPipeline,
          let in1 = arr1,
          let in2 = arr2 else {
        return 0
    }

    let countSwift = Int(count)
    let sizeBytes = countSwift * MemoryLayout<CInt>.size

    let buf1 = device.makeBuffer(bytes: in1, length: sizeBytes, options: [])!
    let buf2 = device.makeBuffer(bytes: in2, length: sizeBytes, options: [])!

    // We'll store just one integer result (the top-right corner)
    var zero: CInt = 0
    let outBuf = device.makeBuffer(bytes: &zero, length: MemoryLayout<CInt>.size, options: [])!

    guard let cmd = queue.makeCommandBuffer(),
          let enc = cmd.makeComputeCommandEncoder() else {
        return 0
    }

    enc.setComputePipelineState(pipeline)
    enc.setBuffer(buf1, offset: 0, index: 0)
    enc.setBuffer(buf2, offset: 0, index: 1)
    enc.setBuffer(outBuf, offset: 0, index: 2)

    // dispatch 'count' threads (one per element)
    let threads = MTLSizeMake(countSwift, 1, 1)
    let tgroup = MTLSizeMake(1, 1, 1)
    enc.dispatchThreads(threads, threadsPerThreadgroup: tgroup)
    enc.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    let ptr = outBuf.contents().bindMemory(to: CInt.self, capacity: 1)
    return ptr[0]
}
  """.trimIndent()

  val dylibFile = File("libMetalBridge.dylib")

  // -------------------------------------------------------------------------
  // 4) Check if we need to recompile .metallib or .dylib
  //    We'll do a simple content-hash to see if the embedded sources changed.
  // -------------------------------------------------------------------------
  val metalHash = md5of(metalSource)
  val bridgingHash = md5of(bridgingSource)

  // We'll store these in hidden files
  val metalHashFile = File(".metalSource.md5")
  val bridgingHashFile = File(".bridgeSource.md5")

  // Function to see if we need to rebuild
  fun needsRebuild(libFile: File, hashFile: File, newHash: String): Boolean {
    if (!libFile.exists()) return true
    if (!hashFile.exists()) return true
    val oldHash = hashFile.readText().trim()
    return oldHash != newHash
  }

  // -------------------------------------------------------------------------
  // 5) Compile the Metal source if needed
  // -------------------------------------------------------------------------
  if (needsRebuild(metallibFile, metalHashFile, metalHash)) {
    println("Compiling .metal -> .metallib ...")
    // Write a temp .metal file
    val metalFile = File("mat_mul.metal").apply { writeText(metalSource) }
    // metal -> AIR
    val airFile = File("mat_mul.air")
    val pb1 = "xcrun metal -c ${metalFile.absolutePath} -o ${airFile.absolutePath}"
    if (pb1.execInheritIO() != 0) error("Failed to compile .metal")

    // AIR -> .metallib
    val pb2 = "xcrun metallib ${airFile.absolutePath} -o ${metallibFile.absolutePath}"
    if (pb2.execInheritIO() != 0) error("Failed to link .metallib")

    // Cache the new hash
    metalHashFile.writeText(metalHash)
  }

  // -------------------------------------------------------------------------
  // 6) Compile the bridging .dylib if needed
  // -------------------------------------------------------------------------
  // 3b) Build Swift .dylib if needed
  if (needsRebuild(dylibFile, bridgingHashFile, bridgingHash)) {
    println("Rebuilding libMetalBridge.dylib...")

    val swiftFile = File("MetalBridge.swift").apply{ writeText(bridgingSource) }

    val compileCmd = "xcrun swiftc -emit-library ${swiftFile.absolutePath} -o ${dylibFile.absolutePath} -module-name MetalBridge -Xlinker -install_name -Xlinker @rpath/libMetalBridge.dylib"

    if (compileCmd.execInheritIO() != 0) error("Failed to compile Swift bridging code")

    // Save MD5
    bridgingHashFile.writeText(bridgingHash)
  }

  // -------------------------------------------------------------------------
  // 7) Load the .dylib via JNA
  // -------------------------------------------------------------------------
  val metalLib = Native.load("libMetalBridge.dylib", MetalBridge::class.java) as MetalBridge
  metalLib.initMetalStuff(metallibFile.absolutePath)

  // Allocate native memory using JNA’s Memory class, which is a Pointer
  val sizeBytes = (count * 4).toLong()  // each int is 4 bytes
  val mem1 = Memory(sizeBytes).apply { write(0, arr1, 0, arr1.size) }

  println("Warmup took: ${System.currentTimeMillis() - allTime}ms")

  // Metal GEMMs start paying off right quick around t=30 and is about 1000x faster for t=400
  val mtRuntime = System.currentTimeMillis()
  val gpuResult = metalLib.metalMatMul(mem1, mem1, count)
  println("MT result: $gpuResult (${System.currentTimeMillis() - mtRuntime}ms)")

  val mat1 = DoubleMatrix(arr1.map { it.toDouble() })

  val ktRuntime = System.currentTimeMillis()
  val cpuResult = (mat1 * mat1)[0, mat1.numRows - 1].toInt()
//  val cpuResult = cpuMatMulTopRight(arr1, t)
  println("KT result: $cpuResult (${System.currentTimeMillis() - ktRuntime}ms)")

  assert(gpuResult==cpuResult)
}

// Helper to compute MD5 of a string
private fun md5of(text: String): String {
  val md = MessageDigest.getInstance("MD5")
  val digest = md.digest(text.toByteArray(Charsets.UTF_8))
  return digest.joinToString("") { "%02x".format(it) }
}