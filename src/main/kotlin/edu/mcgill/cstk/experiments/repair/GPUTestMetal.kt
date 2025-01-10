package edu.mcgill.cstk.experiments.repair

import com.sun.jna.*
import org.intellij.lang.annotations.Language
import java.io.File
import java.security.MessageDigest
import kotlin.random.Random

// 1) JNA Interface to our bridging .dylib
//    Declares a single function: metalDotProduct(int*, int*, int).
interface MetalBridge : Library {
  fun metalDotProduct(arr1: Pointer, arr2: Pointer, count: Int): Int
}

fun main() {
  // Our input data
  val l1 = List(10000) { Random.nextInt() }
  val l2 = List(10000) { Random.nextInt() }
  require(l1.size == l2.size)
  val count = l1.size

  // For timing
  val startTime = System.currentTimeMillis()

  // -------------------------------------------------------------------------
  // 2) Embedded Metal Shader
  //    We'll produce "dot_product.metal" or skip if cached/unchanged.
  // -------------------------------------------------------------------------
  @Language("c++") // close enough
  val metalSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void dot_product(const device int* array1 [[ buffer(0) ]],
                                const device int* array2 [[ buffer(1) ]],
                                device atomic_int* result [[ buffer(2) ]],
                                uint tid [[ thread_position_in_grid ]]) {
            int product = array1[tid] * array2[tid];
            atomic_fetch_add_explicit(result, product, memory_order_relaxed);
        }
    """.trimIndent()

  // Where we'll store the compiled .metallib
  val metallibFile = File("dot_product.metallib")

  // -------------------------------------------------------------------------
  // 3) Embedded Objective-C bridging code
  //    We'll compile to "libMetalBridge.dylib".
  // -------------------------------------------------------------------------

  @Language("c++") // Obj-c, but close enough
  val bridgingSource = """
        #import <Foundation/Foundation.h>
        #import <Metal/Metal.h>

        // We'll keep static references to Metal objects so they're created once
        static id<MTLDevice> sDevice = nil;
        static id<MTLCommandQueue> sCmdQueue = nil;
        static id<MTLComputePipelineState> sPipeline = nil;

        // Constructor runs automatically when .dylib is loaded
        __attribute__((constructor))
        static void initMetalStuff() {
            @autoreleasepool {
                sDevice   = MTLCreateSystemDefaultDevice();
                if(!sDevice) {
                    NSLog(@"[MetalBridge] Failed to create default MTLDevice!");
                    return;
                }

                sCmdQueue = [sDevice newCommandQueue];

                // Load the precompiled .metallib
                NSError* err = nil;
                id<MTLLibrary> lib = [sDevice newLibraryWithFile:@"dot_product.metallib" error:&err];
                if(!lib) {
                    NSLog(@"[MetalBridge] Failed to load dot_product.metallib: %@", err);
                    return;
                }

                id<MTLFunction> func = [lib newFunctionWithName:@"dot_product"];
                if(!func) {
                    NSLog(@"[MetalBridge] Failed to find function 'dot_product'");
                    return;
                }

                sPipeline = [sDevice newComputePipelineStateWithFunction:func error:&err];
                if(!sPipeline) {
                    NSLog(@"[MetalBridge] Failed to create pipeline: %@", err);
                    return;
                }
            }
        }

        // The function we'll call from JNA. Java signature: (I[] I[] I)I
        int metalDotProduct(const int* arr1, const int* arr2, int count) {
            @autoreleasepool {
                if(!sDevice || !sPipeline || !sCmdQueue) {
                    return 0; // If initialization failed
                }

                NSUInteger sizeBytes = sizeof(int) * count;
                id<MTLBuffer> buf1 = [sDevice newBufferWithBytes:arr1 length:sizeBytes options:0];
                id<MTLBuffer> buf2 = [sDevice newBufferWithBytes:arr2 length:sizeBytes options:0];

                int zero = 0;
                id<MTLBuffer> outBuf = [sDevice newBufferWithBytes:&zero length:sizeof(int) options:0];

                id<MTLCommandBuffer> cmd = [sCmdQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:sPipeline];
                [enc setBuffer:buf1 offset:0 atIndex:0];
                [enc setBuffer:buf2 offset:0 atIndex:1];
                [enc setBuffer:outBuf offset:0 atIndex:2];

                MTLSize threads = MTLSizeMake(count, 1, 1);
                MTLSize tgroup = MTLSizeMake(1, 1, 1);
                [enc dispatchThreads:threads threadsPerThreadgroup:tgroup];
                [enc endEncoding];

                [cmd commit];
                [cmd waitUntilCompleted];

                int* ptr = (int*)outBuf.contents;
                return ptr[0];
            }
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
    val metalFile = File("dot_product.metal").apply {
      writeText(metalSource)
    }
    // metal -> AIR
    val airFile = File("dot_product.air")
    val pb1 = ProcessBuilder(
      "xcrun", "metal", "-c", metalFile.absolutePath, "-o", airFile.absolutePath
    ).inheritIO().start()
    if (pb1.waitFor() != 0) error("Failed to compile .metal")

    // AIR -> .metallib
    val pb2 = ProcessBuilder(
      "xcrun", "metallib", airFile.absolutePath, "-o", metallibFile.absolutePath
    ).inheritIO().start()
    if (pb2.waitFor() != 0) error("Failed to link .metallib")

    // Cache the new hash
    metalHashFile.writeText(metalHash)
  }

  // -------------------------------------------------------------------------
  // 6) Compile the bridging .dylib if needed
  // -------------------------------------------------------------------------
  if (needsRebuild(dylibFile, bridgingHashFile, bridgingHash)) {
    println("Compiling bridging .m -> .dylib ...")
    // Write bridging code to bridging.m
    val bridgingM = File("MetalBridge.m").apply {
      writeText(bridgingSource)
    }

    // Compile bridging .m to .o
    val objFile = File("MetalBridge.o")
    val pb1 = ProcessBuilder(
      "clang", "-fobjc-arc",
      "-framework", "Metal",
      "-framework", "Foundation",
      "-c", bridgingM.absolutePath, "-o", objFile.absolutePath
    ).inheritIO().start()
    if (pb1.waitFor() != 0) error("Failed to compile bridging .m")

    // Link to .dylib
    val pb2 = ProcessBuilder(
      "clang", "-shared",
      objFile.absolutePath,
      "-o", dylibFile.absolutePath,
      "-framework", "Metal",
      "-framework", "Foundation"
    ).inheritIO().start()
    if (pb2.waitFor() != 0) error("Failed to link libMetalBridge.dylib")

    bridgingHashFile.writeText(bridgingHash)
  }

  // -------------------------------------------------------------------------
  // 7) Load the .dylib via JNA
  // -------------------------------------------------------------------------
  val metalLib = Native.load("libMetalBridge.dylib", MetalBridge::class.java) as MetalBridge

  // Convert Kotlin Lists to int[]
  val arr1 = l1.toIntArray()
  val arr2 = l2.toIntArray()

  // Allocate native memory using JNA’s Memory class, which is a Pointer
  val sizeBytes = (count * 4).toLong()  // each int is 4 bytes
  val mem1 = Memory(sizeBytes).apply {
    write(0, arr1, 0, arr1.size)
  }
  val mem2 = Memory(sizeBytes).apply {
    write(0, arr2, 0, arr2.size)
  }

  // Now we can pass mem1 and mem2 (both Pointer) to metalDotProduct
  val gpuResult = metalLib.metalDotProduct(mem1, mem2, count)

  val expected = l1.zip(l2).sumOf { (a, b) -> a * b }

  println("GPU Dot Product = $gpuResult (expected $expected)")
  println("Took: ${System.currentTimeMillis() - startTime}ms")
}

// Helper to compute MD5 of a string
private fun md5of(text: String): String {
  val md = MessageDigest.getInstance("MD5")
  val digest = md.digest(text.toByteArray(Charsets.UTF_8))
  return digest.joinToString("") { "%02x".format(it) }
}