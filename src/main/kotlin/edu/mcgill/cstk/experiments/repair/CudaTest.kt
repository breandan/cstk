package edu.mcgill.cstk.experiments.repair

import java.nio.ByteBuffer
import java.nio.ByteOrder

fun runCudaKernel(ptxCode: String): Int {
  // Load the CUDA Driver library
  System.loadLibrary("cuda")

  // Initialize the CUDA driver
  val cuInit = Class.forName("com.nvidia.cuda.CudaDriver").getDeclaredMethod("cuInit", Int::class.java)
  cuInit(null, 0)

  // Compile the PTX code into a module
  val modulePointer = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
  val cuModuleLoadData = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod("cuModuleLoadData", ByteBuffer::class.java, ByteArray::class.java)
  cuModuleLoadData(null, modulePointer, ptxCode.toByteArray(Charsets.UTF_8))
  val module = modulePointer.getLong(0)

  // Get the kernel function
  val functionPointer = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
  val cuModuleGetFunction = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod("cuModuleGetFunction", ByteBuffer::class.java, Long::class.java, String::class.java)
  cuModuleGetFunction(null, functionPointer, module, "kernelFunction")
  val function = functionPointer.getLong(0)

  // Allocate GPU memory for the result
  val deviceResult = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
  val cuMemAlloc = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod("cuMemAlloc", ByteBuffer::class.java, Int::class.java)
  cuMemAlloc(null, deviceResult, 4) // 4 bytes for an Int
  val resultPointer = deviceResult.getLong(0)

  // Initialize the memory with zero
  val cuMemsetD32 = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod("cuMemsetD32", Long::class.java, Int::class.java, Int::class.java)
  cuMemsetD32(null, resultPointer, 0, 1)

  // Launch the kernel with minimal configuration
  val cuLaunchKernel = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod(
      "cuLaunchKernel",
      Long::class.java,
      Int::class.java, Int::class.java, Int::class.java,
      Int::class.java, Int::class.java, Int::class.java,
      Int::class.java, Long::class.java,
      ByteBuffer::class.java, ByteBuffer::class.java
    )
  cuLaunchKernel(
    null, function,
    1, 1, 1, // Grid size
    1, 1, 1, // Block size
    0, null, null, null
  )

  // Retrieve the result from the GPU
  val resultBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
  val cuMemcpyDtoH = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod("cuMemcpyDtoH", ByteBuffer::class.java, Long::class.java, Int::class.java)
  cuMemcpyDtoH(null, resultBuffer, resultPointer, 4)

  // Free GPU memory
  val cuMemFree = Class.forName("com.nvidia.cuda.CudaDriver")
    .getDeclaredMethod("cuMemFree", Long::class.java)
  cuMemFree(null, resultPointer)

  return resultBuffer.getInt(0)
}

fun main() {
  val ptxCode = """
        .version 6.0
        .target sm_50
        .address_size 64

        .visible .entry kernelFunction(
            .param .u64 result
        ) {
            .reg .s32 t0;
            ld.param.u64 t0, [result];
            st.global.s32 [t0], 42; // Write 42 to the result
            ret;
        }
    """.trimIndent()

  val startTime = System.currentTimeMillis()
  val result = runCudaKernel(ptxCode)
  println("Kernel Result: $result")
  println("Time: ${System.currentTimeMillis() - startTime}ms")
}