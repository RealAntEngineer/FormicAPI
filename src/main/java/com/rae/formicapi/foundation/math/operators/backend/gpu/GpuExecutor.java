package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.jetbrains.annotations.Nullable;
import org.jocl.*;

import static org.jocl.CL.*;

/**
 * Owns the OpenCL platform/device/context/queue and the single compiled
 * kernel program used by every {@code Gpu*Vector}. This is the GPU
 * counterpart of {@link com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutor}:
 * where {@code CpuExecutor} owns a thread pool and dispatches ranges of work
 * across it, {@code GpuExecutor} owns a device/queue and dispatches kernels
 * onto it.
 *
 * <p>One {@code GpuExecutor} should be created per device and shared across
 * every {@code Gpu*Vector} that needs to interoperate (buffers from one
 * context cannot be used in kernels launched through another). Vectors
 * receive it via {@link GpuExecutable#setExecutor(GpuExecutor)}.
 *
 * <p><b>Backend abstraction:</b> this class is deliberately the <i>only</i>
 * place that touches the OpenCL API directly. {@code Gpu*Vector} classes
 * only ever call methods on {@code GpuExecutor} (allocate/upload/download/
 * copy/fill/launch*), never JOCL types themselves. The intent is that a
 * future move to Vulkan compute only requires rewriting this class (and
 * swapping the kernel source language from OpenCL C to a SPIR-V/GLSL
 * compute shader), not touching the vector classes' logic.
 *
 * <p><b>Thread-safety:</b> a single {@code cl_command_queue} and the kernel
 * objects built from it hold mutable state (arguments are set on the kernel
 * object before each enqueue), so all kernel-launching methods here are
 * {@code synchronized}. This context is not designed for concurrent
 * multithreaded dispatch from several Java threads at once - it processes
 * one operation at a time, same as a single OpenCL in-order queue would
 * anyway.
 *
 * <p><b>Precision:</b> requires a device that supports the
 * {@code cl_khr_fp64} extension (double precision). Selection fails fast
 * with a clear message if none is found, rather than silently truncating to
 * float.
 */
public final class GpuExecutor implements AutoCloseable {
//TODO generate with claude, treat with caution
    /**
     * Preferred local work-group size for elementwise/reduction kernels; capped to the device's actual max.
     */
    private static final int PREFERRED_LOCAL_SIZE = 256;

    private static final String KERNEL_SOURCE = """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            
            __kernel void axpy(__global double* d, const double a, __global const double* x) {
                int i = get_global_id(0);
                d[i] += a * x[i];
            }
            
            __kernel void add_scalar(__global double* d, const double a) {
                int i = get_global_id(0);
                d[i] += a;
            }
            
            __kernel void add_vector(__global double* d, __global const double* x) {
                int i = get_global_id(0);
                d[i] += x[i];
            }
            
            __kernel void scale(__global double* d, const double a) {
                int i = get_global_id(0);
                d[i] *= a;
            }
            
            // NOTE: no atomics here - this mirrors CpuDoubleVector.scatterAxpy's
            // parallel path, which also does a plain `d[idx.get(i)] += ...` across
            // worker threads with no synchronization. Same caller contract applies:
            // idx must not contain duplicate targets across work-items that can run
            // concurrently, or the add is a race (last-writer-wins on that lane's
            // read-modify-write), same as on the CPU executor.
            __kernel void scatter_axpy(__global double* d, const double alpha, __global const double* s, __global const int* idx) {
                int i = get_global_id(0);
                int target = idx[i];
                d[target] += alpha * s[i];
            }
            
            // Tree reduction within each work-group; writes one partial sum per
            // group into `partials`. Host sums the (small) partials array, same
            // shape as CpuExecutor.parallelReduceDouble summing one double per
            // worker thread.
            __kernel void dot_partial(__global const double* a, __global const double* b,
                                       __local double* scratch, __global double* partials, const int n) {
                int gid = get_global_id(0);
                int lid = get_local_id(0);
                int lsize = get_local_size(0);
            
                scratch[lid] = (gid < n) ? a[gid] * b[gid] : 0.0;
                barrier(CLK_LOCAL_MEM_FENCE);
            
                for (int offset = lsize / 2; offset > 0; offset >>= 1) {
                    if (lid < offset)
                        scratch[lid] += scratch[lid + offset];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            
                if (lid == 0)
                    partials[get_group_id(0)] = scratch[0];
            }
            
            __kernel void skipped_dot_partial(__global const double* a, __global const double* b,
                                               __global const int* unknownIdx,
                                               __local double* scratch, __global double* partials, const int n) {
                int gid = get_global_id(0);
                int lid = get_local_id(0);
                int lsize = get_local_size(0);
            
                scratch[lid] = (gid < n) ? a[unknownIdx[gid]] * b[gid] : 0.0;
                barrier(CLK_LOCAL_MEM_FENCE);
            
                for (int offset = lsize / 2; offset > 0; offset >>= 1) {
                    if (lid < offset)
                        scratch[lid] += scratch[lid + offset];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            
                if (lid == 0)
                    partials[get_group_id(0)] = scratch[0];
            }
            
            // One work-item per row - mirrors CpuPaddedCSRMatrix.applyRange exactly.
            __kernel void csr_matvec(__global const double* values, __global const int* colIndex,
                                      const int nnzPerRow, __global const double* x, __global double* result) {
                int row = get_global_id(0);
                int base = row * nnzPerRow;
                double sum = 0.0;
                for (int i = 0; i < nnzPerRow; i++)
                    sum += values[base + i] * x[colIndex[base + i]];
                result[row] = sum;
            }
            
            // Mirrors CpuPaddedCSR2Tensor.applyRange: F_i(x) = sum(c * x[j] * x[k]).
            __kernel void csr2_apply(__global const double* values, __global const int* var1Index, __global const int* var2Index,
                                      const int termsPerEquation, __global const double* x, __global double* result) {
                int row = get_global_id(0);
                int base = row * termsPerEquation;
                double sum = 0.0;
                for (int i = 0; i < termsPerEquation; i++)
                    sum += values[base + i] * x[var1Index[base + i]] * x[var2Index[base + i]];
                result[row] = sum;
            }
            
            // Mirrors CpuPaddedCSR2Tensor.applyJacobianRange: contribution of one term is
            // c * (direction_j * x_k + x_j * direction_k), which also handles j == k correctly.
            __kernel void csr2_apply_jacobian(__global const double* values, __global const int* var1Index, __global const int* var2Index,
                                               const int termsPerEquation, __global const double* x, __global const double* direction,
                                               __global double* result) {
                int row = get_global_id(0);
                int base = row * termsPerEquation;
                double sum = 0.0;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    double c = values[idx];
                    int j = var1Index[idx];
                    int k = var2Index[idx];
                    sum += c * (direction[j] * x[k] + x[j] * direction[k]);
                }
                result[row] = sum;
            }
            
            // Mirrors CpuPaddedCSRTensor.applyRange for general order N. varIndices is the
            // flattened [equations * termsPerEquation * orderMinusOne] form of the Cpu
            // version's int[term][dimension] jagged array - term `idx`'s indices live at
            // varIndices[idx * orderMinusOne .. idx * orderMinusOne + orderMinusOne).
            __kernel void csrn_apply(__global const double* values, __global const int* varIndices,
                                      const int termsPerEquation, const int orderMinusOne,
                                      __global const double* x, __global double* result) {
                int row = get_global_id(0);
                int base = row * termsPerEquation;
                double sum = 0.0;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    double p = values[idx];
                    int vbase = idx * orderMinusOne;
                    for (int d = 0; d < orderMinusOne; d++)
                        p *= x[varIndices[vbase + d]];
                    sum += p;
                }
                result[row] = sum;
            }
            
            // Mirrors CpuPaddedCSRTensor.applyJacobianRange.
            __kernel void csrn_apply_jacobian(__global const double* values, __global const int* varIndices,
                                               const int termsPerEquation, const int orderMinusOne,
                                               __global const double* x, __global const double* direction,
                                               __global double* result) {
                int row = get_global_id(0);
                int base = row * termsPerEquation;
                double sum = 0.0;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    int vbase = idx * orderMinusOne;
                    for (int m = 0; m < orderMinusOne; m++) {
                        double term = values[idx];
                        for (int r = 0; r < orderMinusOne; r++) {
                            int v = varIndices[vbase + r];
                            term *= (r == m) ? direction[v] : x[v];
                        }
                        sum += term;
                    }
                }
                result[row] = sum;
            }
            """;

    /**
     * Scatter-add kernel for {@code GpuPaddedCSRMatrix.transposeApply}. Kept
     * as a separate source blob from {@link #KERNEL_SOURCE}, only appended
     * and compiled when {@link #supportsScatterAtomics} - unlike
     * {@code scatter_axpy} (which gets away with a plain {@code +=} because
     * its caller contract requires distinct targets), a CSR transpose has no
     * such guarantee: arbitrary rows can and typically do share columns, so
     * this needs a real atomic add. OpenCL has no native atomic double add,
     * so this does the standard compare-and-swap-on-the-bit-pattern trick,
     * which needs {@code cl_khr_int64_base_atomics} for 64-bit
     * {@code atom_cmpxchg}. Devices without that extension would fail to
     * even compile this source, so it's only concatenated onto the program
     * when the device actually supports it - see the constructor.
     */
    private static final String ATOMIC_KERNEL_SOURCE = """
            #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
            
            inline void atomic_add_double(volatile __global double* addr, double val) {
                union { ulong u; double f; } next, expected, current;
                current.f = *addr;
                do {
                    expected.f = current.f;
                    next.f = expected.f + val;
                    current.u = atom_cmpxchg((volatile __global ulong*) addr, expected.u, next.u);
                } while (current.u != expected.u);
            }
            
            // One work-item per row, scatter-adding into result[colIndex[...]] -
            // mirrors CpuPaddedCSRMatrix.transposeApplyRange, but that version's
            // per-thread-private-buffer trick (cheap for ~8 CPU threads) doesn't
            // scale to GPU work-item counts (thousands of output-sized buffers),
            // hence the atomic add instead. Caller must zero `result` first, same
            // as the Cpu version's Arrays.fill(resultArr, 0.0).
            __kernel void csr_matvec_transpose(__global const double* values, __global const int* colIndex,
                                                const int nnzPerRow, __global const double* x, __global double* result) {
                int row = get_global_id(0);
                int base = row * nnzPerRow;
                double xr = x[row];
                for (int i = 0; i < nnzPerRow; i++)
                    atomic_add_double(&result[colIndex[base + i]], values[base + i] * xr);
            }
            """;

    private final cl_platform_id   platform;
    private final cl_device_id     device;
    private final cl_context       context;
    private final cl_command_queue queue;
    private final cl_program       program;

    private final cl_kernel kAxpy;
    private final cl_kernel kAddScalar;
    private final cl_kernel kAddVector;
    private final cl_kernel kScale;
    private final cl_kernel kScatterAxpy;
    private final cl_kernel kDotPartial;
    private final cl_kernel kSkippedDotPartial;
    private final cl_kernel kCsrMatvec;
    private final cl_kernel kCsr2Apply;
    private final cl_kernel kCsr2ApplyJacobian;
    private final cl_kernel kCsrNApply;
    private final cl_kernel kCsrNApplyJacobian;

    /**
     * Null when the device lacks {@code cl_khr_int64_base_atomics} - see {@link #ATOMIC_KERNEL_SOURCE}.
     */
    private final @Nullable cl_kernel kCsrMatvecTranspose;
    private final           boolean   supportsScatterAtomics;

    private final    int     localSize;
    private volatile boolean closed = false;

    /**
     * Opens a context on the first fp64-capable GPU device found; falls back to any fp64-capable device.
     */
    public GpuExecutor() {
        this(selectDefaultDevice());
    }

    public GpuExecutor(cl_device_id device) {
        CL.setExceptionsEnabled(true);

        this.device = device;
        this.platform = queryPlatform(device);
        this.supportsScatterAtomics = deviceExtensions(device).contains("cl_khr_int64_base_atomics");

        cl_context_properties props = new cl_context_properties();
        props.addProperty(CL_CONTEXT_PLATFORM, platform);

        this.context = clCreateContext(props, 1, new cl_device_id[]{device}, null, null, null);
        this.queue = clCreateCommandQueue(context, device, 0, null);

        String source = supportsScatterAtomics ? KERNEL_SOURCE + ATOMIC_KERNEL_SOURCE : KERNEL_SOURCE;
        this.program = clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        try {
            clBuildProgram(program, 1, new cl_device_id[]{device}, "", null, null);
        } catch (CLException e) {
            throw new RuntimeException("OpenCL kernel build failed: " + buildLog(), e);
        }

        this.kAxpy = clCreateKernel(program, "axpy", null);
        this.kAddScalar = clCreateKernel(program, "add_scalar", null);
        this.kAddVector = clCreateKernel(program, "add_vector", null);
        this.kScale = clCreateKernel(program, "scale", null);
        this.kScatterAxpy = clCreateKernel(program, "scatter_axpy", null);
        this.kDotPartial = clCreateKernel(program, "dot_partial", null);
        this.kSkippedDotPartial = clCreateKernel(program, "skipped_dot_partial", null);
        this.kCsrMatvec = clCreateKernel(program, "csr_matvec", null);
        this.kCsr2Apply = clCreateKernel(program, "csr2_apply", null);
        this.kCsr2ApplyJacobian = clCreateKernel(program, "csr2_apply_jacobian", null);
        this.kCsrNApply = clCreateKernel(program, "csrn_apply", null);
        this.kCsrNApplyJacobian = clCreateKernel(program, "csrn_apply_jacobian", null);
        this.kCsrMatvecTranspose = supportsScatterAtomics ? clCreateKernel(program, "csr_matvec_transpose", null) : null;

        this.localSize = largestPowerOfTwoLEQ(Math.min(PREFERRED_LOCAL_SIZE, queryMaxWorkGroupSize(device)));
    }


    // ---------------------------------------------------------------- device selection

    private static cl_device_id selectDefaultDevice() {
        CL.setExceptionsEnabled(true);

        int[] numPlatforms = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);
        if (numPlatforms[0] == 0)
            throw new RuntimeException("No OpenCL platforms found");

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        cl_device_id fallback = null;

        for (cl_platform_id platform : platforms) {
            for (cl_device_id candidate : devicesOf(platform, CL_DEVICE_TYPE_GPU)) {
                if (supportsFp64(candidate))
                    return candidate;
            }
            for (cl_device_id candidate : devicesOf(platform, CL_DEVICE_TYPE_ALL)) {
                if (fallback == null && supportsFp64(candidate))
                    fallback = candidate;
            }
        }

        if (fallback == null)
            throw new RuntimeException("No OpenCL device with cl_khr_fp64 (double precision) support found");

        return fallback;
    }

    private static cl_platform_id queryPlatform(cl_device_id device) {
        cl_platform_id[] out = new cl_platform_id[1];
        clGetDeviceInfo(device, CL_DEVICE_PLATFORM, Sizeof.cl_platform_id, Pointer.to(out), null);
        return out[0];
    }

    private static String deviceExtensions(cl_device_id device) {
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, buffer.length, Pointer.to(buffer), null);
        return new String(buffer).trim();
    }

    private String buildLog() {
        long[] logSize = new long[1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, logSize);
        byte[] log = new byte[(int) logSize[0]];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
        return new String(log);
    }

    private static int largestPowerOfTwoLEQ(int n) {
        int p = 1;
        while (p * 2 <= n)
            p *= 2;
        return Math.max(p, 1);
    }

    private static int queryMaxWorkGroupSize(cl_device_id device) {
        long[] out = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(out), null);
        return (int) out[0];
    }

    private static cl_device_id[] devicesOf(cl_platform_id platform, long deviceType) {
        int[] numDevices = new int[1];
        try {
            clGetDeviceIDs(platform, deviceType, 0, null, numDevices);
        } catch (CLException e) {
            return new cl_device_id[0]; // e.g. CL_DEVICE_NOT_FOUND when a platform has no GPU
        }
        if (numDevices[0] == 0)
            return new cl_device_id[0];

        cl_device_id[] devices = new cl_device_id[numDevices[0]];
        clGetDeviceIDs(platform, deviceType, devices.length, devices, null);
        return devices;
    }

    private static boolean supportsFp64(cl_device_id device) {
        return deviceExtensions(device).contains("cl_khr_fp64");
    }

    // ---------------------------------------------------------------- buffer lifecycle

    public cl_mem allocateIntBuffer(int length) {
        return clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_int, null, null);
    }

    /**
     * Boolean vectors are stored as one {@code cl_char} per element - OpenCL has no packed device-side bool type.
     */
    public cl_mem allocateByteBuffer(int length) {
        return clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_char, null, null);
    }

    public synchronized void uploadDoubles(cl_mem buffer, double[] host, int length) {
        uploadDoubles(buffer, 0, host, length);
    }

    public synchronized void uploadDoubles(cl_mem buffer, int elementOffset, double[] host, int length) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                (long) length * Sizeof.cl_double, Pointer.to(host), 0, null, null);
    }

    public synchronized void uploadInts(cl_mem buffer, int[] host, int length) {
        uploadInts(buffer, 0, host, length);
    }

    public synchronized void uploadInts(cl_mem buffer, int elementOffset, int[] host, int length) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                (long) length * Sizeof.cl_int, Pointer.to(host), 0, null, null);
    }

    public synchronized void downloadInts(cl_mem buffer, int[] host, int length) {
        downloadInts(buffer, 0, host, length);
    }

    public synchronized void downloadInts(cl_mem buffer, int elementOffset, int[] host, int length) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                (long) length * Sizeof.cl_int, Pointer.to(host), 0, null, null);
    }

    public synchronized void uploadBytes(cl_mem buffer, byte[] host, int length) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_char, Pointer.to(host), 0, null, null);
    }

    public synchronized void downloadBytes(cl_mem buffer, byte[] host, int length) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_char, Pointer.to(host), 0, null, null);
    }

    public synchronized void uploadDoubleAt(cl_mem buffer, int elementOffset, double value) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                Sizeof.cl_double, Pointer.to(new double[]{value}), 0, null, null);
    }

    public synchronized double downloadDoubleAt(cl_mem buffer, int elementOffset) {
        double[] out = new double[1];
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                Sizeof.cl_double, Pointer.to(out), 0, null, null);
        return out[0];
    }

    public synchronized void uploadIntAt(cl_mem buffer, int elementOffset, int value) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                Sizeof.cl_int, Pointer.to(new int[]{value}), 0, null, null);
    }

    // Single-element offset transfers, backing Gpu*Vector.get/set. A device
    // round-trip per call - fine for setup/debugging, not for hot loops
    // (bulk upload/download above is the fast path for that).

    public synchronized int downloadIntAt(cl_mem buffer, int elementOffset) {
        int[] out = new int[1];
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                Sizeof.cl_int, Pointer.to(out), 0, null, null);
        return out[0];
    }

    public synchronized void uploadByteAt(cl_mem buffer, int elementOffset, byte value) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_char,
                Sizeof.cl_char, Pointer.to(new byte[]{value}), 0, null, null);
    }

    public synchronized byte downloadByteAt(cl_mem buffer, int elementOffset) {
        byte[] out = new byte[1];
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_char,
                Sizeof.cl_char, Pointer.to(out), 0, null, null);
        return out[0];
    }

    public synchronized void copyDoubleBuffer(cl_mem src, cl_mem dst, int length) {
        if (length <= 0) return;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, (long) length * Sizeof.cl_double, 0, null, null);
        clFinish(queue);
    }

    public synchronized void copyIntBuffer(cl_mem src, cl_mem dst, int length) {
        if (length <= 0) return;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, (long) length * Sizeof.cl_int, 0, null, null);
        clFinish(queue);
    }

    public synchronized void copyByteBuffer(cl_mem src, cl_mem dst, int length) {
        if (length <= 0) return;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, (long) length * Sizeof.cl_char, 0, null, null);
        clFinish(queue);
    }

    public synchronized void fillDoubleBuffer(cl_mem buffer, double value, int length) {
        if (length <= 0) return;
        clEnqueueFillBuffer(queue, buffer, Pointer.to(new double[]{value}), Sizeof.cl_double, 0, (long) length * Sizeof.cl_double, 0, null, null);
    }

    /**
     * Fresh {@code cl_mem} buffers hold undefined content until written -
     * unlike a Java {@code new int[]}/{@code new double[]}, OpenCL gives no
     * zero-initialization guarantee. {@code GpuPaddedCSRMatrix} and the two
     * tensor classes rely on this to zero-init their column/variable-index
     * buffers at construction and on grow-resize, matching what a plain
     * {@code new int[]} (all zeros, i.e. every unset slot safely points at
     * index 0) already gives the Cpu* versions for free.
     */
    public synchronized void fillIntBuffer(cl_mem buffer, int value, int length) {
        if (length <= 0) return;
        clEnqueueFillBuffer(queue, buffer, Pointer.to(new int[]{value}), Sizeof.cl_int, 0, (long) length * Sizeof.cl_int, 0, null, null);
    }

    public synchronized void fillByteBuffer(cl_mem buffer, byte value, int length) {
        if (length <= 0) return;
        clEnqueueFillBuffer(queue, buffer, Pointer.to(new byte[]{value}), Sizeof.cl_char, 0, (long) length * Sizeof.cl_char, 0, null, null);
    }

    public synchronized void launchAxpy(cl_mem d, double a, cl_mem x, int size) {
        if (size <= 0) return;
        clSetKernelArg(kAxpy, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kAxpy, 1, Sizeof.cl_double, Pointer.to(new double[]{a}));
        clSetKernelArg(kAxpy, 2, Sizeof.cl_mem, Pointer.to(x));
        enqueueRange(kAxpy, size);
    }

    private void enqueueRange(cl_kernel kernel, int size) {
        clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[]{size}, null, 0, null, null);
    }

    public synchronized void launchAddScalar(cl_mem d, double a, int size) {
        if (size <= 0) return;
        clSetKernelArg(kAddScalar, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kAddScalar, 1, Sizeof.cl_double, Pointer.to(new double[]{a}));
        enqueueRange(kAddScalar, size);
    }

    // ---------------------------------------------------------------- kernel launches

    public synchronized void launchAddVector(cl_mem d, cl_mem x, int size) {
        if (size <= 0) return;
        clSetKernelArg(kAddVector, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kAddVector, 1, Sizeof.cl_mem, Pointer.to(x));
        enqueueRange(kAddVector, size);
    }

    public synchronized void launchScale(cl_mem d, double a, int size) {
        if (size <= 0) return;
        clSetKernelArg(kScale, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kScale, 1, Sizeof.cl_double, Pointer.to(new double[]{a}));
        enqueueRange(kScale, size);
    }

    /**
     * See the {@code scatter_axpy} kernel doc: caller must not pass duplicate targets in {@code idx}.
     */
    public synchronized void launchScatterAxpy(cl_mem d, double alpha, cl_mem s, cl_mem idx, int size) {
        if (size <= 0) return;
        clSetKernelArg(kScatterAxpy, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kScatterAxpy, 1, Sizeof.cl_double, Pointer.to(new double[]{alpha}));
        clSetKernelArg(kScatterAxpy, 2, Sizeof.cl_mem, Pointer.to(s));
        clSetKernelArg(kScatterAxpy, 3, Sizeof.cl_mem, Pointer.to(idx));
        enqueueRange(kScatterAxpy, size);
    }

    // ---------------------------------------------------------------- CSR matrix / tensor kernels

    public synchronized void launchCsrMatvec(cl_mem values, cl_mem colIndex, int nnzPerRow, cl_mem x, cl_mem result, int rows) {
        if (rows <= 0) return;
        clSetKernelArg(kCsrMatvec, 0, Sizeof.cl_mem, Pointer.to(values));
        clSetKernelArg(kCsrMatvec, 1, Sizeof.cl_mem, Pointer.to(colIndex));
        clSetKernelArg(kCsrMatvec, 2, Sizeof.cl_int, Pointer.to(new int[]{nnzPerRow}));
        clSetKernelArg(kCsrMatvec, 3, Sizeof.cl_mem, Pointer.to(x));
        clSetKernelArg(kCsrMatvec, 4, Sizeof.cl_mem, Pointer.to(result));
        enqueueRange(kCsrMatvec, rows);
    }

    /**
     * Scatter-adds into {@code result} - caller must zero it first (matching
     * {@code CpuPaddedCSRMatrix.transposeApply}'s {@code Arrays.fill(resultArr, 0.0)}).
     *
     * @throws UnsupportedOperationException if the device lacks {@code cl_khr_int64_base_atomics} - see {@link #ATOMIC_KERNEL_SOURCE}
     */
    public synchronized void launchCsrMatvecTranspose(cl_mem values, cl_mem colIndex, int nnzPerRow, cl_mem x, cl_mem result, int rows) {
        if (rows <= 0) return;
        if (!supportsScatterAtomics)
            throw new UnsupportedOperationException(
                    "GpuPaddedCSRMatrix.transposeApply(...) requires cl_khr_int64_base_atomics, which this device doesn't report support for");

        clSetKernelArg(kCsrMatvecTranspose, 0, Sizeof.cl_mem, Pointer.to(values));
        clSetKernelArg(kCsrMatvecTranspose, 1, Sizeof.cl_mem, Pointer.to(colIndex));
        clSetKernelArg(kCsrMatvecTranspose, 2, Sizeof.cl_int, Pointer.to(new int[]{nnzPerRow}));
        clSetKernelArg(kCsrMatvecTranspose, 3, Sizeof.cl_mem, Pointer.to(x));
        clSetKernelArg(kCsrMatvecTranspose, 4, Sizeof.cl_mem, Pointer.to(result));
        enqueueRange(kCsrMatvecTranspose, rows);
    }

    public boolean supportsScatterAtomics() {
        return supportsScatterAtomics;
    }

    public synchronized void launchCsr2Apply(cl_mem values, cl_mem var1Index, cl_mem var2Index, int termsPerEquation,
                                             cl_mem x, cl_mem result, int equations) {
        if (equations <= 0) return;
        clSetKernelArg(kCsr2Apply, 0, Sizeof.cl_mem, Pointer.to(values));
        clSetKernelArg(kCsr2Apply, 1, Sizeof.cl_mem, Pointer.to(var1Index));
        clSetKernelArg(kCsr2Apply, 2, Sizeof.cl_mem, Pointer.to(var2Index));
        clSetKernelArg(kCsr2Apply, 3, Sizeof.cl_int, Pointer.to(new int[]{termsPerEquation}));
        clSetKernelArg(kCsr2Apply, 4, Sizeof.cl_mem, Pointer.to(x));
        clSetKernelArg(kCsr2Apply, 5, Sizeof.cl_mem, Pointer.to(result));
        enqueueRange(kCsr2Apply, equations);
    }

    public synchronized void launchCsr2ApplyJacobian(cl_mem values, cl_mem var1Index, cl_mem var2Index, int termsPerEquation,
                                                     cl_mem x, cl_mem direction, cl_mem result, int equations) {
        if (equations <= 0) return;
        clSetKernelArg(kCsr2ApplyJacobian, 0, Sizeof.cl_mem, Pointer.to(values));
        clSetKernelArg(kCsr2ApplyJacobian, 1, Sizeof.cl_mem, Pointer.to(var1Index));
        clSetKernelArg(kCsr2ApplyJacobian, 2, Sizeof.cl_mem, Pointer.to(var2Index));
        clSetKernelArg(kCsr2ApplyJacobian, 3, Sizeof.cl_int, Pointer.to(new int[]{termsPerEquation}));
        clSetKernelArg(kCsr2ApplyJacobian, 4, Sizeof.cl_mem, Pointer.to(x));
        clSetKernelArg(kCsr2ApplyJacobian, 5, Sizeof.cl_mem, Pointer.to(direction));
        clSetKernelArg(kCsr2ApplyJacobian, 6, Sizeof.cl_mem, Pointer.to(result));
        enqueueRange(kCsr2ApplyJacobian, equations);
    }

    public synchronized void launchCsrNApply(cl_mem values, cl_mem varIndices, int termsPerEquation, int orderMinusOne,
                                             cl_mem x, cl_mem result, int equations) {
        if (equations <= 0) return;
        clSetKernelArg(kCsrNApply, 0, Sizeof.cl_mem, Pointer.to(values));
        clSetKernelArg(kCsrNApply, 1, Sizeof.cl_mem, Pointer.to(varIndices));
        clSetKernelArg(kCsrNApply, 2, Sizeof.cl_int, Pointer.to(new int[]{termsPerEquation}));
        clSetKernelArg(kCsrNApply, 3, Sizeof.cl_int, Pointer.to(new int[]{orderMinusOne}));
        clSetKernelArg(kCsrNApply, 4, Sizeof.cl_mem, Pointer.to(x));
        clSetKernelArg(kCsrNApply, 5, Sizeof.cl_mem, Pointer.to(result));
        enqueueRange(kCsrNApply, equations);
    }

    public synchronized void launchCsrNApplyJacobian(cl_mem values, cl_mem varIndices, int termsPerEquation, int orderMinusOne,
                                                     cl_mem x, cl_mem direction, cl_mem result, int equations) {
        if (equations <= 0) return;
        clSetKernelArg(kCsrNApplyJacobian, 0, Sizeof.cl_mem, Pointer.to(values));
        clSetKernelArg(kCsrNApplyJacobian, 1, Sizeof.cl_mem, Pointer.to(varIndices));
        clSetKernelArg(kCsrNApplyJacobian, 2, Sizeof.cl_int, Pointer.to(new int[]{termsPerEquation}));
        clSetKernelArg(kCsrNApplyJacobian, 3, Sizeof.cl_int, Pointer.to(new int[]{orderMinusOne}));
        clSetKernelArg(kCsrNApplyJacobian, 4, Sizeof.cl_mem, Pointer.to(x));
        clSetKernelArg(kCsrNApplyJacobian, 5, Sizeof.cl_mem, Pointer.to(direction));
        clSetKernelArg(kCsrNApplyJacobian, 6, Sizeof.cl_mem, Pointer.to(result));
        enqueueRange(kCsrNApplyJacobian, equations);
    }

    public synchronized double launchDot(cl_mem a, cl_mem b, int size) {
        if (size <= 0) return 0.0;
        return reduceDouble(kDotPartial, size, (kernel, argBase) -> {
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        });
    }

    /**
     * Shared plumbing for the two reduction kernels: both take a fixed set of
     * data args at the front (bound by {@code binder}), then
     * {@code (__local scratch, __global partials, const int n)} - this method
     * appends those three and does the group-count sizing / partial-sum work
     * common to both.
     */
    private double reduceDouble(cl_kernel kernel, int size, DataArgBinder binder) {
        int numGroups        = (size + localSize - 1) / localSize;
        int paddedGlobalSize = numGroups * localSize;

        binder.bind(kernel, 0);

        int    nextArg  = countDataArgs(kernel);
        cl_mem partials = allocateDoubleBuffer(numGroups);
        try {
            clSetKernelArg(kernel, nextArg, (long) localSize * Sizeof.cl_double, null);
            clSetKernelArg(kernel, nextArg + 1, Sizeof.cl_mem, Pointer.to(partials));
            clSetKernelArg(kernel, nextArg + 2, Sizeof.cl_int, Pointer.to(new int[]{size}));

            clEnqueueNDRangeKernel(queue, kernel, 1, null,
                    new long[]{paddedGlobalSize}, new long[]{localSize}, 0, null, null);

            double[] partialsHost = new double[numGroups];
            downloadDoubles(partials, partialsHost, numGroups);

            double sum = 0.0;
            for (double v : partialsHost)
                sum += v;
            return sum;
        } finally {
            release(partials);
        }
    }

    /**
     * {@code dot_partial} has 2 data args (a, b); {@code skipped_dot_partial} has 3 (a, b, unknownIdx).
     */
    private int countDataArgs(cl_kernel kernel) {
        return kernel == kSkippedDotPartial ? 3 : 2;
    }

    public cl_mem allocateDoubleBuffer(int length) {
        return clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_double, null, null);
    }

    public synchronized void downloadDoubles(cl_mem buffer, double[] host, int length) {
        downloadDoubles(buffer, 0, host, length);
    }

    public void release(cl_mem buffer) {
        if (buffer != null)
            clReleaseMemObject(buffer);
    }

    public synchronized void downloadDoubles(cl_mem buffer, int elementOffset, double[] host, int length) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                (long) length * Sizeof.cl_double, Pointer.to(host), 0, null, null);
    }

    public synchronized double launchSkippedDot(cl_mem a, cl_mem b, cl_mem unknownIdx, int size) {
        if (size <= 0) return 0.0;
        return reduceDouble(kSkippedDotPartial, size, (kernel, argBase) -> {
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(unknownIdx));
        });
    }

    /**
     * Blocks until every previously enqueued operation on this context's queue has completed.
     */
    public void finish() {
        clFinish(queue);
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;

        clReleaseKernel(kAxpy);
        clReleaseKernel(kAddScalar);
        clReleaseKernel(kAddVector);
        clReleaseKernel(kScale);
        clReleaseKernel(kScatterAxpy);
        clReleaseKernel(kDotPartial);
        clReleaseKernel(kSkippedDotPartial);
        clReleaseKernel(kCsrMatvec);
        clReleaseKernel(kCsr2Apply);
        clReleaseKernel(kCsr2ApplyJacobian);
        clReleaseKernel(kCsrNApply);
        clReleaseKernel(kCsrNApplyJacobian);
        if (kCsrMatvecTranspose != null)
            clReleaseKernel(kCsrMatvecTranspose);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    @FunctionalInterface
    private interface DataArgBinder {
        void bind(cl_kernel kernel, int argBase);
    }
}