package com.rae.formicapi.foundation.math.operators.backend.gpu;

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
 * multi-threaded dispatch from several Java threads at once - it processes
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
    /** Preferred local work-group size for elementwise/reduction kernels; capped to the device's actual max. */
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
            """;

    private final cl_platform_id platform;
    private final cl_device_id   device;
    private final cl_context     context;
    private final cl_command_queue queue;
    private final cl_program     program;

    private final cl_kernel kAxpy;
    private final cl_kernel kAddScalar;
    private final cl_kernel kAddVector;
    private final cl_kernel kScale;
    private final cl_kernel kScatterAxpy;
    private final cl_kernel kDotPartial;
    private final cl_kernel kSkippedDotPartial;

    private final int localSize;
    private volatile boolean closed = false;

    /** Opens a context on the first fp64-capable GPU device found; falls back to any fp64-capable device. */
    public GpuExecutor() {
        this(selectDefaultDevice());
    }

    public GpuExecutor(cl_device_id device) {
        CL.setExceptionsEnabled(true);

        this.device = device;
        this.platform = queryPlatform(device);

        cl_context_properties props = new cl_context_properties();
        props.addProperty(CL_CONTEXT_PLATFORM, platform);

        this.context = clCreateContext(props, 1, new cl_device_id[]{device}, null, null, null);
        this.queue = clCreateCommandQueue(context, device, 0, null);

        this.program = clCreateProgramWithSource(context, 1, new String[]{KERNEL_SOURCE}, null, null);
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

    private static String deviceExtensions(cl_device_id device) {
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, buffer.length, Pointer.to(buffer), null);
        return new String(buffer).trim();
    }

    private static cl_platform_id queryPlatform(cl_device_id device) {
        cl_platform_id[] out = new cl_platform_id[1];
        clGetDeviceInfo(device, CL_DEVICE_PLATFORM, Sizeof.cl_platform_id, Pointer.to(out), null);
        return out[0];
    }

    private static int queryMaxWorkGroupSize(cl_device_id device) {
        long[] out = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(out), null);
        return (int) out[0];
    }

    private static int largestPowerOfTwoLEQ(int n) {
        int p = 1;
        while (p * 2 <= n)
            p *= 2;
        return Math.max(p, 1);
    }

    private String buildLog() {
        long[] logSize = new long[1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, logSize);
        byte[] log = new byte[(int) logSize[0]];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
        return new String(log);
    }

    // ---------------------------------------------------------------- buffer lifecycle

    public cl_mem allocateDoubleBuffer(int length) {
        return clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_double, null, null);
    }

    public cl_mem allocateIntBuffer(int length) {
        return clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_int, null, null);
    }

    /** Boolean vectors are stored as one {@code cl_char} per element - OpenCL has no packed device-side bool type. */
    public cl_mem allocateByteBuffer(int length) {
        return clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_char, null, null);
    }

    public void release(cl_mem buffer) {
        if (buffer != null)
            clReleaseMemObject(buffer);
    }

    public synchronized void uploadDoubles(cl_mem buffer, double[] host, int length) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_double, Pointer.to(host), 0, null, null);
    }

    public synchronized void downloadDoubles(cl_mem buffer, double[] host, int length) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_double, Pointer.to(host), 0, null, null);
    }

    public synchronized void uploadInts(cl_mem buffer, int[] host, int length) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_int, Pointer.to(host), 0, null, null);
    }

    public synchronized void downloadInts(cl_mem buffer, int[] host, int length) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_int, Pointer.to(host), 0, null, null);
    }

    public synchronized void uploadBytes(cl_mem buffer, byte[] host, int length) {
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_char, Pointer.to(host), 0, null, null);
    }

    public synchronized void downloadBytes(cl_mem buffer, byte[] host, int length) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, (long) length * Sizeof.cl_char, Pointer.to(host), 0, null, null);
    }

    // Single-element offset transfers, backing Gpu*Vector.get/set. A device
    // round-trip per call - fine for setup/debugging, not for hot loops
    // (bulk upload/download above is the fast path for that).

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

    public synchronized void fillByteBuffer(cl_mem buffer, byte value, int length) {
        if (length <= 0) return;
        clEnqueueFillBuffer(queue, buffer, Pointer.to(new byte[]{value}), Sizeof.cl_char, 0, (long) length * Sizeof.cl_char, 0, null, null);
    }

    // ---------------------------------------------------------------- kernel launches

    public synchronized void launchAxpy(cl_mem d, double a, cl_mem x, int size) {
        if (size <= 0) return;
        clSetKernelArg(kAxpy, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kAxpy, 1, Sizeof.cl_double, Pointer.to(new double[]{a}));
        clSetKernelArg(kAxpy, 2, Sizeof.cl_mem, Pointer.to(x));
        enqueueRange(kAxpy, size);
    }

    public synchronized void launchAddScalar(cl_mem d, double a, int size) {
        if (size <= 0) return;
        clSetKernelArg(kAddScalar, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kAddScalar, 1, Sizeof.cl_double, Pointer.to(new double[]{a}));
        enqueueRange(kAddScalar, size);
    }

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

    /** See the {@code scatter_axpy} kernel doc: caller must not pass duplicate targets in {@code idx}. */
    public synchronized void launchScatterAxpy(cl_mem d, double alpha, cl_mem s, cl_mem idx, int size) {
        if (size <= 0) return;
        clSetKernelArg(kScatterAxpy, 0, Sizeof.cl_mem, Pointer.to(d));
        clSetKernelArg(kScatterAxpy, 1, Sizeof.cl_double, Pointer.to(new double[]{alpha}));
        clSetKernelArg(kScatterAxpy, 2, Sizeof.cl_mem, Pointer.to(s));
        clSetKernelArg(kScatterAxpy, 3, Sizeof.cl_mem, Pointer.to(idx));
        enqueueRange(kScatterAxpy, size);
    }

    public synchronized double launchDot(cl_mem a, cl_mem b, int size) {
        if (size <= 0) return 0.0;
        return reduceDouble(kDotPartial, size, (kernel, argBase) -> {
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        });
    }

    public synchronized double launchSkippedDot(cl_mem a, cl_mem b, cl_mem unknownIdx, int size) {
        if (size <= 0) return 0.0;
        return reduceDouble(kSkippedDotPartial, size, (kernel, argBase) -> {
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(unknownIdx));
        });
    }

    @FunctionalInterface
    private interface DataArgBinder {
        void bind(cl_kernel kernel, int argBase);
    }

    /**
     * Shared plumbing for the two reduction kernels: both take a fixed set of
     * data args at the front (bound by {@code binder}), then
     * {@code (__local scratch, __global partials, const int n)} - this method
     * appends those three and does the group-count sizing / partial-sum work
     * common to both.
     */
    private double reduceDouble(cl_kernel kernel, int size, DataArgBinder binder) {
        int numGroups = (size + localSize - 1) / localSize;
        int paddedGlobalSize = numGroups * localSize;

        binder.bind(kernel, 0);

        int nextArg = countDataArgs(kernel);
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

    /** {@code dot_partial} has 2 data args (a, b); {@code skipped_dot_partial} has 3 (a, b, unknownIdx). */
    private int countDataArgs(cl_kernel kernel) {
        return kernel == kSkippedDotPartial ? 3 : 2;
    }

    private void enqueueRange(cl_kernel kernel, int size) {
        clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[]{size}, null, 0, null, null);
    }

    /** Blocks until every previously enqueued operation on this context's queue has completed. */
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
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
}