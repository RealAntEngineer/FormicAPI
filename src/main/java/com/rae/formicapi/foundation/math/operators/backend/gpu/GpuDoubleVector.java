package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLResource;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

/**
 * GPU backend for {@link RealVector}. Data lives entirely on the device in
 * a {@link GpuResource} for the vector's whole lifetime.
 *
 * <p>Kernels are defined as static constants right here, next to the
 * methods that use them (see {@link OpenCLKernel}'s class doc). They're
 * bound once, eagerly, in {@link #bindKernels} -- called automatically by
 * {@link #setExecutor} -- so every arithmetic method below dispatches
 * directly with zero lookup. {@code dot}/{@code skippedDot} are not a
 * special case: their kernels are ordinary {@link Kernel}s whose result
 * happens to land in a small partials buffer passed as one more argument
 * (see {@link #reduceSum}), not a distinct "reduction" dispatch path.
 */
public final class GpuDoubleVector extends GpuExecutable implements RealVector, AutoCloseable {

    private static final int PREFERRED_LOCAL_SIZE = 256;

    /** Refined from {@link #PREFERRED_LOCAL_SIZE} against the real device the first time any vector binds; shared, since kernel binding already assumes one active executor at a time. */
    private static int localSize = PREFERRED_LOCAL_SIZE;

    private static final OpenCLKernel AXPY = new OpenCLKernel("axpy", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void axpy(__global double* d, const double a, __global const double* x) {
                int i = get_global_id(0);
                d[i] += a * x[i];
            }
            """, "cl_khr_fp64");

    private static final OpenCLKernel ADD_SCALAR = new OpenCLKernel("add_scalar", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void add_scalar(__global double* d, const double a) {
                int i = get_global_id(0);
                d[i] += a;
            }
            """, "cl_khr_fp64");

    private static final OpenCLKernel ADD_VECTOR = new OpenCLKernel("add_vector", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void add_vector(__global double* d, __global const double* x) {
                int i = get_global_id(0);
                d[i] += x[i];
            }
            """, "cl_khr_fp64");

    private static final OpenCLKernel SCALE = new OpenCLKernel("scale", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void scale(__global double* d, const double a) {
                int i = get_global_id(0);
                d[i] *= a;
            }
            """, "cl_khr_fp64");

    // NOTE: no atomics here - mirrors CpuDoubleVector.scatterAxpy's parallel
    // path, which also does a plain read-modify-write across worker threads
    // with no synchronization. Caller must not pass duplicate targets in
    // idx across work-items that can run concurrently, or this races.
    private static final OpenCLKernel SCATTER_AXPY = new OpenCLKernel("scatter_axpy", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void scatter_axpy(__global double* d, const double alpha, __global const double* s, __global const int* idx) {
                int i = get_global_id(0);
                int target = idx[i];
                d[target] += alpha * s[i];
            }
            """, "cl_khr_fp64");

    // Tree reduction within each work-group; writes one partial sum per
    // group into `partials`. reduceSum(...) below sums the (small)
    // partials array on the host -- the kernel itself is an ordinary
    // Kernel, nothing reduction-specific about the abstraction.
    private static final OpenCLKernel DOT_PARTIAL = new OpenCLKernel("dot_partial", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void dot_partial(__global const double* a, __global const double* b,
                                       __local double* scratch, __global double* partials, const int n) {
                int gid = get_global_id(0);
                int lid = get_local_id(0);
                int lsize = get_local_size(0);
                scratch[lid] = (gid < n) ? a[gid] * b[gid] : 0.0;
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int offset = lsize / 2; offset > 0; offset >>= 1) {
                    if (lid < offset) scratch[lid] += scratch[lid + offset];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (lid == 0) partials[get_group_id(0)] = scratch[0];
            }
            """, "cl_khr_fp64");

    private static final OpenCLKernel SKIPPED_DOT_PARTIAL = new OpenCLKernel("skipped_dot_partial", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void skipped_dot_partial(__global const double* a, __global const double* b,
                                               __global const int* unknownIdx,
                                               __local double* scratch, __global double* partials, const int n) {
                int gid = get_global_id(0);
                int lid = get_local_id(0);
                int lsize = get_local_size(0);
                scratch[lid] = (gid < n) ? a[unknownIdx[gid]] * b[gid] : 0.0;
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int offset = lsize / 2; offset > 0; offset >>= 1) {
                    if (lid < offset) scratch[lid] += scratch[lid + offset];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (lid == 0) partials[get_group_id(0)] = scratch[0];
            }
            """, "cl_khr_fp64");

    @Override
    protected void bindKernels(GpuExecutor executor) {
        AXPY.bind(executor);
        ADD_SCALAR.bind(executor);
        ADD_VECTOR.bind(executor);
        SCALE.bind(executor);
        SCATTER_AXPY.bind(executor);
        DOT_PARTIAL.bind(executor);
        SKIPPED_DOT_PARTIAL.bind(executor);

        localSize = largestPowerOfTwoLEQ((int) Math.min(PREFERRED_LOCAL_SIZE, executor.maxWorkGroupSize()));
    }

    private static int largestPowerOfTwoLEQ(int n) {
        int p = 1;
        while (p * 2 <= n)
            p *= 2;
        return Math.max(p, 1);
    }

    private GpuResource buffer;
    private int       size;
    private int       capacity;

    public GpuDoubleVector(OpenCLGpuExecutor executor, double[] hostData) {
        this(executor, hostData.length);
        upload(hostData);
        //requireExecutor().finish();
    }

    public GpuDoubleVector(OpenCLGpuExecutor executor, int size) {
        setExecutor(executor); // also binds kernels, see bindKernels above
        this.capacity = size;
        this.size = size;
        this.buffer = executor.allocateDoubleBuffer(Math.max(size, 1));
        executor.fillDoubleBuffer(buffer, 0.0, size);
        //requireExecutor().finish();
    }

    // ---------------------------------------------------------------- host interop

    /** Blocking copy of {@code host} onto the device, replacing this vector's contents. Does not resize. */
    public void upload(double[] host) {
        if (host.length != size)
            throw new IllegalArgumentException("host array length " + host.length + " != vector size " + size);
        requireExecutor().uploadDoubles(buffer, host, size);
    }

    /** Blocking copy of this vector's current contents back to the host. */
    public double[] download() {
        double[] out = new double[size];
        download(out);
        return out;
    }

    public void download(double[] host) {
        if (host.length != size)
            throw new IllegalArgumentException("host array length " + host.length + " != vector size " + size);
        requireExecutor().downloadDoubles(buffer, host, size);
    }

    /** Package-visible accessor so sibling Gpu*Vector types (e.g. index vectors) can be passed into kernel launches. */
    GpuResource buffer() {
        return buffer;
    }

    // ---------------------------------------------------------------- RealVector

    @Override
    public double dot(RealVector other) {
        GpuDoubleVector o = requireSameBackend(other);
        return reduceSum(DOT_PARTIAL, size, buffer, o.buffer);
    }

    @Override
    public double skippedDot(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip) {
        GpuDoubleVector o = requireSameBackend(other);
        if (!(unknowIdx instanceof GpuIntegerVector idx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + unknowIdx.getClass());

        return reduceSum(SKIPPED_DOT_PARTIAL, idx.size(), buffer, o.buffer, idx.buffer());
    }

    /**
     * Shared bookkeeping for {@code dot_partial}/{@code skipped_dot_partial}-shaped
     * kernels: appends the {@code (local scratch, partials buffer, n)}
     * triple every such kernel expects -- the partials buffer is passed as
     * an ordinary {@link OpenCLResource#buffer}, nothing special about it
     * from {@link Kernel}'s point of view -- dispatches, then downloads and
     * sums the (small) partials array on the host.
     */
    private double reduceSum(Kernel kernel, int size, GpuResource... dataArgs) {
        if (size <= 0) return 0.0;

        int numGroups        = (size + localSize - 1) / localSize;
        int paddedGlobalSize = numGroups * localSize;

        OpenCLGpuExecutor ex       = requireExecutor();
        GpuResource         partials = ex.allocateDoubleBuffer(numGroups);
        try {
            GpuResource[] args = new GpuResource[dataArgs.length + 3];
            System.arraycopy(dataArgs, 0, args, 0, dataArgs.length);
            args[dataArgs.length]     = OpenCLResource.local(executor, (long) localSize * Double.BYTES);
            args[dataArgs.length + 1] = partials;
            args[dataArgs.length + 2] = OpenCLResource.of(executor,size);

            // We're about to read the partials buffer this dispatch just
            // wrote to -- block on this specific command rather than relying
            // on downloadDoubles's own blocking transfer to cover it implicitly.
            kernel.useBlocking(paddedGlobalSize, args);

            double[] partialsHost = new double[numGroups];
            ex.downloadDoubles(partials, partialsHost, numGroups);

            double sum = 0.0;
            for (double v : partialsHost)
                sum += v;
            return sum;
        } finally {
            partials.release();
        }
    }

    @Override
    public RealVector axpy(double a, RealVector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        // Block: about to return `this` for the caller to read/release/reuse.
        AXPY.use(size, buffer, OpenCLResource.of(executor, a), vec.buffer);
        return this;
    }

    @Override
    public RealVector skippedAxpy(double alpha, RealVector other, IntegerVector unknowIdx, boolean thisSkipped, boolean otherSkipped) {
        GpuDoubleVector vec = requireSameBackend(other);
        if (!(unknowIdx instanceof GpuIntegerVector gpuIdx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + unknowIdx.getClass());

        SCATTER_AXPY.use(unknowIdx.size(), buffer, OpenCLResource.of(executor,alpha), vec.buffer, gpuIdx.buffer());
        return this;
    }

    @Override
    public RealVector scale(double a) {
        SCALE.use(size, buffer, OpenCLResource.of(executor, a));
        return this;
    }

    @Override
    public RealVector scale(RealVector x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector skippedScale(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector divide(RealVector x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector skippedDivide(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector add(double a) {
        ADD_SCALAR.use(size, buffer, OpenCLResource.of(executor, a));
        return this;
    }

    @Override
    public RealVector add(RealVector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        ADD_VECTOR.use(size, buffer, vec.buffer);
        return this;
    }

    @Override
    public RealVector skippedAdd(RealVector source, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector subtract(RealVector x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector skippedSubtract(RealVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RealVector gather(RealVector source, IntegerVector indices) {
        throw new UnsupportedOperationException();
    }

    private GpuDoubleVector requireSameBackend(Vector x) {
        if (!(x instanceof GpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());
        if (vec.executor != this.executor)
            throw new UnsupportedOperationException("GpuDoubleVector operands must share the same GpuExecutor");
        return vec;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public RealVector resize(int newSize) {
        OpenCLGpuExecutor ctx = requireExecutor();

        if (newSize > capacity) {
            GpuResource newBuffer = ctx.allocateDoubleBuffer(Math.max(newSize, 1));
            ctx.fillDoubleBuffer(newBuffer, 0.0, newSize);
            if (capacity > 0)
                ctx.copyDoubleBuffer(buffer, newBuffer, Math.min(capacity, newSize));
            buffer.release();
            buffer = newBuffer;
            capacity = newSize;
        }

        size = newSize;
        return this;
    }

    @Override
    public RealVector copy(Vector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        resize(vec.size);
        requireExecutor().copyDoubleBuffer(vec.buffer, buffer, vec.size);
        //requireExecutor().finish();
        return this;
    }

    @Override
    public Vector copy() {
        GpuDoubleVector out = new GpuDoubleVector(requireExecutor(), size);
        requireExecutor().copyDoubleBuffer(buffer, out.buffer, size);
        //requireExecutor().finish();
        return out;
    }

    @Override
    public RealVector clear() {
        requireExecutor().fillDoubleBuffer(buffer, 0.0, size);
        //requireExecutor().finish();
        return this;
    }

    /** Releases the underlying device buffer. The vector is unusable afterwards. */
    @Override
    public void close() {
        buffer.release();
        buffer = null;
    }
}