package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

/**
 * GPU backend for {@link RealVector}. Data lives entirely on the device in
 * a {@link GpuResource} for the vector's whole lifetime.
 *
 * <p>Kernels are defined as static {@link PortableKernel} constants right
 * here, next to the methods that use them, with one OpenCL C source and one
 * GLSL source each so the class runs unchanged on every backend. They're
 * bound once, eagerly, in {@link #bindKernels} -- called automatically by
 * {@link #setExecutor} -- so every arithmetic method below dispatches
 * directly with zero lookup. {@code dot}/{@code skippedDot} are not a
 * special case: their kernels are ordinary {@link Kernel}s whose result
 * happens to land in a small partials buffer passed as one more argument
 * (see {@link #reduceSum}), not a distinct "reduction" dispatch path.
 *
 * <p><b>Skipped operations.</b> One kernel per operation family handles all
 * three {@code thisSkip}/{@code otherSkip} combinations: each work-item
 * computes {@code ti = thisSkip ? idx[i] : i} and
 * {@code oi = otherSkip ? idx[i] : i} and works on {@code this[ti]},
 * {@code other[oi]}. Like the CPU implementation, the scatter forms assume
 * the targets in {@code unknowIdx} are distinct -- duplicates race.
 */
public final class GpuDoubleVector extends GpuExecutable implements RealVector, AutoCloseable {

    private static final int PREFERRED_LOCAL_SIZE = 256;

    /** Refined from {@link #PREFERRED_LOCAL_SIZE} against the real device the first time any vector binds; shared, since kernel binding already assumes one active executor at a time. */
    private static int localSize = PREFERRED_LOCAL_SIZE;

    // Operation codes shared by BINARY_OP / SKIPPED_BINARY (both languages).
    private static final int OP_ADD = 0;
    private static final int OP_SUB = 1;
    private static final int OP_MUL = 2;
    private static final int OP_DIV = 3;

    private static final PortableKernel AXPY = new PortableKernel("axpy")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void axpy(__global double* d, const double a, __global const double* x) {
                        int i = get_global_id(0);
                        d[i] += a * x[i];
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(std430, binding = 1) buffer X { double x[]; };
                    layout(push_constant) uniform PC { int gsz; double a; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        d[i] += pc.a * x[i];
                    }
                    """, "shaderFloat64");

    private static final PortableKernel ADD_SCALAR = new PortableKernel("add_scalar")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void add_scalar(__global double* d, const double a) {
                        int i = get_global_id(0);
                        d[i] += a;
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(push_constant) uniform PC { int gsz; double a; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        d[i] += pc.a;
                    }
                    """, "shaderFloat64");

    private static final PortableKernel ADD_VECTOR = new PortableKernel("add_vector")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void add_vector(__global double* d, __global const double* x) {
                        int i = get_global_id(0);
                        d[i] += x[i];
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(std430, binding = 1) buffer X { double x[]; };
                    layout(push_constant) uniform PC { int gsz; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        d[i] += x[i];
                    }
                    """, "shaderFloat64");

    private static final PortableKernel SCALE = new PortableKernel("scale")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void scale(__global double* d, const double a) {
                        int i = get_global_id(0);
                        d[i] *= a;
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(push_constant) uniform PC { int gsz; double a; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        d[i] *= pc.a;
                    }
                    """, "shaderFloat64");

    /** {@code d[i] = d[i] <op> x[i]} for op in {@code SUB, MUL, DIV}; safe with {@code d == x} (each work-item only touches index i). */
    private static final PortableKernel BINARY_OP = new PortableKernel("binary_op")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void binary_op(__global double* d, __global const double* x, const int op) {
                        int i = get_global_id(0);
                        double a = d[i];
                        double b = x[i];
                        d[i] = (op == 0) ? a + b : (op == 1) ? a - b : (op == 2) ? a * b : a / b;
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(std430, binding = 1) buffer X { double x[]; };
                    layout(push_constant) uniform PC { int gsz; int op; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        double a = d[i];
                        double b = x[i];
                        d[i] = (pc.op == 0) ? a + b : (pc.op == 1) ? a - b : (pc.op == 2) ? a * b : a / b;
                    }
                    """, "shaderFloat64");

    /**
     * {@code this[ti] = this[ti] <op> other[oi]} over the {@code n} entries of
     * {@code idx}, with {@code ti}/{@code oi} remapped through {@code idx} per
     * the two skip flags (see class doc).
     *
     * <p>No atomics -- mirrors the CPU implementation, whose parallel path also
     * does a plain read-modify-write per target: the caller must not pass
     * duplicate targets in {@code idx}.
     */
    private static final PortableKernel SKIPPED_BINARY = new PortableKernel("skipped_binary")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void skipped_binary(__global double* d, __global const double* x, __global const int* idx,
                                                 const int op, const int thisSkip, const int otherSkip) {
                        int i = get_global_id(0);
                        int ti = thisSkip  != 0 ? idx[i] : i;
                        int oi = otherSkip != 0 ? idx[i] : i;
                        double a = d[ti];
                        double b = x[oi];
                        d[ti] = (op == 0) ? a + b : (op == 1) ? a - b : (op == 2) ? a * b : a / b;
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(std430, binding = 1) buffer X { double x[]; };
                    layout(std430, binding = 2) buffer I { int idx[]; };
                    layout(push_constant) uniform PC { int gsz; int op; int thisSkip; int otherSkip; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        uint ti = pc.thisSkip  != 0 ? uint(idx[i]) : i;
                        uint oi = pc.otherSkip != 0 ? uint(idx[i]) : i;
                        double a = d[ti];
                        double b = x[oi];
                        d[ti] = (pc.op == 0) ? a + b : (pc.op == 1) ? a - b : (pc.op == 2) ? a * b : a / b;
                    }
                    """, "shaderFloat64");

    private static final PortableKernel SKIPPED_AXPY = new PortableKernel("skipped_axpy")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void skipped_axpy(__global double* d, const double alpha, __global const double* x,
                                               __global const int* idx, const int thisSkip, const int otherSkip) {
                        int i = get_global_id(0);
                        int ti = thisSkip  != 0 ? idx[i] : i;
                        int oi = otherSkip != 0 ? idx[i] : i;
                        d[ti] += alpha * x[oi];
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer D { double d[]; };
                    layout(std430, binding = 1) buffer X { double x[]; };
                    layout(std430, binding = 2) buffer I { int idx[]; };
                    layout(push_constant) uniform PC { int gsz; double alpha; int thisSkip; int otherSkip; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        uint ti = pc.thisSkip  != 0 ? uint(idx[i]) : i;
                        uint oi = pc.otherSkip != 0 ? uint(idx[i]) : i;
                        d[ti] += pc.alpha * x[oi];
                    }
                    """, "shaderFloat64");

    /** {@code dst[i] = src[idx[i]]} for {@code i in [0, n)}. */
    private static final PortableKernel GATHER = new PortableKernel("gather_double")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void gather_double(__global double* dst, __global const double* src, __global const int* idx) {
                        int i = get_global_id(0);
                        dst[i] = src[idx[i]];
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer Dst { double dst[]; };
                    layout(std430, binding = 1) buffer Src { double src[]; };
                    layout(std430, binding = 2) buffer I   { int idx[]; };
                    layout(push_constant) uniform PC { int gsz; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        dst[i] = src[idx[i]];
                    }
                    """, "shaderFloat64");

    // Tree reduction within each work-group; writes one partial sum per
    // group into `partials`. reduceSum(...) below sums the (small)
    // partials array on the host -- the kernel itself is an ordinary
    // Kernel, nothing reduction-specific about the abstraction.
    private static final PortableKernel DOT_PARTIAL = new PortableKernel("dot_partial")
            .opencl("""
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
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer A { double a[]; };
                    layout(std430, binding = 1) buffer B { double b[]; };
                    layout(std430, binding = 2) buffer P { double partials[]; };
                    layout(push_constant) uniform PC { int gsz; int n; } pc;
                    shared double scratch[gl_WorkGroupSize.x];
                    void main() {
                        uint gid = GID;
                        uint lid = gl_LocalInvocationID.x;
                        scratch[lid] = (gid < uint(pc.n)) ? a[gid] * b[gid] : 0.0LF;
                        memoryBarrierShared();
                        barrier();
                        for (uint offset = gl_WorkGroupSize.x / 2u; offset > 0u; offset >>= 1u) {
                            if (lid < offset) scratch[lid] += scratch[lid + offset];
                            memoryBarrierShared();
                            barrier();
                        }
                        if (lid == 0u) partials[GROUP_ID] = scratch[0];
                    }
                    """, "shaderFloat64");

    /** Reduced-space dot: {@code sum a[thisSkip ? idx[i] : i] * b[otherSkip ? idx[i] : i]} over {@code i in [0, n)}, {@code n = idx.size()}. */
    private static final PortableKernel SKIPPED_DOT_PARTIAL = new PortableKernel("skipped_dot_partial")
            .opencl("""
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    __kernel void skipped_dot_partial(__global const double* a, __global const double* b,
                                                       __global const int* idx,
                                                       __local double* scratch, __global double* partials, const int n,
                                                       const int thisSkip, const int otherSkip) {
                        int gid = get_global_id(0);
                        int lid = get_local_id(0);
                        int lsize = get_local_size(0);
                        double v = 0.0;
                        if (gid < n) {
                            int ai = thisSkip  != 0 ? idx[gid] : gid;
                            int bi = otherSkip != 0 ? idx[gid] : gid;
                            v = a[ai] * b[bi];
                        }
                        scratch[lid] = v;
                        barrier(CLK_LOCAL_MEM_FENCE);
                        for (int offset = lsize / 2; offset > 0; offset >>= 1) {
                            if (lid < offset) scratch[lid] += scratch[lid + offset];
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (lid == 0) partials[get_group_id(0)] = scratch[0];
                    }
                    """, "cl_khr_fp64")
            .glsl("""
                    layout(std430, binding = 0) buffer A { double a[]; };
                    layout(std430, binding = 1) buffer B { double b[]; };
                    layout(std430, binding = 2) buffer I { int idx[]; };
                    layout(std430, binding = 3) buffer P { double partials[]; };
                    layout(push_constant) uniform PC { int gsz; int n; int thisSkip; int otherSkip; } pc;
                    shared double scratch[gl_WorkGroupSize.x];
                    void main() {
                        uint gid = GID;
                        uint lid = gl_LocalInvocationID.x;
                        double v = 0.0LF;
                        if (gid < uint(pc.n)) {
                            uint ai = pc.thisSkip  != 0 ? uint(idx[gid]) : gid;
                            uint bi = pc.otherSkip != 0 ? uint(idx[gid]) : gid;
                            v = a[ai] * b[bi];
                        }
                        scratch[lid] = v;
                        memoryBarrierShared();
                        barrier();
                        for (uint offset = gl_WorkGroupSize.x / 2u; offset > 0u; offset >>= 1u) {
                            if (lid < offset) scratch[lid] += scratch[lid + offset];
                            memoryBarrierShared();
                            barrier();
                        }
                        if (lid == 0u) partials[GROUP_ID] = scratch[0];
                    }
                    """, "shaderFloat64");

    @Override
    protected void bindKernels(GpuExecutor executor) {
        AXPY.bind(executor);
        ADD_SCALAR.bind(executor);
        ADD_VECTOR.bind(executor);
        SCALE.bind(executor);
        BINARY_OP.bind(executor);
        SKIPPED_BINARY.bind(executor);
        SKIPPED_AXPY.bind(executor);
        GATHER.bind(executor);
        DOT_PARTIAL.bind(executor);
        SKIPPED_DOT_PARTIAL.bind(executor);

        // The reduction kernels size their scratch/partials from localSize, so
        // it must be a power of two that BOTH the device and these specific
        // kernels (register pressure can cap a kernel below the device max)
        // can actually launch with.
        long limit = Math.min(PREFERRED_LOCAL_SIZE, executor.maxWorkGroupSize());
        limit = Math.min(limit, Math.min(DOT_PARTIAL.maxLocalSize(), SKIPPED_DOT_PARTIAL.maxLocalSize()));
        localSize = largestPowerOfTwoLEQ((int) limit);
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

    public GpuDoubleVector(GpuExecutor executor, double[] hostData) {
        this(executor, hostData.length);
        upload(hostData);
        //requireExecutor().finish();
    }

    public GpuDoubleVector(GpuExecutor executor, int size) {
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
        requireAtLeast(o, size);
        return reduceSum(DOT_PARTIAL, size, new GpuResource[]{buffer, o.buffer});
    }

    @Override
    public double skippedDot(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip) {
        requireSkip(thisSkip, sourceSkip);
        GpuDoubleVector o = requireSameBackend(other);
        GpuIntegerVector idx = requireGpuIndex(unknowIdx);

        GpuExecutor ex = requireExecutor();
        return reduceSum(SKIPPED_DOT_PARTIAL, idx.size(), new GpuResource[]{buffer, o.buffer, idx.buffer()},
                ex.scalar(thisSkip ? 1 : 0), ex.scalar(sourceSkip ? 1 : 0));
    }

    /**
     * Shared bookkeeping for {@code dot_partial}/{@code skipped_dot_partial}-shaped
     * kernels: appends the {@code (local scratch, partials buffer, n)}
     * triple every such kernel expects -- the partials buffer is passed as
     * an ordinary buffer argument, nothing special about it from
     * {@link Kernel}'s point of view -- then any {@code trailing} scalar
     * arguments, dispatches with the explicit work-group size the kernel's
     * scratch and partials were sized for, then downloads and sums the
     * (small) partials array on the host.
     */
    private double reduceSum(Kernel kernel, int n, GpuResource[] dataArgs, GpuResource... trailing) {
        if (n <= 0) return 0.0;

        int numGroups        = (n + localSize - 1) / localSize;
        int paddedGlobalSize = numGroups * localSize;

        GpuExecutor ex       = requireExecutor();
        GpuResource partials = ex.allocateDoubleBuffer(numGroups);
        try {
            GpuResource[] args = new GpuResource[dataArgs.length + 3 + trailing.length];
            System.arraycopy(dataArgs, 0, args, 0, dataArgs.length);
            args[dataArgs.length]     = ex.localMemory((long) localSize * Double.BYTES);
            args[dataArgs.length + 1] = partials;
            args[dataArgs.length + 2] = ex.scalar(n);
            System.arraycopy(trailing, 0, args, dataArgs.length + 3, trailing.length);

            // We're about to read the partials buffer this dispatch just
            // wrote to -- block on this specific command rather than relying
            // on downloadDoubles's own blocking transfer to cover it implicitly.
            kernel.useBlocking(paddedGlobalSize, localSize, args);

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
        requireAtLeast(vec, size);
        AXPY.use(size, buffer, requireExecutor().scalar(a), vec.buffer);
        return this;
    }

    @Override
    public RealVector skippedAxpy(double alpha, RealVector other, IntegerVector unknowIdx, boolean thisSkipped, boolean otherSkipped) {
        requireSkip(thisSkipped, otherSkipped);
        GpuDoubleVector vec = requireSameBackend(other);
        GpuIntegerVector gpuIdx = requireGpuIndex(unknowIdx);

        GpuExecutor ex = requireExecutor();
        SKIPPED_AXPY.use(gpuIdx.size(), buffer, ex.scalar(alpha), vec.buffer, gpuIdx.buffer(),
                ex.scalar(thisSkipped ? 1 : 0), ex.scalar(otherSkipped ? 1 : 0));
        return this;
    }

    @Override
    public RealVector scale(double a) {
        SCALE.use(size, buffer, requireExecutor().scalar(a));
        return this;
    }

    @Override
    public RealVector scale(RealVector x) {
        return binary(x, OP_MUL);
    }

    @Override
    public RealVector skippedScale(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        return skippedBinary(other, unknowIdx, thisSkip, otherSkip, OP_MUL);
    }

    @Override
    public RealVector divide(RealVector x) {
        return binary(x, OP_DIV);//TODO inline, we are creating indirection on something completely trivial.
    }

    @Override
    public RealVector skippedDivide(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        return skippedBinary(other, unknowIdx, thisSkip, otherSkip, OP_DIV);
    }

    @Override
    public RealVector set(double value, int idx) {
        requireExecutor().uploadDoubleAt(buffer, idx, value);
        return this;
    }

    /** Single-element read - a device round-trip. Fine for setup/debugging; use {@link #download()} for bulk data. */
    @Override
    public double get(int idx) {
        return requireExecutor().downloadDoubleAt(buffer, idx);
    }

    /** {@code this[idx] += value} -- a read-modify-write round trip (two blocking transfers); fine for setup, use {@link #skippedAxpy} / {@link #add} for bulk work. */
    @Override
    public RealVector setAdd(double value, int idx) {
        GpuExecutor ex = requireExecutor();
        ex.uploadDoubleAt(buffer, idx, ex.downloadDoubleAt(buffer, idx) + value);
        return this;
    }

    @Override
    public RealVector add(double a) {
        ADD_SCALAR.use(size, buffer, requireExecutor().scalar(a));
        return this;
    }

    @Override
    public RealVector add(RealVector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        requireAtLeast(vec, size);
        ADD_VECTOR.use(size, buffer, vec.buffer);
        return this;
    }

    @Override
    public RealVector skippedAdd(RealVector source, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        return skippedBinary(source, unknowIdx, thisSkip, otherSkip, OP_ADD);
    }

    @Override
    public RealVector subtract(RealVector x) {
        return binary(x, OP_SUB);
    }

    @Override
    public RealVector skippedSubtract(RealVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        return skippedBinary(x, unknowIdx, thisSkip, otherSkip, OP_SUB);
    }

    /**
     * {@code this[i] = source[indices[i]]} for {@code i in [0, indices.size())}.
     *
     * <p>When {@code source} is this very vector the work-items would race
     * (one may overwrite a slot another still has to read), so that case
     * gathers into a temporary device buffer first and copies it back --
     * the contract {@link RealVector#gather} asks for, and, unlike the CPU
     * version, it works for any permutation rather than only monotone ones.
     */
    @Override
    public RealVector gather(RealVector source, IntegerVector indices) {
        GpuDoubleVector src = requireSameBackend(source);
        GpuIntegerVector idx = requireGpuIndex(indices);

        final int n = idx.size();
        if (size < n)
            throw new IllegalArgumentException("gather target must have >= indices.size() (" + n + ") elements, has " + size);
        if (n == 0)
            return this;

        GpuExecutor ex = requireExecutor();
        if (src.buffer != buffer) {
            GATHER.use(n, buffer, src.buffer, idx.buffer());
            return this;
        }

        GpuResource tmp = ex.allocateDoubleBuffer(n);
        try {
            GATHER.use(n, tmp, src.buffer, idx.buffer());
            ex.copyDoubleBuffer(tmp, buffer, n);
        } finally {
            tmp.release(); // the backend defers destruction until the queued work using it is done
        }
        return this;
    }

    private RealVector binary(RealVector x, int op) {
        GpuDoubleVector vec = requireSameBackend(x);
        requireAtLeast(vec, size);
        BINARY_OP.use(size, buffer, vec.buffer, requireExecutor().scalar(op));
        return this;
    }

    private RealVector skippedBinary(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip, int op) {
        requireSkip(thisSkip, otherSkip);
        GpuDoubleVector vec = requireSameBackend(other);
        GpuIntegerVector idx = requireGpuIndex(unknowIdx);

        GpuExecutor ex = requireExecutor();
        SKIPPED_BINARY.use(idx.size(), buffer, vec.buffer, idx.buffer(),
                ex.scalar(op), ex.scalar(thisSkip ? 1 : 0), ex.scalar(otherSkip ? 1 : 0));
        return this;
    }

    private static void requireSkip(boolean thisSkip, boolean otherSkip) {
        if (!thisSkip && !otherSkip)
            throw new IllegalArgumentException("When calling a skipping method at least one skip should be true");
    }

    private GpuIntegerVector requireGpuIndex(IntegerVector idx) {
        if (!(idx instanceof GpuIntegerVector gpuIdx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + idx.getClass());
        if (gpuIdx.executor != this.executor)
            throw new UnsupportedOperationException("GpuDoubleVector and its index vector must share the same GpuExecutor");
        return gpuIdx;
    }

    private static void requireAtLeast(GpuDoubleVector vec, int n) {
        if (vec.size < n)
            throw new IllegalArgumentException("operand has " + vec.size + " elements, need at least " + n);
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
        GpuExecutor ctx = requireExecutor();

        if (newSize > capacity) {
            GpuResource newBuffer = ctx.allocateDoubleBuffer(Math.max(newSize, 1));
            ctx.fillDoubleBuffer(newBuffer, 0.0, newSize);
            if (size > 0) // only the live elements: slots between size and capacity may be stale
                ctx.copyDoubleBuffer(buffer, newBuffer, Math.min(size, newSize));
            buffer.release();
            buffer = newBuffer;
            capacity = newSize;
        } else if (newSize > size) {
            // Growing back inside already-allocated capacity: those slots may
            // hold stale values from before an earlier shrink, but the contract
            // is that new elements read as zero.
            ctx.fillDoubleBuffer(buffer, size, 0.0, newSize - size);
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

    @Override
    public Vector internalCopy(int srcStart, int dstStart, int count) {
        if (count < 0 || srcStart < 0 || dstStart < 0 || srcStart + (long) count > size || dstStart + (long) count > size)
            throw new IndexOutOfBoundsException("internalCopy(" + srcStart + ", " + dstStart + ", " + count + ") on a vector of size " + size);
        requireExecutor().copyDoubleWithin(buffer, srcStart, dstStart, count);
        return this;
    }

    /** Releases the underlying device buffer. The vector is unusable afterwards. */
    @Override
    public void close() {
        buffer.release();
        buffer = null;
    }
}
