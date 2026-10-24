package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.vectors.BooleanVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

/**
 * GPU backend for {@link BooleanVector}. Stored as one {@code cl_char} per
 * element on device (0/1) since OpenCL has no packed device-side boolean
 * type - see {@link OpenCLGpuExecutor#allocateByteBuffer(int)}.
 *
 * <p>Same single-element-access caveat as {@link GpuIntegerVector}: {@link #get}/
 * {@link #set} are a device round-trip each, fine for setup/masking logic
 * that isn't in a hot loop.
 */
public final class GpuBooleanVector extends GpuExecutable implements BooleanVector {

    /**
     * {@code dst[i] = src[idx[i]]} on the one-byte-per-element device layout.
     * Needs 8-bit buffer access on Vulkan ({@code storageBuffer8BitAccess},
     * core since Vulkan 1.2); OpenCL's {@code char} needs nothing special.
     */
    private static final PortableKernel GATHER = new PortableKernel("gather_byte")
            .opencl("""
                    __kernel void gather_byte(__global char* dst, __global const char* src, __global const int* idx) {
                        int i = get_global_id(0);
                        dst[i] = src[idx[i]];
                    }
                    """)
            .glsl("""
                    layout(std430, binding = 0) buffer Dst { uint8_t dst[]; };
                    layout(std430, binding = 1) buffer Src { uint8_t src[]; };
                    layout(std430, binding = 2) buffer I   { int idx[]; };
                    layout(push_constant) uniform PC { int gsz; } pc;
                    void main() {
                        uint i = GID;
                        if (i >= uint(pc.gsz)) return;
                        dst[i] = src[idx[i]];
                    }
                    """, "storageBuffer8BitAccess");

    @Override
    protected void bindKernels(GpuExecutor executor) {
        // Only needed by gather(); a device without the capability still gets
        // every other boolean-vector operation, so don't fail construction.
        if (executor.supports(GATHER))
            GATHER.bind(executor);
    }

    private GpuResource buffer;
    private int    size;
    private int    capacity;

    public GpuBooleanVector(GpuExecutor executor, boolean[] hostData) {
        this(executor, hostData.length);
        upload(hostData);
        //requireExecutor().finish();
    }

    public GpuBooleanVector(GpuExecutor executor, int size) {
        setExecutor(executor);
        this.capacity = size;
        this.size = size;
        this.buffer = executor.allocateByteBuffer(Math.max(size, 1));
        executor.fillByteBuffer(buffer, (byte) 0, size);
        //requireExecutor().finish();
    }

    public void upload(boolean[] host) {
        if (host.length != size)
            throw new IllegalArgumentException("host array length " + host.length + " != vector size " + size);
        requireExecutor().uploadBytes(buffer, toBytes(host), size);
    }

    private static byte[] toBytes(boolean[] src) {
        byte[] out = new byte[src.length];
        for (int i = 0; i < src.length; i++)
            out[i] = (byte) (src[i] ? 1 : 0);
        return out;
    }

    public boolean[] download() {
        byte[] raw = new byte[size];
        requireExecutor().downloadBytes(buffer, raw, size);
        return toBooleans(raw);
    }

    private static boolean[] toBooleans(byte[] src) {
        boolean[] out = new boolean[src.length];
        for (int i = 0; i < src.length; i++)
            out[i] = src[i] != 0;
        return out;
    }

    @Override
    public BooleanVector set(boolean value, int idx) {
        requireExecutor().uploadByteAt(buffer, idx, (byte) (value ? 1 : 0));
        return this;
    }

    @Override
    public boolean get(int idx) {
        return requireExecutor().downloadByteAt(buffer, idx) != 0;
    }

    @Override
    public BooleanVector fill(boolean value, int fromInclusive, int toExclusive) {
        if (fromInclusive < 0 || toExclusive > size || fromInclusive > toExclusive)
            throw new IndexOutOfBoundsException("fill range [" + fromInclusive + ", " + toExclusive + ") on a vector of size " + size);
        requireExecutor().fillByteBuffer(buffer, fromInclusive, (byte) (value ? 1 : 0), toExclusive - fromInclusive);
        return this;
    }

    /**
     * {@code this[i] = source[indices[i]]} for {@code i in [0, indices.size())}.
     * Aliased ({@code source == this}) gathers go through a temporary buffer,
     * see {@link GpuDoubleVector#gather}.
     */
    @Override
    public BooleanVector gather(BooleanVector source, IntegerVector indices) {
        if (!(source instanceof GpuBooleanVector src) || src.executor != this.executor)
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + source.getClass());
        if (!(indices instanceof GpuIntegerVector idx) || idx.executor != this.executor)
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + indices.getClass());

        final int n = idx.size();
        if (size < n)
            throw new IllegalArgumentException("gather target must have >= indices.size() (" + n + ") elements, has " + size);
        if (n == 0)
            return this;

        GpuExecutor ex = requireExecutor();
        if (!ex.supports(GATHER))
            throw new UnsupportedOperationException("BooleanVector.gather needs 8-bit storage buffer access, which this device doesn't report");

        if (src.buffer != buffer) {
            GATHER.use(n, buffer, src.buffer, idx.buffer());
            return this;
        }

        GpuResource tmp = ex.allocateByteBuffer(n);
        try {
            GATHER.use(n, tmp, src.buffer, idx.buffer());
            ex.copyByteBuffer(tmp, buffer, n);
        } finally {
            tmp.release();
        }
        return this;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public BooleanVector resize(int newSize) {
        GpuExecutor ctx = requireExecutor();

        if (newSize > capacity) {
            GpuResource newBuffer = ctx.allocateByteBuffer(Math.max(newSize, 1));
            ctx.fillByteBuffer(newBuffer, (byte) 0, newSize);
            if (size > 0) // only the live elements: slots between size and capacity may be stale
                ctx.copyByteBuffer(buffer, newBuffer, Math.min(size, newSize));
            buffer.release();
            buffer = newBuffer;
            capacity = newSize;
        } else if (newSize > size) {
            // growing back inside existing capacity: re-zero slots that may hold stale values
            ctx.fillByteBuffer(buffer, size, (byte) 0, newSize - size);
        }

        size = newSize;
        return this;
    }

    @Override
    public BooleanVector copy(Vector x) {
        if (!(x instanceof GpuBooleanVector vec) || vec.executor != this.executor)
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        resize(vec.size);
        requireExecutor().copyByteBuffer(vec.buffer, buffer, vec.size);
        //requireExecutor().finish();
        return this;
    }

    @Override
    public BooleanVector copy() {
        GpuBooleanVector out = new GpuBooleanVector(requireExecutor(), size);
        requireExecutor().copyByteBuffer(buffer, out.buffer, size);
        //requireExecutor().finish();
        return out;
    }

    @Override
    public BooleanVector clear() {
        requireExecutor().fillByteBuffer(buffer, (byte) 0, size);
        //requireExecutor().finish();
        return this;
    }

    @Override
    public Vector internalCopy(int srcStart, int dstStart, int count) {
        if (count < 0 || srcStart < 0 || dstStart < 0 || srcStart + (long) count > size || dstStart + (long) count > size)
            throw new IndexOutOfBoundsException("internalCopy(" + srcStart + ", " + dstStart + ", " + count + ") on a vector of size " + size);
        requireExecutor().copyByteWithin(buffer, srcStart, dstStart, count);
        return this;
    }

    public void release() {
        buffer.release();
        buffer = null;
    }
}