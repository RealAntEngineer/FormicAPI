package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.vectors.BooleanVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jocl.cl_mem;

/**
 * GPU backend for {@link BooleanVector}. Stored as one {@code cl_char} per
 * element on device (0/1) since OpenCL has no packed device-side boolean
 * type - see {@link GpuExecutor#allocateByteBuffer(int)}.
 *
 * <p>Same single-element-access caveat as {@link GpuIntegerVector}: {@link #get}/
 * {@link #set} are a device round-trip each, fine for setup/masking logic
 * that isn't in a hot loop.
 */
public final class GpuBooleanVector extends GpuExecutable implements BooleanVector {

    private cl_mem buffer;
    private int    size;
    private int    capacity;

    public GpuBooleanVector(GpuExecutor executor, boolean[] hostData) {
        this(executor, hostData.length);
        upload(hostData);
    }

    public GpuBooleanVector(GpuExecutor executor, int size) {
        setExecutor(executor);
        this.capacity = size;
        this.size = size;
        this.buffer = executor.allocateByteBuffer(Math.max(size, 1));
        executor.fillByteBuffer(buffer, (byte) 0, size);
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
    public void set(boolean value, int idx) {
        requireExecutor().uploadByteAt(buffer, idx, (byte) (value ? 1 : 0));
    }

    @Override
    public boolean get(int idx) {
        return requireExecutor().downloadByteAt(buffer, idx) != 0;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void resize(int newSize) {
        GpuExecutor ctx = requireExecutor();

        if (newSize >= capacity) {
            cl_mem newBuffer = ctx.allocateByteBuffer(Math.max(newSize, 1));
            ctx.fillByteBuffer(newBuffer, (byte) 0, newSize);
            if (capacity > 0)
                ctx.copyByteBuffer(buffer, newBuffer, Math.min(capacity, newSize));
            ctx.release(buffer);
            buffer = newBuffer;
            capacity = newSize;
        }

        size = newSize;
    }

    @Override
    public void copy(Vector x) {
        if (!(x instanceof GpuBooleanVector vec) || vec.executor != this.executor)
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        resize(vec.size);
        requireExecutor().copyByteBuffer(vec.buffer, buffer, vec.size);
    }

    @Override
    public Vector copy() {
        GpuBooleanVector out = new GpuBooleanVector(requireExecutor(), size);
        requireExecutor().copyByteBuffer(buffer, out.buffer, size);
        return out;
    }

    @Override
    public void clear() {
        requireExecutor().fillByteBuffer(buffer, (byte) 0, size);
    }

    public void release() {
        if (executor != null)
            executor.release(buffer);
        buffer = null;
    }
}