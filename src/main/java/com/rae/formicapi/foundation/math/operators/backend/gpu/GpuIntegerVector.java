package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jetbrains.annotations.Nullable;

/**
 * GPU backend for {@link IntegerVector}. Mainly exists to hold the index
 * buffers ({@code unknowIdx} in {@link RealVector#skippedDot}, {@code idx}
 * in {@link RealVector#skippedAxpy}) on-device so kernels can read them
 * directly rather than paying a host round-trip per call.
 *
 * <p>{@link #get(int)}/{@link #set(int, int)} are implemented via a
 * blocking 1-element transfer -- correct, but a device round-trip per
 * call. Fine for setup/debugging; for bulk index data prefer
 * {@link #upload(int[])} / {@link #download()}.
 */
public final class GpuIntegerVector extends GpuExecutable implements IntegerVector, AutoCloseable  {

    private @Nullable GpuResource buffer;
    private           int       size;
    private           int       capacity;

    public GpuIntegerVector(GpuExecutor executor, int[] hostData) {
        this(executor, hostData.length);
        upload(hostData);
        //requireExecutor().finish();
    }

    public GpuIntegerVector(GpuExecutor executor, int size) {
        setExecutor(executor);
        this.capacity = size;
        this.size = size;
        this.buffer = executor.allocateIntBuffer(Math.max(size, 1));
        // int buffers don't get zero-initialized here since there's currently
        // no consumer that needs it; callers populate via upload() before use.
    }

    public void upload(int[] host) {
        if (host.length != size)
            throw new IllegalArgumentException("host array length " + host.length + " != vector size " + size);
        requireExecutor().uploadInts(buffer, host, size);
        //requireExecutor().finish();
    }

    public int[] download() {
        int[] out = new int[size];
        requireExecutor().downloadInts(buffer, out, size);
        return out;
    }

    GpuResource buffer() {
        return buffer;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public IntegerVector resize(int newSize) {
        GpuExecutor ctx = requireExecutor();

        if (newSize >= capacity) {
            GpuResource newBuffer = ctx.allocateIntBuffer(Math.max(newSize, 1));
            if (capacity > 0)
                ctx.copyIntBuffer(buffer, newBuffer, Math.min(capacity, newSize));
            buffer.release();
            buffer = newBuffer;
            capacity = newSize;
        }

        size = newSize;
        return this;
    }

    @Override
    public IntegerVector copy(Vector x) {
        if (!(x instanceof GpuIntegerVector vec) || vec.executor != this.executor)
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        resize(vec.size);
        requireExecutor().copyIntBuffer(vec.buffer, buffer, vec.size);
        //requireExecutor().finish();
        return this;
    }

    @Override
    public IntegerVector copy() {
        GpuIntegerVector out = new GpuIntegerVector(requireExecutor(), size);
        requireExecutor().copyIntBuffer(buffer, out.buffer, size);
        //requireExecutor().finish();
        return out;
    }

    @Override
    public IntegerVector clear() {
        int[] zeros = new int[size];
        requireExecutor().uploadInts(buffer, zeros, size);
        //requireExecutor().finish();
        return this;
    }

    /** Single-element write - a device round-trip. Fine for setup/debugging; use {@link #upload(int[])} for bulk data. */
    @Override
    public IntegerVector set(int value, int idx) {
        requireExecutor().uploadIntAt(buffer, idx, value);
        return this;
    }

    /** Single-element read - a device round-trip. Fine for setup/debugging; use {@link #download()} for bulk data. */
    @Override
    public int get(int idx) {
        return requireExecutor().downloadIntAt(buffer, idx);
    }

    //TODO make release automatic on either executor crash or buffer destruction.
    public void close() {
        if (buffer != null) {
            buffer.release();
            buffer = null;
        }
    }
}