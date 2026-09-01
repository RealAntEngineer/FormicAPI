package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jocl.cl_mem;

/**
 * GPU backend for {@link DoubleVector}. Data lives entirely on the device in
 * a {@code cl_mem} buffer for the vector's whole lifetime; every arithmetic
 * op here is a thin wrapper that sets kernel args on the shared
 * {@link GpuExecutor} and enqueues a launch - no host round-trip on the hot
 * path.
 *
 * <p>Mirrors {@link com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector}'s
 * capacity/size split: {@link #resize} only reallocates the device buffer
 * when growing past current capacity (matching {@code Arrays.copyOf}'s
 * grow-and-zero-pad semantics), shrinking just adjusts the logical size.
 *
 * <p>{@link #copy(Vector)} and {@link #copy()} only interoperate with other
 * {@code GpuDoubleVector}s attached to the <b>same</b> {@link GpuExecutor} -
 * same restriction as the CPU vectors only interoperating with their own
 * type. Moving data between CPU and GPU backends is a separate, explicit
 * step via {@link #upload(double[])} / {@link #download(double[])}.
 */
public final class GpuDoubleVector extends GpuExecutable implements DoubleVector {

    private cl_mem buffer;
    private int    size;
    private int    capacity;

    public GpuDoubleVector(GpuExecutor executor, double[] hostData) {
        this(executor, hostData.length);
        upload(hostData);
        requireExecutor().finish();
    }

    public GpuDoubleVector(GpuExecutor executor, int size) {
        setExecutor(executor);
        this.capacity = size;
        this.size = size;
        this.buffer = executor.allocateDoubleBuffer(Math.max(size, 1));
        executor.fillDoubleBuffer(buffer, 0.0, size);
        requireExecutor().finish();
    }

    // ---------------------------------------------------------------- host interop

    /**
     * Blocking copy of {@code host} onto the device, replacing this vector's contents. Does not resize.
     */
    public void upload(double[] host) {
        if (host.length != size)
            throw new IllegalArgumentException("host array length " + host.length + " != vector size " + size);
        requireExecutor().uploadDoubles(buffer, host, size);
    }

    /**
     * Blocking copy of this vector's current contents back to the host.
     */
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

    /**
     * Package-visible accessor so sibling Gpu*Vector types (e.g. index vectors) can be passed into kernel launches.
     */
    cl_mem buffer() {
        return buffer;
    }

    // ---------------------------------------------------------------- DoubleVector

    @Override
    public double dot(DoubleVector other) {
        GpuDoubleVector o = requireSameBackend(other);
        return requireExecutor().launchDot(buffer, o.buffer, size);
    }

    @Override
    public double skippedDot(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip) {
        GpuDoubleVector o = requireSameBackend(other);
        if (!(unknowIdx instanceof GpuIntegerVector idx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + unknowIdx.getClass());

        return requireExecutor().launchSkippedDot(buffer, o.buffer, idx.buffer(), idx.size());
    }

    @Override
    public void axpy(double a, DoubleVector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        requireExecutor().launchAxpy(buffer, a, vec.buffer, size);
        requireExecutor().finish();
    }

    @Override
    public void skippedAxpy(double alpha, DoubleVector other, IntegerVector unknowIdx, boolean thisSkipped, boolean otherSkipped) {
        GpuDoubleVector vec = requireSameBackend(other);
        if (!(unknowIdx instanceof GpuIntegerVector gpuIdx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + unknowIdx.getClass());

        requireExecutor().launchScatterAxpy(buffer, alpha, vec.buffer, gpuIdx.buffer(), unknowIdx.size());
        requireExecutor().finish();
    }

    @Override
    public void scale(double a) {
        requireExecutor().launchScale(buffer, a, size);
        requireExecutor().finish();
    }

    @Override
    public void scale(DoubleVector x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void skippedScale(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void divide(DoubleVector x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void skippedDivide(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void add(double a) {
        requireExecutor().launchAddScalar(buffer, a, size);
        requireExecutor().finish();
    }

    @Override
    public void add(DoubleVector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        requireExecutor().launchAddVector(buffer, vec.buffer, size);
        requireExecutor().finish();
    }

    @Override
    public void skippedAdd(DoubleVector source, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void subtract(DoubleVector x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void skippedSubtract(DoubleVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        throw new UnsupportedOperationException();
    }

    //TODO this probably belong to Executable directly ? or maybe annotation ?
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
    public void resize(int newSize) {
        GpuExecutor ctx = requireExecutor();

        if (newSize > capacity) {
            cl_mem newBuffer = ctx.allocateDoubleBuffer(Math.max(newSize, 1));
            ctx.fillDoubleBuffer(newBuffer, 0.0, newSize);
            if (capacity > 0)
                ctx.copyDoubleBuffer(buffer, newBuffer, Math.min(capacity, newSize));
            ctx.release(buffer);
            buffer = newBuffer;
            capacity = newSize;
        }

        size = newSize;
    }

    @Override
    public void copy(Vector x) {
        GpuDoubleVector vec = requireSameBackend(x);
        resize(vec.size);
        requireExecutor().copyDoubleBuffer(vec.buffer, buffer, vec.size);
        requireExecutor().finish();
    }

    @Override
    public Vector copy() {
        GpuDoubleVector out = new GpuDoubleVector(requireExecutor(), size);
        requireExecutor().copyDoubleBuffer(buffer, out.buffer, size);
        requireExecutor().finish();
        return out;
    }

    @Override
    public void clear() {
        requireExecutor().fillDoubleBuffer(buffer, 0.0, size);
        requireExecutor().finish();
    }

    /**
     * Releases the underlying device buffer. The vector is unusable afterwards.
     */
    public void release() {
        if (executor != null)
            executor.release(buffer);
        buffer = null;
    }
}