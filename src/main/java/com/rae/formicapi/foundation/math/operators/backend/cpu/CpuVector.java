package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuVector implements Vector {

    private double[] data;
    private int      size;

    /** Optional. When null, every operation falls back to a plain serial loop. */
    private CpuExecutor executor;

    /**
     * Per-vector override for the parallel dispatch threshold. -1 (default)
     * means "defer to whatever the attached executor is configured with" via
     * {@link CpuExecutor#getParallelThreshold()} - which itself defaults to
     * {@link CpuExecutor#DEFAULT_PARALLEL_THRESHOLD} unless the executor was
     * tuned with {@link CpuExecutor#calibrateThreshold(int)}. Set this only
     * if a particular vector's workload genuinely behaves differently from
     * the executor's general-purpose calibration.
     */
    private int parallelThresholdOverride = -1;

    public CpuVector(int size) {
        this.data = new double[size];
        this.size = size;
    }

    public CpuVector(double[] data) {
        this.data = data;
        this.size = data.length;
    }

    /** Attach an executor to enable parallel execution for large vectors. */
    public void setExecutor(CpuExecutor executor) {
        this.executor = executor;
    }

    /** Override the parallel dispatch threshold for this vector specifically. Pass -1 to clear the override. */
    public void setParallelThreshold(int threshold) {
        this.parallelThresholdOverride = threshold;
    }

    private boolean useParallel() {
        if (executor == null)
            return false;

        int threshold = parallelThresholdOverride >= 0
                ? parallelThresholdOverride
                : executor.getParallelThreshold();

        return size >= threshold;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void resize(int size) {
        if (size >= data.length)
            data = Arrays.copyOf(data, size);
        this.size = size;
    }

    @Override
    public double dot(Vector other) {
        CpuVector o = (CpuVector) other;
        final double[] a = data;
        final double[] b = o.data;

        if (useParallel()) {
            return executor.parallelReduceDouble(size, (start, end) -> {
                double sum = 0.0;
                for (int i = start; i < end; i++)
                    sum += a[i] * b[i];
                return sum;
            });
        }

        double sum = 0.0;
        for (int i = 0; i < size; i++)
            sum += a[i] * b[i];
        return sum;
    }

    @Override
    public void axpy(double a, Vector x) {
        if (!(x instanceof CpuVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d = data;
        final double[] xd = vec.data;

        if (useParallel()) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += a * xd[i];
            });
            return;
        }

        for (int i = 0; i < size; i++)
            d[i] += a * xd[i];
    }

    @Override
    public void scale(double a) {
        final double[] d = data;

        if (useParallel()) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] *= a;
            });
            return;
        }

        for (int i = 0; i < size; i++)
            d[i] *= a;
    }

    @Override
    public void add(double a) {
        final double[] d = data;

        if (useParallel()) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += a;
            });
            return;
        }

        for (int i = 0; i < size; i++)
            d[i] += a;
    }

    @Override
    public void add(Vector x) {
        if (!(x instanceof CpuVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d = data;
        final double[] xd = vec.data;

        if (useParallel()) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += xd[i];
            });
            return;
        }

        for (int i = 0; i < size; i++)
            d[i] += xd[i];
    }

    @Override
    public void copy(Vector x) {
        if (!(x instanceof CpuVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (x.size() < this.size)
            throw new IllegalArgumentException("can't copy a smaller vector into itself");

        // System.arraycopy is already an intrinsic memcpy - splitting it
        // across threads doesn't win anything, so this stays serial.
        System.arraycopy(vec.data, 0, data, 0, size);
    }

    @Override
    public void clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, 0.0);
    }

    public double[] array() {
        return data;
    }
}