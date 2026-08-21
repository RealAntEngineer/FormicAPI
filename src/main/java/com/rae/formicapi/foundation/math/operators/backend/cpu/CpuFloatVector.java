package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.FloatVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuFloatVector extends CpuExecutable implements FloatVector {

    private float[] data;
    private int     size;

    public CpuFloatVector(int size) {
        this.data = new float[size];
        this.size = size;
    }

    public CpuFloatVector(float[] data) {
        this.data = data;
        this.size = data.length;
    }

    @Override
    public float dot(FloatVector other) {
        CpuFloatVector o = (CpuFloatVector) other;
        final float[]  a = data;
        final float[]  b = o.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            return executor.parallelReduceFloat(size, (start, end) -> {
                float sum = 0.0f;
                for (int i = start; i < end; i++)
                    sum += a[i] * b[i];
                return sum;
            });
        }

        float sum = 0.0f;
        for (int i = 0; i < size; i++)
            sum += a[i] * b[i];
        return sum;
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
    public void copy(Vector x) {
        if (!(x instanceof CpuFloatVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        /*if (x.size() < this.size)
            throw new IllegalArgumentException("can't copy a smaller vector into itself");//no, you can resize.*/

        // System.arraycopy is already an intrinsic memcpy - splitting it
        // across threads doesn't win anything, so this stays serial.
        resize(vec.size);
        System.arraycopy(vec.data, 0, data, 0, vec.size);
    }

    @Override
    public Vector copy() {
        Vector vector = new CpuFloatVector(Arrays.copyOf(data, data.length));
        //System.out.println("Size " + size);
        vector.resize(size);
        return vector;
    }

    @Override
    public void clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, 0.0f);
    }

    @Override
    public float skippedDot(FloatVector other, IntegerVector unknowIdx) {
        CpuFloatVector o = (CpuFloatVector) other;
        final float[]  a = data;
        final float[]  b = o.data;
        final int      n = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            return executor.parallelReduceFloat(n, (start, end) -> {
                float sum = 0.0f;
                for (int i = start; i < end; i++)
                    sum += a[unknowIdx.get(i)] * b[i];
                return sum;
            });
        }

        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += a[unknowIdx.get(i)] * b[i];
        return sum;
    }

    @Override
    public void axpy(float a, FloatVector x) {
        if (!(x instanceof CpuFloatVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final float[] d  = data;
        final float[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
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
    public void scatterAxpy(float alpha, FloatVector source, IntegerVector idx) {
        if (!(source instanceof CpuFloatVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + source.getClass());

        final float[] d  = data;
        final float[] sd = vec.data;
        final int     n  = idx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(n, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[idx.get(i)] += alpha * sd[i];
            });
            return;
        }

        for (int i = 0; i < n; i++)
            d[idx.get(i)] += alpha * sd[i];
    }

    @Override
    public void scale(float a) {
        final float[] d = data;

        if (executor != null && executor.shouldUseParallel(size())) {
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
    public void add(float a) {
        final float[] d = data;
        if (executor != null && executor.shouldUseParallel(size())) {
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
    public void add(FloatVector x) {
        if (!(x instanceof CpuFloatVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final float[] d  = data;
        final float[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += xd[i];
            });
            return;
        }

        for (int i = 0; i < size; i++)
            d[i] += xd[i];
    }

    public float[] array() {
        return data;
    }
}