package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuDoubleVector extends CpuExecutable implements DoubleVector {

    private double[] data;
    private int      size;

    public CpuDoubleVector(int size) {
        this.data = new double[size];
        this.size = size;
    }

    public CpuDoubleVector(double[] data) {
        this.data = data;
        this.size = data.length;
    }

    @Override
    public double dot(DoubleVector other) {
        CpuDoubleVector o = (CpuDoubleVector) other;
        final double[]  a = data;
        final double[]  b = o.data;

        if (executor != null && executor.shouldUseParallel(size())) {
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
        if (!(x instanceof CpuDoubleVector vec))
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
        Vector vector = new CpuDoubleVector(Arrays.copyOf(data, data.length));
        //System.out.println("Size " + size);
        vector.resize(size);
        return vector;
    }

    @Override
    public void clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, 0.0);
    }

    @Override
    public double skippedDot(DoubleVector other, IntegerVector unknowIdx) {
        CpuDoubleVector o = (CpuDoubleVector) other;
        final double[]  a = data;
        final double[]  b = o.data;
        final int       n = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            return executor.parallelReduceDouble(unknowIdx.size(), (start, end) -> {
                double sum = 0.0;
                for (int i = start; i < end; i++)
                    sum += a[unknowIdx.get(i)] * b[i];
                return sum;
            });
        }

        double sum = 0.0;
        for (int i = 0; i < unknowIdx.size(); i++)
            sum += a[unknowIdx.get(i)] * b[i];
        return sum;
    }

    @Override
    public void axpy(double a, DoubleVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

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
    public void scatterAxpy(double alpha, DoubleVector source, IntegerVector unknowIdx) {
        if (!(source instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + source.getClass());

        final double[] d  = data;
        final double[] sd = vec.data;
        final int      n  = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(n, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[unknowIdx.get(i)] += alpha * sd[i];
            });
            return;
        }

        for (int i = 0; i < n; i++)
            d[unknowIdx.get(i)] += alpha * sd[i];
    }

    @Override
    public void scale(double a) {
        final double[] d = data;

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
    public void add(double a) {
        final double[] d = data;
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
    public void add(Vector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

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

    public double[] array() {
        return data;
    }
}