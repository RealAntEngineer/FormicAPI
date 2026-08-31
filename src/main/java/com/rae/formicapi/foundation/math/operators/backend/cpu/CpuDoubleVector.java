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
    public double skippedDot(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {

        if (!otherSkip && !thisSkip) {
            throw new IllegalArgumentException("When calling a skipping method at least one skip should be true");
        }

        CpuDoubleVector o = (CpuDoubleVector) other;
        final double[]  a = data;
        final double[]  b = o.data;
        final int       n = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                return executor.parallelReduceDouble(unknowIdx.size(), (start, end) -> {
                    double sum = 0.0;
                    for (int i = start; i < end; i++)
                        sum += a[unknowIdx.get(i)] * b[i];
                    return sum;
                });
            } else if (!thisSkip){
                return executor.parallelReduceDouble(unknowIdx.size(), (start, end) -> {
                    double sum = 0.0;
                    for (int i = start; i < end; i++)
                        sum += a[i] * b[unknowIdx.get(i)];
                    return sum;
                });
            } else {
                return executor.parallelReduceDouble(unknowIdx.size(), (start, end) -> {
                    double sum = 0.0;
                    for (int i = start; i < end; i++)
                        sum += a[unknowIdx.get(i)] * b[unknowIdx.get(i)];
                    return sum;
                });
            }
        }

        double sum = 0.0;
        if (!otherSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                sum += a[unknowIdx.get(i)] * b[i];
        } else if (!thisSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                sum += a[i] * b[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < unknowIdx.size(); i++)
                sum += a[unknowIdx.get(i)] * b[unknowIdx.get(i)];
        }
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
    public void skippedAxpy(double alpha, DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        if (!(other instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + other.getClass());

        if (!otherSkip && !thisSkip)
            throw new IllegalArgumentException("When calling a skipping method at least one skip should be true");


        final double[] a  = data;
        final double[] b = vec.data;
        final int      n  = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        a[unknowIdx.get(i)] += alpha * b[i];
                });
            } else if (!thisSkip){
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        a[i] += alpha * b[unknowIdx.get(i)];
                });
            } else {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        a[unknowIdx.get(i)] += alpha * b[unknowIdx.get(i)];
                });
            }
            return;
        }

        if (!otherSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                a[unknowIdx.get(i)] += alpha * b[i];
        } else if (!thisSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                a[i] += alpha * b[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < unknowIdx.size(); i++)
                a[unknowIdx.get(i)] += alpha * b[unknowIdx.get(i)];
        }
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
    public void add(DoubleVector x) {
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

    @Override
    public void skippedAdd(DoubleVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;
        final int      n  = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] += xd[i];
                });
            } else if (!thisSkip){
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[i] += xd[unknowIdx.get(i)];
                });
            } else {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] += xd[unknowIdx.get(i)];
                });
            }
            return;
        }

        if (!otherSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                d[unknowIdx.get(i)] += xd[i];
        } else if (!thisSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                d[i] += xd[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < unknowIdx.size(); i++)
                d[unknowIdx.get(i)] += xd[unknowIdx.get(i)];
        }
    }

    @Override
    public void subtract(DoubleVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] -= xd[i];
            });
            return;
        }

        for (int i = 0; i < size; i++)
            d[i] += xd[i];
    }

    @Override
    public void skippedSubtract(DoubleVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;
        final int      n  = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] -= xd[i];
                });
            } else if (!thisSkip){
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[i] -= xd[unknowIdx.get(i)];
                });
            } else {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] -= xd[unknowIdx.get(i)];
                });
            }
            return;
        }

        if (!otherSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                d[unknowIdx.get(i)] -= xd[i];
        } else if (!thisSkip) {
            for (int i = 0; i < unknowIdx.size(); i++)
                d[i] -= xd[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < unknowIdx.size(); i++)
                d[unknowIdx.get(i)] -= xd[unknowIdx.get(i)];
        }

    }

    public double[] array() {
        return data;
    }
}