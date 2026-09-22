package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuDoubleVector extends CpuExecutable implements RealVector {

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
    public double dot(RealVector other) {
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
    public RealVector resize(int size) {
        if (size >= data.length)
            data = Arrays.copyOf(data, size);
        this.size = size;
        return this;
    }

    @Override
    public RealVector copy(Vector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        // System.arraycopy is already an intrinsic memcpy - splitting it
        // across threads doesn't win anything, so this stays serial.
        resize(vec.size);
        System.arraycopy(vec.data, 0, data, 0, vec.size);
        return this;
    }

    @Override
    public Vector copy() {
        Vector vector = new CpuDoubleVector(Arrays.copyOf(data, data.length));
        //System.out.println("Size " + size);
        vector.resize(size);
        return vector;
    }

    @Override
    public RealVector clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, 0.0);
        return this;
    }

    @Override
    public double skippedDot(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {

        if (!otherSkip && !thisSkip) {
            throw new IllegalArgumentException("When calling a skipping method at least one skip should be true");
        }

        CpuDoubleVector o = (CpuDoubleVector) other;
        final double[]  a = data;
        final double[]  b = o.data;
        final int       n = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                return executor.parallelReduceDouble(n, (start, end) -> {
                    double sum = 0.0;
                    for (int i = start; i < end; i++)
                        sum += a[unknowIdx.get(i)] * b[i];
                    return sum;
                });
            } else if (!thisSkip){
                return executor.parallelReduceDouble(n, (start, end) -> {
                    double sum = 0.0;
                    for (int i = start; i < end; i++)
                        sum += a[i] * b[unknowIdx.get(i)];
                    return sum;
                });
            } else {
                return executor.parallelReduceDouble(n, (start, end) -> {
                    double sum = 0.0;
                    for (int i = start; i < end; i++)
                        sum += a[unknowIdx.get(i)] * b[unknowIdx.get(i)];
                    return sum;
                });
            }
        }

        double sum = 0.0;
        if (!otherSkip) {
            for (int i = 0; i < n; i++)
                sum += a[unknowIdx.get(i)] * b[i];
        } else if (!thisSkip) {
            for (int i = 0; i < n; i++)
                sum += a[i] * b[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < n; i++)
                sum += a[unknowIdx.get(i)] * b[unknowIdx.get(i)];
        }
        return sum;
    }

    @Override
    public RealVector axpy(double a, RealVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += a * xd[i];
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] += a * xd[i];
        return this;
    }

    @Override
    public RealVector skippedAxpy(double alpha, RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
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
            return this;
        }

        if (!otherSkip) {
            for (int i = 0; i < n; i++)
                a[unknowIdx.get(i)] += alpha * b[i];
        } else if (!thisSkip) {
            for (int i = 0; i < n; i++)
                a[i] += alpha * b[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < n; i++)
                a[unknowIdx.get(i)] += alpha * b[unknowIdx.get(i)];
        }
        return this;
    }

    @Override
    public RealVector scale(double a) {
        final double[] d = data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] *= a;
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] *= a;
        return this;
    }

    @Override
    public RealVector scale(RealVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (x.size() < size)
            throw new UnsupportedOperationException("can't multiply by a vector smaller than us : "+x.size() + " < " +  size);

        final double[] d  = data;
        final double[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] *= xd[i];
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] *= xd[i];
        return this;
    }

    @Override
    public RealVector skippedScale(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        if (!(other instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + other.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;
        final int      n  = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] *= xd[i];
                });
            } else if (!thisSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[i] *= xd[unknowIdx.get(i)];
                });
            } else {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] *= xd[unknowIdx.get(i)];
                });
            }
            return this;
        }

        if (!otherSkip) {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] *= xd[i];
        } else if (!thisSkip) {
            for (int i = 0; i < n; i++)
                d[i] *= xd[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] *= xd[unknowIdx.get(i)];
        }
        return this;
    }

    @Override
    public RealVector divide(RealVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] /= xd[i];
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] /= xd[i];
        return this;
    }

    @Override
    public RealVector skippedDivide(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
        if (!(other instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + other.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;
        final int      n  = unknowIdx.size();

        if (executor != null && executor.shouldUseParallel(size())) {
            if (!otherSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] /= xd[i];
                });
            } else if (!thisSkip) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[i] /= xd[unknowIdx.get(i)];
                });
            } else {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[unknowIdx.get(i)] /= xd[unknowIdx.get(i)];
                });
            }
            return this;
        }

        if (!otherSkip) {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] /= xd[i];
        } else if (!thisSkip) {
            for (int i = 0; i < n; i++)
                d[i] /= xd[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] /= xd[unknowIdx.get(i)];
        }
        return this;
    }

    @Override
    public RealVector add(double a) {
        final double[] d = data;
        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += a;
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] += a;
        return this;
    }

    @Override
    public RealVector add(RealVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] += xd[i];
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] += xd[i];
        return this;
    }

    @Override
    public RealVector skippedAdd(RealVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
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
            } else if (!thisSkip) {
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
            return this;
        }

        if (!otherSkip) {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] += xd[i];
        } else if (!thisSkip) {
            for (int i = 0; i < n; i++)
                d[i] += xd[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] += xd[unknowIdx.get(i)];
        }
        return this;
    }

    @Override
    public RealVector subtract(RealVector x) {
        if (!(x instanceof CpuDoubleVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        final double[] d  = data;
        final double[] xd = vec.data;

        if (executor != null && executor.shouldUseParallel(size())) {
            executor.parallelFor(size, (start, end) -> {
                for (int i = start; i < end; i++)
                    d[i] -= xd[i];
            });
            return this;
        }

        for (int i = 0; i < size; i++)
            d[i] -= xd[i];
        return this;
    }

    @Override
    public RealVector skippedSubtract(RealVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip) {
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
            } else if (!thisSkip) {
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
            return this;
        }

        if (!otherSkip) {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] -= xd[i];
        } else if (!thisSkip) {
            for (int i = 0; i < n; i++)
                d[i] -= xd[unknowIdx.get(i)];
        } else {
            for (int i = 0; i < n; i++)
                d[unknowIdx.get(i)] -= xd[unknowIdx.get(i)];
        }
        return this;
    }

    @Override
    public RealVector gather(RealVector source, IntegerVector indices) {
        if (!(source instanceof CpuDoubleVector vec && indices instanceof CpuIntegerVector vecIdx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + source.getClass());

        final int n = indices.size();
        if (this.size() < n)
            throw new IllegalArgumentException(
                    "gather target must have >= indices.size() (" + n + ") elements, has " + this.size());

        final double[] xd = vec.data;
        final int[] id = vecIdx.array();
        final boolean aliased = (xd == this.data);

        if (!aliased) {
            final double[] d = this.data;
            if (executor != null && executor.shouldUseParallel(n)) {
                executor.parallelFor(n, (start, end) -> {
                    for (int i = start; i < end; i++)
                        d[i] = xd[id[i]];
                });
            } else {
                for (int i = 0; i < n; i++)
                    d[i] = xd[id[i]];
            }
            return this;
        }

        // Aliased: only safe to do in place if the permutation is monotone in
        // one direction relative to i -- id[i] >= i for every i (forward
        // pass safe, e.g. swap-with-las permutation),
        // or id[i] <= i for every i (backward pass safe).
        // Anything else risks reading a slot after it's already been
        // overwritten by an earlier iteration -- silently wrong, not a crash,
        // which is worse.
        boolean forwardSafe = true, backwardSafe = true;
        for (int i = 0; i < n && (forwardSafe || backwardSafe); i++) {
            if (id[i] < i) forwardSafe = false;
            if (id[i] > i) backwardSafe = false;
        }

        if (!forwardSafe && !backwardSafe)
            throw new IllegalArgumentException(
                    "gather: source and this share the same backing array, and this permutation " +
                            "is not safe to apply in place (neither id[i] >= i nor id[i] <= i holds for " +
                            "all i). This method never allocates to work around that. Use a WorkingBuffer " +
                            "instead: gather source into a scratch DoubleVector, then copy() the scratch " +
                            "into this.");

        // Both directions must stay single-threaded: the safety argument
        // depends on strict sequential order -- position id[i] is guaranteed
        // untouched only because everything before it in *this* traversal
        // order has already run. A parallel split would let one chunk read a
        // slot before the chunk responsible for writing it first has executed.

        final double[] d = this.data;
        if (forwardSafe) {
            for (int i = 0; i < n; i++) d[i] = xd[id[i]];
        } else {
            for (int i = n - 1; i >= 0; i--) d[i] = xd[id[i]];
        }
        return this;
    }

    public double[] array() {
        return data;
    }
}