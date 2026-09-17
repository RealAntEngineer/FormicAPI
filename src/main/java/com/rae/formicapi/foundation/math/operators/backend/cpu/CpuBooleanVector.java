package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.BooleanVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuBooleanVector extends CpuExecutable implements BooleanVector {

    private boolean[] data;
    private int       size;

    public CpuBooleanVector(int size) {
        this.data = new boolean[size];
        this.size = size;
    }

    public CpuBooleanVector(boolean[] data) {
        this.data = data;
        this.size = data.length;
    }

    public boolean[] array() {
        return data;
    }

    @Override
    public BooleanVector set(boolean value, int idx) {
        data[idx] = value;
        return this;
    }

    @Override
    public boolean get(int idx) {
        return data[idx];
    }

    @Override
    public BooleanVector fill(boolean value, int fromInclusive, int toExclusive) {
        Arrays.fill(data, fromInclusive, toExclusive, false);
        return this;
    }

    @Override
    public BooleanVector gather(BooleanVector source, IntegerVector indices) {
        if (!(source instanceof CpuBooleanVector vec && indices instanceof CpuIntegerVector vecIdx))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + source.getClass());

        final int n = indices.size();
        if (this.size() < n)
            throw new IllegalArgumentException(
                    "gather target must have >= indices.size() (" + n + ") elements, has " + this.size());

        final boolean[] xd      = vec.data;
        final int[]     id      = vecIdx.array();
        final boolean   aliased = (xd == this.data);

        if (!aliased) {
            final boolean[] d = this.data;
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
        // pass safe, e.g. ParticleSystem.compact's swap-with-last
        // permutation), or id[i] <= i for every i (backward pass safe).
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
        final boolean[] d = this.data;
        if (forwardSafe) {
            for (int i = 0; i < n; i++) d[i] = xd[id[i]];
        } else {
            for (int i = n - 1; i >= 0; i--) d[i] = xd[id[i]];
        }
        return this;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public BooleanVector resize(int size) {
        if (size >= data.length)
            data = Arrays.copyOf(data, size);
        this.size = size;
        return this;
    }

    @Override
    public BooleanVector copy(Vector x) {
        if (!(x instanceof CpuBooleanVector vec))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        /*if (x.size() < this.size)
            throw new IllegalArgumentException("can't copy a smaller vector into itself");//no, you can resize.*/

        // System.arraycopy is already an intrinsic memcpy - splitting it
        // across threads doesn't win anything, so this stays serial.
        resize(vec.size);
        System.arraycopy(vec.data, 0, data, 0, vec.size);
        return this;
    }

    @Override
    public Vector copy() {
        Vector vector = new CpuBooleanVector(Arrays.copyOf(data, data.length));
        //System.out.println("Size " + size);
        vector.resize(size);
        return vector;
    }

    @Override
    public BooleanVector clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, false);
        return this;
    }
}