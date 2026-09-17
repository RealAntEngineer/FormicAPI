package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuIntegerVector extends CpuExecutable implements IntegerVector {

    private int[] data;
    private int   size;

    public CpuIntegerVector(int size) {
        this.data = new int[size];
        this.size = size;
    }

    public CpuIntegerVector(int[] data) {
        this.data = data;
        this.size = data.length;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public IntegerVector resize(int size) {
        if (size >= data.length)
            data = Arrays.copyOf(data, size);
        this.size = size;
        return this;
    }

    @Override
    public IntegerVector copy(Vector x) {
        if (!(x instanceof CpuIntegerVector vec))
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
    public IntegerVector copy() {
        IntegerVector vector = new CpuIntegerVector(Arrays.copyOf(data, data.length));
        //System.out.println("Size " + size);
        vector.resize(size);
        return vector;
    }

    @Override
    public IntegerVector clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, 0);
        return this;
    }

    //@Override
    public int[] array() {
        return data;
    }

    @Override
    public IntegerVector set(int value, int idx) {
        data[idx] = value;
        return this;
    }

    @Override
    public int get(int idx) {
        return data[idx];
    }


}