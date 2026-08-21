package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.BooleanVector;
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
        if (!(x instanceof CpuBooleanVector vec))
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
        Vector vector = new CpuBooleanVector(Arrays.copyOf(data, data.length));
        //System.out.println("Size " + size);
        vector.resize(size);
        return vector;
    }

    @Override
    public void clear() {
        // Same reasoning as copy(): Arrays.fill is already an optimized
        // intrinsic and is memory-bandwidth bound, not compute bound.
        Arrays.fill(data, false);
    }

    public boolean[] array() {
        return data;
    }

    @Override
    public void set(boolean value, int idx) {
        data[idx] = value;
    }

    @Override
    public boolean get(int idx) {
        return data[idx];
    }
}