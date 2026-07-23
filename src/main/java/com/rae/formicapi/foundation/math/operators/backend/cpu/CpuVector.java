package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public final class CpuVector implements Vector {

    private double[] data;
    private int      size;

    public CpuVector(int size) {
        this.data = new double[size];
    }

    public CpuVector(double[] data) {
        this.data = data;
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

        double sum = 0.0;

        for (int i = 0; i < data.length; i++)
            sum += data[i] * o.data[i];

        return sum;
    }

    @Override
    public void axpy(double a, Vector x) {
        if (x instanceof CpuVector vec) {
            for (int i = 0; i < size; i++)
                data[i] += a * vec.data[i];
        } else {
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());
        }
    }

    @Override
    public void scale(double a) {

        for (int i = 0; i < size; i++)
            data[i] *= a;
    }

    @Override
    public void add(double a) {
        for (int i = 0; i < size; i++)
            data[i] += a;
    }

    @Override
    public void add(Vector x) {
        if (x instanceof CpuVector vec) {
            for (int i = 0; i < size; i++)
                data[i] += vec.data[i];
        } else {
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());
        }
    }

    @Override
    public void copy(Vector x) {
        if (x instanceof CpuVector vec) {
            if (x.size() < this.size) throw new IllegalArgumentException("can't copy a smaller vector into itself");
            System.arraycopy(vec.data, 0, data, 0, size);
        } else {
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());
        }
    }

    @Override
    public void clear() {
        Arrays.fill(data, 0.0);
    }

    public double[] array() {
        return data;
    }
}
