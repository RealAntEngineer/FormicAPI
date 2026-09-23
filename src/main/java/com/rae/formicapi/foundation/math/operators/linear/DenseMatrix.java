package com.rae.formicapi.foundation.math.operators.linear;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;

public class DenseMatrix implements MutableMatrix {

    private final double[][] data;

    public DenseMatrix(int rows, int cols) {
        data = new double[rows][cols];
    }

    @Override
    public void add(int r, int c, double v) {
        data[r][c] += v;
    }

    @Override
    public void set(int r, int c, double v) {
        data[r][c] = v;
    }

    @Override
    public void multiply(double[] x, double[] result) {
        // Check input vector size
        if (x.length != cols()) {
            throw new IllegalArgumentException(
                    "Cannot multiply: input vector length (" + x.length +
                            ") does not match number of matrix columns (" + cols() + ")"
            );
        }

        // Check result array size
        if (result.length != rows()) {
            throw new IllegalArgumentException(
                    "Cannot store result: result array length (" + result.length +
                            ") does not match number of matrix rows (" + rows() + ")"
            );
        }

        // Perform multiplication
        for (int i = 0; i < rows(); i++) {
            double sum = 0;
            for (int j = 0; j < cols(); j++) {
                sum += data[i][j] * x[j];
            }
            result[i] = sum;
        }
    }

    @Override
    public int rows() {
        return data.length;
    }

    @Override
    public int cols() {
        return data[0].length;
    }

    @Override
    public double get(int r, int c) {
        return data[r][c];
    }

    @Override
    public RealVector getValues(IntegerVector r, IntegerVector c) {
        if (r instanceof CpuIntegerVector rCpu && c instanceof CpuIntegerVector cCpu) {
            int[] rArr = rCpu.array();
            int[] cArr = cCpu.array();

            double[] values = new double[rCpu.size()];
            for (int i = 0; i < rCpu.size(); i++) {
                values[i] = get(rArr[i], cArr[i]);
            }

            return new CpuDoubleVector(values);

        } else {
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("DenseMatrix{\n");
        for (int i = 0; i < rows(); i++) {
            sb.append("  [");
            for (int j = 0; j < cols(); j++) {
                sb.append(data[i][j]);
                if (j < cols() - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        sb.append("}");
        return sb.toString();
    }
}