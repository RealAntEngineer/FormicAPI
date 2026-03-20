package com.rae.formicapi.fondation.math.operators;

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