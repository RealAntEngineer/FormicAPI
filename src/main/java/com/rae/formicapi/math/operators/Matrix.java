package com.rae.formicapi.math.operators;

public interface Matrix {

    int rows();

    int cols();

    double get(int row, int col);

    void multiply(double[] x, double[] result);
}
