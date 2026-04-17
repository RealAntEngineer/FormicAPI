package com.rae.formicapi.fondation.math.operators;

public interface MutableMatrix extends Matrix {

    void add(int row, int col, double value);

    void set(int row, int col, double value);

}
