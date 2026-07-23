package com.rae.formicapi.foundation.math.operators.linear;

public interface MutableMatrix extends Matrix {

    void add(int row, int col, double value);

    void set(int row, int col, double value);

}