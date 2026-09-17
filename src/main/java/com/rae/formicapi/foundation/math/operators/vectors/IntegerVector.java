package com.rae.formicapi.foundation.math.operators.vectors;

public interface IntegerVector extends Vector {

    //int[] array();

    IntegerVector set(int value, int idx);

    int get(int idx);
}
