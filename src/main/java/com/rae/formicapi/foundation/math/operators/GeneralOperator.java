package com.rae.formicapi.foundation.math.operators;

import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;

public interface GeneralOperator {

    /**
     * Applies this operator to {@code x}, storing the result in {@code result}.
     *
     * @param x      input vector
     * @param result output vector, overwritten by the result
     * @throws IllegalArgumentException if the vector dimensions are incompatible
     */
    void apply(DoubleVector x, DoubleVector result);
}
