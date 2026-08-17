package com.rae.formicapi.foundation.math.operators.linear;

import com.rae.formicapi.foundation.math.operators.GeneralOperator;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;

/**
 * Represents an object that applies a linear transformation:
 *
 * y = A*x
 */
public interface LinearOperator extends GeneralOperator {

    int rows();

    int cols();

    void apply(DoubleVector x, DoubleVector result);
}