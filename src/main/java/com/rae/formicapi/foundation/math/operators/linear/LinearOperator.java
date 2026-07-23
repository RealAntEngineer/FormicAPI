package com.rae.formicapi.foundation.math.operators.linear;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;

/**
 * Represents an object that applies a linear transformation:
 *
 * y = A*x
 */
public interface LinearOperator {

    int rows();

    int cols();

    void apply(Vector x, Vector result);
}