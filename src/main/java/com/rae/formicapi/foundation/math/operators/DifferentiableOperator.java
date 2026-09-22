package com.rae.formicapi.foundation.math.operators;

import com.rae.formicapi.foundation.math.operators.vectors.RealVector;

public interface DifferentiableOperator extends GeneralOperator {

    void multiplyJacobian(RealVector x, RealVector direction, RealVector result);

}
