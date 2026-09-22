package com.rae.formicapi.foundation.math.operators.nonlinear;

import com.rae.formicapi.foundation.math.operators.DifferentiableOperator;
import com.rae.formicapi.foundation.math.operators.GeneralOperator;

public interface PolynomialOperator extends DifferentiableOperator {

    int order();
}