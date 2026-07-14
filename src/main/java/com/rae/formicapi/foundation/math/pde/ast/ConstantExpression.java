package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

public class ConstantExpression extends Expression {

    private final double value;

    public ConstantExpression(double value, int dimensions){
        super(dimensions);
        this.value = value;
    }

    @Override
    public FieldType resultType() {
        return FieldType.SCALAR;
    }

    @Override
    public String prettyPrint() {
        return Double.toString(value);
    }

    @Override
    public Expression expand() {
        return this;
    }

    @Override
    public String debugPrint() {
        return Double.toString(value);
    }

    @Override
    public boolean isTimeDifferentiable() {
        return false;
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return false;
    }
}