package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

public record ConstantExpression(double value) implements Expression {

    @Override
    public String toString() {
        return debugPrint();
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
    public String debugPrint() {
        return Double.toString(value);
    }
}