package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Objects;

public class ConstantExpression extends Expression {

    private final double value;

    public ConstantExpression(double value, int dimensions) {
        super(dimensions);
        this.value = value;
    }

    public double getValue() {
        return value;
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
    public boolean isTimeDifferentiable() {
        return false;
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return false;
    }

    @Override
    public boolean appearAfter(Expression expression) {
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(value);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ConstantExpression that)) return false;
        return Double.compare(value, that.value) == 0;
    }

    @Override
    public String toString() {
        return "ConstantExpression{" +
                "value=" + value +
                '}';
    }
}