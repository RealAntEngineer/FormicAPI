package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

public record BinaryExpression(BinaryOperators operator, Expression left, Expression right) implements Expression {

    public BinaryExpression {
        operator.resultType(left.resultType(), right.resultType());
    }

    @Override
    public String toString() {
        return debugPrint();
    }

    @Override
    public FieldType resultType() {
        return operator.resultType(left.resultType(), right.resultType());
    }

    @Override
    public String prettyPrint() {
        String leftString = left instanceof BinaryExpression b
                && b.operator.priority < operator.priority
                ? "(" + left + ")"
                : left.prettyPrint();

        String rightString = right instanceof BinaryExpression b
                && b.operator.priority <= operator.priority
                ? "(" + right + ")"
                : right.prettyPrint();

        return leftString
                + " "
                + operator.representation
                + " "
                + rightString;
    }

    @Override
    public String debugPrint() {
        return "(" + left.debugPrint() + " " + operator.representation + " " + right.debugPrint() + ")";
    }
}