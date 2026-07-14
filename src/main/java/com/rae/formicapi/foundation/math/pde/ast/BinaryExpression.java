package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

public class BinaryExpression extends Expression {

    BinaryOperators operator;
    Expression left;
    Expression right;

    public BinaryOperators getOperator() {
        return operator;
    }

    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }

    public BinaryExpression(
            BinaryOperators operator,
            Expression left,
            Expression right) {

        super(left.dimensions());

        if (left.dimensions() != right.dimensions()) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: "
                            + left.dimensions() + " != "
                            + right.dimensions()
            );
        }

        this.operator = operator;
        this.left = left;
        this.right = right;

        operator.resultType(left.resultType(), right.resultType());
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

    @Override
    public Expression expand() {
        return operator.expand(left.expand(), right.expand());
    }

    @Override
    public boolean isTimeDifferentiable() {
        return left.isTimeDifferentiable() || right.isTimeDifferentiable();
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return left.isSpaceDifferentiable() || right.isSpaceDifferentiable();
    }
}