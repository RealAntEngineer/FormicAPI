package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Objects;

public class BinaryExpression extends Expression {

    BinaryOperators operator;
    Expression      left;
    Expression      right;

    //TODO add automatic sorting on associative terms

    public BinaryExpression(
            BinaryOperators operator,
            Expression left,
            Expression right) {
        super(left.dimensions(), Math.max(left.getDepth(), right.getDepth()) + 1);

        if (left.dimensions() != right.dimensions()) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: "
                            + left.dimensions() + " != "
                            + right.dimensions()
            );
        }

        this.operator = operator;
        if (operator.commutative && (left.appearAfter(right) && !right.appearAfter(left))) {
            this.left = right;
            this.right = left;
        } else {
            this.left = left;
            this.right = right;
        }

        operator.resultType(left.resultType(), right.resultType());
    }

    public BinaryOperators getOperator() {
        return operator;
    }

    public Expression getLeft() {
        return left;
    }

    @Override
    public FieldType resultType() {
        return operator.resultType(left.resultType(), right.resultType());
    }

    @Override
    public String prettyPrint() {
        String leftString =
                left instanceof BinaryExpression b && needsParentheses(b, true)
                        ? "(" + left.prettyPrint() + ")"
                        : left.prettyPrint();

        String rightString =
                right instanceof BinaryExpression b && needsParentheses(b, false)
                        ? "(" + right.prettyPrint() + ")"
                        : right.prettyPrint();

        return leftString
                + " "
                + operator.representation
                + " "
                + rightString;
    }

    private boolean needsParentheses(BinaryExpression child, boolean isLeft) {

        if (child.operator.priority < operator.priority)
            return true;

        if (child.operator.priority > operator.priority)
            return false;

        return !operator.commutative;
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

    @Override
    public boolean appearAfter(Expression expression) {

        if (expression instanceof BinaryExpression be) {
            return this.operator.priority > be.operator.priority ||
                    this.operator.priority == be.operator.priority
                    && !be.right.appearAfter(this.getRight()) && be.getDepth() <= this.getDepth();
        }

        return this.left.appearAfter(expression);
    }

    public Expression getRight() {
        return right;
    }

    @Override
    public int appearanceOrder() {
        return 3;
    }

    @Override
    public int hashCode() {
        return Objects.hash(operator, left, right);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof BinaryExpression that)) return false;
        return operator == that.operator && Objects.equals(left, that.left) && Objects.equals(right, that.right);
    }

    @Override
    public String toString() {
        return "BinaryExpression{" +
                "operator=" + operator +
                ", left=" + left +
                ", right=" + right +
                '}';
    }
}