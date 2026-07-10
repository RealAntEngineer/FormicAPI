package com.rae.formicapi.foundation.math.pde.ast;

import org.jetbrains.annotations.NotNull;

public record BinaryExpression(BinaryOperator operator, Expression left, Expression right) implements Expression {

    @Override
    public @NotNull String toString() {
        return Expression.print(this);
    }
}