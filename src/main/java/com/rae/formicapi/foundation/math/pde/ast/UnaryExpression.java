package com.rae.formicapi.foundation.math.pde.ast;

import org.jetbrains.annotations.NotNull;

public record UnaryExpression(UnaryOperators operator, Expression child) implements Expression {

    @Override
    public @NotNull String toString() {
        return Expression.print(this);
    }
}