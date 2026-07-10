package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import org.jetbrains.annotations.NotNull;

public record VariableExpression(SymbolBinding name) implements Expression {

    @Override
    public @NotNull String toString() {
        return Expression.print(this);
    }
}