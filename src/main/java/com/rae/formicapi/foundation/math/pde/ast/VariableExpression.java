package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;

import java.util.Objects;

public class VariableExpression extends Expression {
    private final SymbolBinding symbol;

    public VariableExpression(SymbolBinding symbol, int dimensions) {
        super(dimensions);
        this.symbol = symbol;
    }

    public SymbolBinding getSymbol() {
        return symbol;
    }

    @Override
    public FieldType resultType() {
        return symbol.field().type();
    }

    @Override
    public String prettyPrint() {
        return symbol.field().name();
    }

    @Override
    public Expression expand() {
        return this;
    }

    @Override
    public boolean isTimeDifferentiable() {
        return symbol.role().isTimeDifferentiable();
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return symbol.role().isSpaceDifferentiable();
    }

    @Override
    public boolean appearAfter(Expression expression) {
        if (expression instanceof VariableExpression variable) {
            return variable.symbol.field().name().compareTo(this.symbol.field().name()) < 0;
        } else if (expression instanceof BinaryExpression be) {
            // variable appears after binary if it is after the largest child and there is no nesting
            return this.appearAfter(be.getRight()) && be.getDepth() <= this.getDepth();
        }
        return super.appearAfter(expression);
    }

    @Override
    public int appearanceOrder() {
        return 1;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(symbol);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof VariableExpression that)) return false;
        return Objects.equals(symbol, that.symbol);
    }

    @Override
    public String toString() {
        return "VariableExpression{" +
                "name=" + symbol +
                '}';
    }
}