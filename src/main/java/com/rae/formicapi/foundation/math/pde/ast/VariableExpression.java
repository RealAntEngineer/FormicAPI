package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;

public class VariableExpression extends Expression {

    private final SymbolBinding name;
    public VariableExpression(SymbolBinding name, int dimensions){
        super(dimensions);
        this.name = name;
    }

    @Override
    public String toString() {
        return debugPrint();
    }

    @Override
    public FieldType resultType() {
        return name.field().type();
    }

    @Override
    public String prettyPrint() {
        return name.field().name();
    }

    @Override
    public Expression expand() {
        return null;
    }

    @Override
    public String debugPrint() {
        return name.field().name();
    }

    // VariableExpression
    @Override public boolean isTimeDifferentiable() { return name.role().isTimeDifferentiable(); }
    @Override public boolean isSpaceDifferentiable() { return name.role().isSpaceDifferentiable(); }
}