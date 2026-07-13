package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;

public record VariableExpression(SymbolBinding name) implements Expression {

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
    public String debugPrint() {
        return name.field().name();
    }
}