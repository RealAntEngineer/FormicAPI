package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

public record UnaryExpression(UnaryOperators operator, Expression child) implements Expression {
    public UnaryExpression {
        operator.resultType(child.resultType()); // validates eagerly; result discarded here
    }

    @Override
    public FieldType resultType() {
        return operator.resultType(child.resultType());
    }

    @Override
    public String prettyPrint() {
        return operator.representation() + "(" + child.prettyPrint() + ")";
    }

    @Override
    public String debugPrint() {
        return operator.representation() + "(" + child.debugPrint() + ")";
    }

    @Override
    public String toString() {
        return debugPrint();
    }
}