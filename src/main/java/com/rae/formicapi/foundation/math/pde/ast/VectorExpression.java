package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.List;
import java.util.stream.Collectors;

public record VectorExpression(List<Expression> components) implements Expression {

    public VectorExpression {
        components = List.copyOf(components);
        for (Expression c : components) {
            if (c.resultType() != FieldType.SCALAR) {
                throw new FieldType.TypeMismatchException("VectorExpression components must be SCALAR, got " + c.resultType());
            }
        }
    }

    public int dimension() {
        return components.size();
    }

    public Expression component(int axis) {
        return components.get(axis);
    }

    @Override
    public String toString() {
        return debugPrint();
    }

    @Override
    public FieldType resultType() {
        return FieldType.VECTOR;
    }

    @Override
    public String prettyPrint() {
        return components.stream()
                .map(Expression::prettyPrint)
                .collect(Collectors.joining(", ", "(", ")"));
    }

    @Override
    public String debugPrint() {
        return components.stream()
                .map(Expression::debugPrint)
                .collect(Collectors.joining(", ", "(", ")"));
    }
}