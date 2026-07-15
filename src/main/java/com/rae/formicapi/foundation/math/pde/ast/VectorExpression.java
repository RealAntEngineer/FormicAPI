package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class VectorExpression extends Expression {
    private final List<Expression> components;

    public VectorExpression(List<Expression> components) {
        super(components.getFirst().dimensions(), components.stream()
                .map(Expression::getDepth).reduce(Math::max)
                .orElse(0) + 1);//crash if components is < 1
        this.components = List.copyOf(components);
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
    public Expression expand() {

        List<Expression> expanded =
                components.stream()
                        .map(Expression::expand)
                        .toList();

        return new VectorExpression(expanded);
    }

    @Override
    public boolean isTimeDifferentiable() {
        return components.stream().anyMatch(Expression::isTimeDifferentiable);

    }

    @Override
    public boolean isSpaceDifferentiable() {
        return components.stream().anyMatch(Expression::isSpaceDifferentiable);
    }

    @Override
    public int appearanceOrder() {
        return 4;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(components);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof VectorExpression that)) return false;
        return Objects.equals(components, that.components);
    }

    @Override
    public String toString() {
        return "VectorExpression{" +
                "components=" + components +
                '}';
    }
}