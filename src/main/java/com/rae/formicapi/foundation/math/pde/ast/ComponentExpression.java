package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Extracts a scalar component from an opaque vector/tensor-valued expression (e.g. a symbol not yet expanded into a VectorExpression/TensorExpression).
 */
public class ComponentExpression extends Expression {

    private final Expression    source;
    private final List<Integer> indices;

    public ComponentExpression(Expression source, List<Integer> indices) {
        super(source.dimensions(), source.getDepth());
        this.source = source;
        this.indices = List.copyOf(indices);
        if (this.indices.isEmpty() || this.indices.size() > 2) {
            throw new IllegalArgumentException(
                    "ComponentExpression supports 1 index (vector entry, or tensor row) or 2 indices (matrix entry), got " + this.indices.size());
        }

        FieldType sourceType = source.resultType();

        if (this.indices.size() == 1 && sourceType != FieldType.VECTOR && sourceType != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException(
                    "Single-index component access requires a VECTOR or MATRIX source, got " + sourceType);
        }
        if (this.indices.size() == 2 && sourceType != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException(
                    "Two-index component access requires a MATRIX source, got " + sourceType);
        }
    }

    public static Expression ofVector(Expression source, int axis) {
        FieldType type = source.resultType();
        if (type != FieldType.VECTOR) {
            throw new FieldType.TypeMismatchException("componentOfVector() requires a VECTOR expression, got " + type + " for: " + source);
        }
        if (source instanceof VectorExpression ve) {
            return ve.component(axis);
        }

        return new ComponentExpression(source, List.of(axis));
    }

    public static Expression ofMatrix(Expression source, int row, int col) {
        FieldType type = source.resultType();
        if (type != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException("componentOfTensor() requires a MATRIX expression, got " + type + " for: " + source);
        }
        if (source instanceof MatrixExpression me) {
            return me.component(row, col);
        }
        return new ComponentExpression(source, List.of(row, col));
    }

    @Override
    public FieldType resultType() {
        if (indices.size() == 1) {
            return source.resultType() == FieldType.MATRIX ? FieldType.VECTOR : FieldType.SCALAR;
        }
        return FieldType.SCALAR;
    }

    @Override
    public String prettyPrint() {
        return source.prettyPrint() + "[" + indices.stream().map(String::valueOf).collect(Collectors.joining(",")) + "]";
    }

    @Override
    public Expression expand() {
        Expression expandedSource = source.expand();
        return indices.size() == 1 ? expandedSource.componentAt(indices.get(0)) : expandedSource.componentAt(indices.get(0), indices.get(1));
    }

    @Override
    public boolean isTimeDifferentiable() {
        return source.isTimeDifferentiable();
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return source.isSpaceDifferentiable();
    }

    @Override
    public boolean appearAfter(Expression expression) {

        if (expression instanceof ComponentExpression component) {

            // Compare the source first
            if (source.equals(component.source)) {
                return compareIndices(component.indices, this.indices) < 0;
            }

            // Different sources: delegate to source ordering
            return source.appearAfter(component.source);
        }

        if (expression instanceof VariableExpression || expression instanceof BinaryExpression) {
            return source.appearAfter(expression);
        }

        return super.appearAfter(expression);
    }

    @Override
    public int appearanceOrder() {
        return 1;
    }

    private static int compareIndices(List<Integer> a, List<Integer> b) {
        int n = Math.min(a.size(), b.size());

        for (int i = 0; i < n; i++) {
            int cmp = Integer.compare(a.get(i), b.get(i));
            if (cmp != 0) {
                return cmp;
            }
        }

        return Integer.compare(a.size(), b.size());
    }

    @Override
    public int hashCode() {
        return Objects.hash(source, indices);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ComponentExpression that)) return false;
        return Objects.equals(source, that.source) && Objects.equals(indices, that.indices);
    }

    @Override
    public String toString() {
        return "ComponentExpression{" +
                "source=" + source +
                ", indices=" + indices +
                '}';
    }
}