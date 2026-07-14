package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Extracts a scalar component from an opaque vector/tensor-valued expression (e.g. a symbol not yet expanded into a VectorExpression/TensorExpression).
 */
public class ComponentExpression extends Expression {

    private final Expression source;
    private final List<Integer> indices;

    public ComponentExpression(Expression source, List<Integer> indices){
        super(source.dimensions());
        this.source = source;
        this.indices = indices;
        indices = List.copyOf(indices);
        if (indices.isEmpty() || indices.size() > 2) {
            throw new IllegalArgumentException(
                    "ComponentExpression supports 1 index (vector entry, or tensor row) or 2 indices (matrix entry), got " + indices.size());
        }

        FieldType sourceType = source.resultType();

        if (indices.size() == 1 && sourceType != FieldType.VECTOR && sourceType != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException(
                    "Single-index component access requires a VECTOR or MATRIX source, got " + sourceType);
        }
        if (indices.size() == 2 && sourceType != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException(
                    "Two-index component access requires a MATRIX source, got " + sourceType);
        }
    }

    public static Expression ofVector(Expression source, int axis) {
        return new ComponentExpression(source, List.of(axis));
    }

    /** Extracts row `row` of a TENSOR source as a VECTOR. */
    public static Expression rowOfMatrix(Expression source, int row) {
        FieldType type = source.resultType();
        if (type != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException("rowOf() requires a MATRIX expression, got " + type + " for: " + source);
        }
        if (source instanceof MatrixExpression te) {
            List<Expression> components = new ArrayList<>();
            for (int col = 0; col < source.dimensions(); col++) {
                components.add(te.component(row, col));
            }
            return new VectorExpression(components);
        }
        return new ComponentExpression(source, List.of(row));
    }

    public static Expression ofMatrix(Expression source, int row, int col) {
        FieldType type = source.resultType();
        if (type != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException("componentOfTensor() requires a TENSOR expression, got " + type + " for: " + source);
        }
        if (source instanceof MatrixExpression te) {
            return te.component(row, col);
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
    public String debugPrint() {
        return source.debugPrint() + "[" + indices.stream().map(String::valueOf).collect(Collectors.joining(",")) + "]";
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
}