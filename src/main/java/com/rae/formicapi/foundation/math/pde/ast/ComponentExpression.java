package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Extracts a scalar component from an opaque vector/tensor-valued expression (e.g. a symbol not yet expanded into a VectorExpression/TensorExpression).
 */
public record ComponentExpression(Expression source, List<Integer> indices) implements Expression {

    public ComponentExpression {
        indices = List.copyOf(indices);
        if (indices.isEmpty() || indices.size() > 2) {
            throw new IllegalArgumentException(
                    "ComponentExpression supports 1 index (vector entry, or tensor row) or 2 indices (tensor entry), got " + indices.size());
        }

        FieldType sourceType = source.resultType();

        if (indices.size() == 1 && sourceType != FieldType.VECTOR && sourceType != FieldType.TENSOR) {
            throw new FieldType.TypeMismatchException(
                    "Single-index component access requires a VECTOR or TENSOR source, got " + sourceType);
        }
        if (indices.size() == 2 && sourceType != FieldType.TENSOR) {
            throw new FieldType.TypeMismatchException(
                    "Two-index component access requires a TENSOR source, got " + sourceType);
        }
    }

    public static ComponentExpression ofVector(Expression source, int axis) {
        return new ComponentExpression(source, List.of(axis));
    }

    /** Extracts row `row` of a TENSOR source as a VECTOR. */
    public static ComponentExpression rowOfTensor(Expression source, int row) {
        return new ComponentExpression(source, List.of(row));
    }

    public static ComponentExpression ofTensor(Expression source, int row, int col) {
        return new ComponentExpression(source, List.of(row, col));
    }

    @Override
    public FieldType resultType() {
        if (indices.size() == 1) {
            return source.resultType() == FieldType.TENSOR ? FieldType.VECTOR : FieldType.SCALAR;
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
    public String toString() {
        return debugPrint();
    }
}