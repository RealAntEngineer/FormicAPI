package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.List;
import java.util.stream.Collectors;

/**
 * rows.get(i).get(j) is the (i,j) entry; assumed square (rowCount == columnCount) everywhere this is used.
 */
public record TensorExpression(List<List<Expression>> rows) implements Expression {

    public TensorExpression {
        rows = rows.stream().map(List::copyOf).toList();
        for (List<Expression> row : rows) {
            for (Expression c : row) {
                if (c.resultType() != FieldType.SCALAR) {
                    throw new FieldType.TypeMismatchException("TensorExpression components must be SCALAR, got " + c.resultType());
                }
            }
        }
    }

    public int rowCount() {
        return rows.size();
    }

    public int columnCount() {
        return rows.isEmpty() ? 0 : rows.getFirst().size();
    }

    public Expression component(int row, int col) {
        return rows.get(row).get(col);
    }

    @Override
    public String toString() {
        return debugPrint();
    }

    @Override
    public FieldType resultType() {
        return FieldType.TENSOR;
    }

    @Override
    public String prettyPrint() {
        return rows.stream()
                .map(row -> row.stream()
                        .map(Expression::prettyPrint)
                        .collect(Collectors.joining(", ", "[", "]")))
                .collect(Collectors.joining(", ", "[", "]"));
    }

    @Override
    public String debugPrint() {
        return rows.stream()
                .map(row -> row.stream()
                        .map(Expression::debugPrint)
                        .collect(Collectors.joining(", ", "[", "]")))
                .collect(Collectors.joining(", ", "[", "]"));
    }
}