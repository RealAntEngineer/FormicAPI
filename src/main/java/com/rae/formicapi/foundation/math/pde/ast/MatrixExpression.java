package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * rows.get(i).get(j) is the (i,j) entry; assumed square (rowCount == columnCount) everywhere this is used.
 */
public class MatrixExpression extends Expression {
    private final List<List<Expression>> rows;


    public MatrixExpression(List<List<Expression>> rows) {
        super(rows.getFirst().getFirst().dimensions(),  rows.stream()
                .map(l -> l.stream().map(Expression::getDepth).reduce(Math::max)
                        .orElse(0)).reduce(Math::max).orElse(0) + 1);
        this.rows = rows;
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
    public FieldType resultType() {
        return FieldType.MATRIX;
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
    public Expression expand() {

        List<List<Expression>> expanded =
                rows.stream().map(
                        l -> l.stream()
                                .map(Expression::expand)
                                .toList()).toList();

        return new MatrixExpression(expanded);
    }

    @Override
    public boolean isTimeDifferentiable() {
        return rows.stream().flatMap(List::stream).anyMatch(Expression::isTimeDifferentiable);
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return rows.stream().flatMap(List::stream).anyMatch(Expression::isSpaceDifferentiable);
    }

    @Override
    public int appearanceOrder() {
        return 5;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(rows);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof MatrixExpression that)) return false;
        return Objects.equals(rows, that.rows);
    }

    @Override
    public String toString() {
        return "MatrixExpression{" +
                "rows=" + rows +
                '}';
    }
}