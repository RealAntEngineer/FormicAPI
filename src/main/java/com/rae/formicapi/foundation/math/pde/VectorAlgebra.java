package com.rae.formicapi.foundation.math.pde;

import com.rae.formicapi.foundation.math.pde.ast.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Lowers vector/tensor-valued operators (grad, div, lap over VECTOR/TENSOR
 * operands, dot/cross/outer product, and scalar-vector/scalar-tensor
 * arithmetic) into explicit per-component scalar expressions.
 *
 * <p>This is a separate pass from {@link ScalarAlgebra#distribute}, run
 * after it: scalar distribution first simplifies the tree (linearity,
 * product rule, axis-derivative composition), then this pass expands
 * whatever remains that is vector/tensor-typed into a {@link VectorExpression}
 * or {@link MatrixExpression} of scalar components. Requires the spatial
 * dimensionality up front, since that isn't recoverable from a SCALAR/VECTOR/
 * TENSOR {@link FieldType} alone.
 */
@Deprecated
public final class VectorAlgebra {

    private static final List<UnaryOperators> AXIS_FIRST_DERIVATIVE  = List.of(UnaryOperators.DDX, UnaryOperators.DDY, UnaryOperators.DDZ);
    private static final List<UnaryOperators> AXIS_SECOND_DERIVATIVE = List.of(UnaryOperators.D2DX2, UnaryOperators.D2DY2, UnaryOperators.D2DZ2);

    private VectorAlgebra() {}

    /**
     * Runs the scalar {@link ScalarAlgebra#distribute} pass, then lowers
     * any remaining vector/tensor operators into per-component scalar expressions.
     *
     * @param expression expression to distribute and lower
     * @param dimensions number of spatial axes (1, 2, or 3)
     * @return an equivalent expression with all vector/tensor operators expanded
     */
    public static Expression distribute(Expression expression, int dimensions) {
        Expression scalarDistributed = ScalarAlgebra.distribute(expression);
        return lowerVectorAlgebra(scalarDistributed, dimensions);
    }

    private static Expression lowerVectorAlgebra(Expression expression, int dimensions) {

        if (expression instanceof UnaryExpression(UnaryOperators op, Expression operand)) {

            Expression lowered     = lowerVectorAlgebra(operand, dimensions);
            FieldType  operandType = lowered.resultType();

            if (op == UnaryOperators.GRAD && operandType == FieldType.SCALAR) {
                return gradOfScalar(lowered, dimensions);
            }
            if (op == UnaryOperators.GRAD && operandType == FieldType.VECTOR) {
                return gradOfVector(lowered, dimensions);
            }
            if (op == UnaryOperators.DIV && operandType == FieldType.VECTOR) {
                return divOfVector(lowered, dimensions);
            }
            if (op == UnaryOperators.DIV && operandType == FieldType.MATRIX) {
                return divOfTensor(lowered, dimensions);
            }
            if (op == UnaryOperators.LAPLACIAN && operandType == FieldType.SCALAR) {
                return laplacianOfScalar(lowered, dimensions);
            }
            if (op == UnaryOperators.LAPLACIAN && operandType == FieldType.VECTOR) {
                List<Expression> components = new ArrayList<>();
                for (int i = 0; i < dimensions; i++) {
                    components.add(laplacianOfScalar(componentOf(lowered, i), dimensions));
                }
                return new VectorExpression(components);
            }

            return new UnaryExpression(op, lowered);
        }

        if (expression instanceof BinaryExpression(BinaryOperators op, Expression left, Expression right)) {

            Expression l = lowerVectorAlgebra(left, dimensions);
            Expression r = lowerVectorAlgebra(right, dimensions);

            if (op == BinaryOperators.DOT_PRODUCT) {
                Expression sum = null;
                for (int i = 0; i < dimensions; i++) {
                    Expression product = mul(componentOf(l, i), componentOf(r, i));
                    sum = (sum == null) ? product : new BinaryExpression(BinaryOperators.ADD, sum, product);
                }
                return sum;
            }

            if (op == BinaryOperators.CROSS_PRODUCT) {
                if (dimensions != 3) {
                    throw new UnsupportedOperationException("Cross product is only defined for 3 dimensions, got " + dimensions);
                }
                return new VectorExpression(List.of(
                        sub(mul(componentOf(l, 1), componentOf(r, 2)), mul(componentOf(l, 2), componentOf(r, 1))),
                        sub(mul(componentOf(l, 2), componentOf(r, 0)), mul(componentOf(l, 0), componentOf(r, 2))),
                        sub(mul(componentOf(l, 0), componentOf(r, 1)), mul(componentOf(l, 1), componentOf(r, 0)))
                ));
            }

            if (op == BinaryOperators.OUTER_PRODUCT) {
                List<List<Expression>> rows = new ArrayList<>();
                for (int i = 0; i < dimensions; i++) {
                    List<Expression> row = new ArrayList<>();
                    for (int j = 0; j < dimensions; j++) {
                        row.add(mul(componentOf(l, i), componentOf(r, j)));
                    }
                    rows.add(row);
                }
                return new MatrixExpression(rows);
            }

            // Scalar * Vector/Tensor (either side) and Vector/Tensor / Scalar:
            // distribute into an explicit componentwise result rather than leaving
            // an opaque compound node that componentOf() can't see through later.
            if ((op == BinaryOperators.MULTIPLY || op == BinaryOperators.DIVIDE)
                    && (l.resultType() != FieldType.SCALAR || r.resultType() != FieldType.SCALAR)) {
                return lowerScalarVectorArithmetic(op, l, r, dimensions);
            }

            // Vector + Vector / Vector - Vector, Tensor +/- Tensor: componentwise.
            if ((op == BinaryOperators.ADD || op == BinaryOperators.SUBTRACT)
                    && l.resultType() == FieldType.VECTOR) {
                List<Expression> components = new ArrayList<>();
                for (int i = 0; i < dimensions; i++) {
                    components.add(new BinaryExpression(op, componentOf(l, i), componentOf(r, i)));
                }
                return new VectorExpression(components);
            }
            if ((op == BinaryOperators.ADD || op == BinaryOperators.SUBTRACT)
                    && l.resultType() == FieldType.MATRIX) {
                List<List<Expression>> rows = new ArrayList<>();
                for (int i = 0; i < dimensions; i++) {
                    List<Expression> row = new ArrayList<>();
                    for (int j = 0; j < dimensions; j++) {
                        row.add(new BinaryExpression(op, componentOfTensor(l, i, j), componentOfTensor(r, i, j)));
                    }
                    rows.add(row);
                }
                return new MatrixExpression(rows);
            }

            return new BinaryExpression(op, l, r);
        }

        return expression;
    }

    private static Expression lowerScalarVectorArithmetic(BinaryOperators op, Expression l, Expression r, int dimensions) {

        boolean    leftIsScalar = l.resultType() == FieldType.SCALAR;
        Expression scalar       = leftIsScalar ? l : r;
        Expression vectorLike   = leftIsScalar ? r : l;
        FieldType  vectorType   = vectorLike.resultType();

        if (vectorType == FieldType.VECTOR) {
            List<Expression> components = new ArrayList<>();
            for (int i = 0; i < dimensions; i++) {
                Expression component = componentOf(vectorLike, i);
                components.add(leftIsScalar
                        ? new BinaryExpression(op, scalar, component)
                        : new BinaryExpression(op, component, scalar));
            }
            return new VectorExpression(components);
        }

        // TENSOR
        List<List<Expression>> rows = new ArrayList<>();
        for (int i = 0; i < dimensions; i++) {
            List<Expression> row = new ArrayList<>();
            for (int j = 0; j < dimensions; j++) {
                Expression component = componentOfTensor(vectorLike, i, j);
                row.add(leftIsScalar
                        ? new BinaryExpression(op, scalar, component)
                        : new BinaryExpression(op, component, scalar));
            }
            rows.add(row);
        }
        return new MatrixExpression(rows);
    }

    private static Expression gradOfScalar(Expression scalar, int dimensions) {
        List<Expression> components = new ArrayList<>();
        for (int axis = 0; axis < dimensions; axis++) {
            components.add(new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(axis), scalar));
        }
        return new VectorExpression(components);
    }

    // Convention: grad(V)_ij = d(V_j)/d(axis_i) — row i is the i-th axis derivative of every component
    private static Expression gradOfVector(Expression vector, int dimensions) {
        List<List<Expression>> rows = new ArrayList<>();
        for (int axis = 0; axis < dimensions; axis++) {
            List<Expression> row = new ArrayList<>();
            for (int component = 0; component < dimensions; component++) {
                row.add(new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(axis), componentOf(vector, component)));
            }
            rows.add(row);
        }
        return new MatrixExpression(rows);
    }

    private static Expression divOfVector(Expression vector, int dimensions) {
        Expression sum = null;
        for (int axis = 0; axis < dimensions; axis++) {
            Expression term = new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(axis), componentOf(vector, axis));
            sum = (sum == null) ? term : new BinaryExpression(BinaryOperators.ADD, sum, term);
        }
        return sum;
    }

    // Convention: div(T)_i = sum_j d(T_ij)/d(axis_j) — contracts the second (column) index
    private static Expression divOfTensor(Expression tensor, int dimensions) {
        List<Expression> components = new ArrayList<>();
        for (int row = 0; row < dimensions; row++) {
            Expression sum = null;
            for (int col = 0; col < dimensions; col++) {
                Expression term = new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(col), componentOfTensor(tensor, row, col));
                sum = (sum == null) ? term : new BinaryExpression(BinaryOperators.ADD, sum, term);
            }
            components.add(sum);
        }
        return new VectorExpression(components);
    }

    private static Expression laplacianOfScalar(Expression scalar, int dimensions) {
        Expression sum = null;
        for (int axis = 0; axis < dimensions; axis++) {
            Expression term = new UnaryExpression(AXIS_SECOND_DERIVATIVE.get(axis), scalar);
            sum = (sum == null) ? term : new BinaryExpression(BinaryOperators.ADD, sum, term);
        }
        return sum;
    }

    private static Expression componentOf(Expression vectorLike, int axis) {
        FieldType type = vectorLike.resultType();
        if (type != FieldType.VECTOR) {
            throw new FieldType.TypeMismatchException("componentOf() requires a VECTOR expression, got " + type + " for: " + vectorLike);
        }
        if (vectorLike instanceof VectorExpression ve) {
            return ve.component(axis);
        }
        return ComponentExpression.ofVector(vectorLike, axis);
    }

    /** Extracts value at row `row` and column `col` of a TENSOR-valued expression as a SCALAR-valued expression. */
    private static Expression componentOfTensor(Expression tensorLike, int row, int col) {
        FieldType type = tensorLike.resultType();
        if (type != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException("componentOfTensor() requires a TENSOR expression, got " + type + " for: " + tensorLike);
        }
        if (tensorLike instanceof MatrixExpression te) {
            return te.component(row, col);
        }
        return ComponentExpression.ofMatrix(tensorLike, row, col);
    }

    /** Extracts row `row` of a TENSOR-valued expression as a VECTOR-valued expression. */
    private static Expression rowOf(Expression tensorLike, int row, int dimensions) {
        FieldType type = tensorLike.resultType();
        if (type != FieldType.MATRIX) {
            throw new FieldType.TypeMismatchException("rowOf() requires a TENSOR expression, got " + type + " for: " + tensorLike);
        }
        if (tensorLike instanceof MatrixExpression te) {
            List<Expression> components = new ArrayList<>();
            for (int col = 0; col < dimensions; col++) {
                components.add(te.component(row, col));
            }
            return new VectorExpression(components);
        }
        return ComponentExpression.rowOfMatrix(tensorLike, row);
    }

    private static Expression mul(Expression a, Expression b) {
        return new BinaryExpression(BinaryOperators.MULTIPLY, a, b);
    }

    private static Expression sub(Expression a, Expression b) {
        return new BinaryExpression(BinaryOperators.SUBTRACT, a, b);
    }
}