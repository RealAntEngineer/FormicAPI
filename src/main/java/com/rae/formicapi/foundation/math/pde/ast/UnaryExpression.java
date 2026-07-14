package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class UnaryExpression extends Expression {
    final UnaryOperators operator;
    final Expression     operand;

    public UnaryOperators getOperator() {
        return operator;
    }

    public Expression getOperand() {
        return operand;
    }

    private static final List<UnaryOperators> AXIS_FIRST_DERIVATIVE  = List.of(UnaryOperators.DDX, UnaryOperators.DDY, UnaryOperators.DDZ);
    private static final List<UnaryOperators> AXIS_SECOND_DERIVATIVE = List.of(UnaryOperators.D2DX2, UnaryOperators.D2DY2, UnaryOperators.D2DZ2);

    public UnaryExpression(UnaryOperators operator, Expression operand) {
        super(operand.dimensions());
        this.operator = operator;
        this.operand = operand;
        operator.resultType(operand.resultType()); // validates eagerly; result discarded here
    }

    @Override
    public FieldType resultType() {
        return operator.resultType(operand.resultType());
    }

    @Override
    public String prettyPrint() {
        return operator.representation() + "(" + operand.prettyPrint() + ")";
    }

    @Override
    public Expression expand() {
        Expression expandedOperand = operand.expand();

        if ((operator.appliesTimeDifferentiation() || operator.appliesSpaceDifferentiation())
                && expandedOperand instanceof ConstantExpression) {
            return new ConstantExpression(0.0, dimensions());
        }

        if (operator.isLinear() && expandedOperand instanceof BinaryExpression inner
                && (inner.getOperator() == BinaryOperators.ADD || inner.getOperator() == BinaryOperators.SUBTRACT)) {
            return inner.getOperator().expand(
                    new UnaryExpression(operator, inner.getLeft()).expand(),
                    new UnaryExpression(operator, inner.getRight()).expand());
        }

        if (operator == UnaryOperators.DIV && expandedOperand instanceof UnaryExpression inner && inner.operator == UnaryOperators.GRAD) {
            return new UnaryExpression(UnaryOperators.LAPLACIAN, inner.getOperand()).expand();
        }

        if (isAxisDerivative(operator) && expandedOperand instanceof UnaryExpression inner && isAxisDerivative(inner.operator)) {
            Optional<UnaryOperators> composed = composeAxisDerivative(operator, inner.operator);
            if (composed.isPresent()) return new UnaryExpression(composed.get(), inner.operand).expand();
        }

        if (operator.isLinear() && expandedOperand instanceof BinaryExpression inner && inner.operator == BinaryOperators.MULTIPLY) {
            boolean leftApplies = appliesTo(operator, inner.getLeft());
            boolean rightApplies = appliesTo(operator, inner.getRight());

            if (!leftApplies && rightApplies) return new BinaryExpression(BinaryOperators.MULTIPLY, inner.getLeft(), new UnaryExpression(operator, inner.getRight()).expand());
            if (leftApplies && !rightApplies) return new BinaryExpression(BinaryOperators.MULTIPLY, inner.getRight(), new UnaryExpression(operator, inner.getLeft()).expand());
            if (leftApplies && operator == UnaryOperators.GRAD) return gradientOfProduct(inner.getLeft(), inner.getRight()).expand();
        }

        FieldType operandType = expandedOperand.resultType();
        int dimensions = expandedOperand.dimensions();

        if (operator == UnaryOperators.GRAD && operandType == FieldType.SCALAR) return gradOfScalar(expandedOperand, dimensions);
        if (operator == UnaryOperators.GRAD && operandType == FieldType.VECTOR) return gradOfVector(expandedOperand, dimensions);
        if (operator == UnaryOperators.DIV && operandType == FieldType.VECTOR) return divOfVector(expandedOperand, dimensions);
        if (operator == UnaryOperators.DIV && operandType == FieldType.MATRIX) return divOfTensor(expandedOperand, dimensions);
        if (operator == UnaryOperators.LAPLACIAN && operandType == FieldType.SCALAR) return laplacianOfScalar(expandedOperand, dimensions);
        if (operator == UnaryOperators.LAPLACIAN && operandType == FieldType.VECTOR) {
            List<Expression> components = new ArrayList<>();
            for (int i = 0; i < dimensions; i++) components.add(laplacianOfScalar(expandedOperand.componentAt(i), dimensions));
            return new VectorExpression(components);
        }

        return new UnaryExpression(operator, expandedOperand);
    }

    private static boolean appliesTo(UnaryOperators operator, Expression operand) {
        return (operator.appliesTimeDifferentiation() && operand.isTimeDifferentiable())
                || (operator.appliesSpaceDifferentiation() && operand.isSpaceDifferentiable());
    }

    private static Expression gradientOfProduct(Expression left, Expression right) {

        boolean leftIsConstant  = left instanceof ConstantExpression;
        boolean rightIsConstant = right instanceof ConstantExpression;

        if (leftIsConstant && rightIsConstant) {
            return new ConstantExpression(0.0, left.dimensions());
        }
        if (leftIsConstant) {
            return new BinaryExpression(BinaryOperators.MULTIPLY, left, new UnaryExpression(UnaryOperators.GRAD, right));
        }
        if (rightIsConstant) {
            return new BinaryExpression(BinaryOperators.MULTIPLY, right, new UnaryExpression(UnaryOperators.GRAD, left));
        }
        return new BinaryExpression(BinaryOperators.ADD,
                new BinaryExpression(BinaryOperators.MULTIPLY, new UnaryExpression(UnaryOperators.GRAD, left), right),
                new BinaryExpression(BinaryOperators.MULTIPLY, left, new UnaryExpression(UnaryOperators.GRAD, right)));
    }

    private static boolean isAxisDerivative(UnaryOperators operator) {
        return operator == UnaryOperators.DDX || operator == UnaryOperators.DDY || operator == UnaryOperators.DDZ;
    }

    private static Optional<UnaryOperators> composeAxisDerivative(UnaryOperators outer, UnaryOperators inner) {
        if (outer == UnaryOperators.DDX && inner == UnaryOperators.DDX) return Optional.of(UnaryOperators.D2DX2);
        if (outer == UnaryOperators.DDY && inner == UnaryOperators.DDY) return Optional.of(UnaryOperators.D2DY2);
        if (outer == UnaryOperators.DDZ && inner == UnaryOperators.DDZ) return Optional.of(UnaryOperators.D2DZ2);

        if (isPair(outer, inner, UnaryOperators.DDX, UnaryOperators.DDY)) return Optional.of(UnaryOperators.D2DXDY);
        if (isPair(outer, inner, UnaryOperators.DDX, UnaryOperators.DDZ)) return Optional.of(UnaryOperators.D2DXDZ);
        if (isPair(outer, inner, UnaryOperators.DDY, UnaryOperators.DDZ)) return Optional.of(UnaryOperators.D2DYDZ);

        return Optional.empty();
    }

    private static boolean isPair(UnaryOperators a, UnaryOperators b, UnaryOperators x, UnaryOperators y) {
        return (a == x && b == y) || (a == y && b == x);
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
                row.add(new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(axis), vector.componentAt(component)));
            }
            rows.add(row);
        }
        return new MatrixExpression(rows);
    }

    private static Expression divOfVector(Expression vector, int dimensions) {
        Expression sum = null;
        for (int axis = 0; axis < dimensions; axis++) {
            Expression term = new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(axis), vector.componentAt(axis));
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
                Expression term = new UnaryExpression(AXIS_FIRST_DERIVATIVE.get(col), ComponentExpression.ofMatrix(tensor, row, col));
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


    @Override
    public String debugPrint() {
        return operator.representation() + "(" + operand.debugPrint() + ")";
    }

    @Override
    public boolean isTimeDifferentiable() {
        return operand.isTimeDifferentiable();
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return operand.isSpaceDifferentiable();
    }

    @Override
    public String toString() {
        return debugPrint();
    }
}