package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

/**
 * Provides symbolic algebra transformations on {@link Expression} trees.
 *
 * <p>This class contains utilities for simplifying and restructuring algebraic
 * expressions while preserving their mathematical meaning. Operations are
 * performed recursively on the expression tree and return new immutable
 * expression trees.
 *
 * <p>The supported transformations include:
 * <ul>
 *     <li>Distribution of multiplication and division over sums.</li>
 *     <li>Combination of identical additive terms.</li>
 *     <li>Extraction of common multiplicative factors.</li>
 * </ul>
 *
 * <p>Examples:
 * <pre>
 * distribute(a * (b + c))       = (a * b) + (a * c)
 * combineLikeTerms(a + a + b)   = (2 * a) + b
 * factorCommonFactor(a*c + a)   = a * (c + 1)
 * </pre>
 */
public abstract class ExpressionAlgebra {

    private static final List<UnaryOperators> AXIS_FIRST_DERIVATIVE  = List.of(UnaryOperators.DDX, UnaryOperators.DDY, UnaryOperators.DDZ);
    private static final List<UnaryOperators> AXIS_SECOND_DERIVATIVE = List.of(UnaryOperators.D2DX2, UnaryOperators.D2DY2, UnaryOperators.D2DZ2);

    /**
     * Fully distributes multiplicative operators over additive operators.
     *
     * <p>This transformation applies the distributive property recursively:
     *
     * <pre>
     * a * (b + c)     -> (a * b) + (a * c)
     * (a + b) * c     -> (a * c) + (b * c)
     * </pre>
     *
     * <p>Both multiplication and division are handled according to the
     * distributivity rules defined by {@link BinaryOperators#distributesOver}.
     *
     * @param expression expression to distribute
     * @return a new expression tree with all possible distributions applied
     */
    public static Expression distribute(Expression expression) {

        if (expression instanceof BinaryExpression(
                BinaryOperators outerOperator, Expression leftOperand, Expression rightOperand
        )) {

            Expression distributedLeft  = distribute(leftOperand);
            Expression distributedRight = distribute(rightOperand);

            // (a + b) * c -> (a * c) + (b * c)
            if (distributedLeft instanceof BinaryExpression(
                    BinaryOperators innerOperator, Expression innerLeft, Expression innerRight
            )
                    && outerOperator.distributesOver(innerOperator, true)) {

                return distribute(new BinaryExpression(
                        innerOperator,
                        new BinaryExpression(outerOperator, innerLeft, distributedRight),
                        new BinaryExpression(outerOperator, innerRight, distributedRight)));
            }

            // a * (b + c) -> (a * b) + (a * c)
            if (distributedRight instanceof BinaryExpression(
                    BinaryOperators innerOperator, Expression innerLeft, Expression innerRight
            )
                    && outerOperator.distributesOver(innerOperator, false)) {

                return distribute(new BinaryExpression(
                        innerOperator,
                        new BinaryExpression(outerOperator, distributedLeft, innerLeft),
                        new BinaryExpression(outerOperator, distributedLeft, innerRight)));
            }

            return new BinaryExpression(outerOperator, distributedLeft, distributedRight);
        }

        if (expression instanceof UnaryExpression(UnaryOperators operator, Expression operand)) {

            Expression distributedOperand = distribute(operand);

            // Derivative of a compile-time constant is zero, in space or time
            if (operator.isLinear() && distributedOperand instanceof ConstantExpression) {
                return new ConstantExpression(0.0);
            }

            // Linearity: op(a ± b) -> op(a) ± op(b)
            if (operator.isLinear()
                    && distributedOperand instanceof BinaryExpression(
                    BinaryOperators innerOp, Expression innerLeft, Expression innerRight
            )
                    && (innerOp == BinaryOperators.ADD || innerOp == BinaryOperators.SUBTRACT)) {

                return distribute(new BinaryExpression(innerOp,
                        new UnaryExpression(operator, innerLeft),
                        new UnaryExpression(operator, innerRight)));
            }

            // Composition identity: div(grad(x)) collapses into the second derivative, lap(x)
            if (operator == UnaryOperators.DIV
                    && distributedOperand instanceof UnaryExpression(
                    UnaryOperators innerOperator, Expression innerOperand
            )
                    && innerOperator == UnaryOperators.GRAD) {

                return new UnaryExpression(UnaryOperators.LAPLACIAN, innerOperand);
            }

            // Composition identity: nested single-axis derivatives collapse into the
            // corresponding second-order axis operator, e.g. ddx(ddx(x)) -> d2dx2(x),
            // ddx(ddy(x)) -> d2dxdy(x) (order-independent: ddy(ddx(x)) collapses the same way)
            if (isAxisDerivative(operator)
                    && distributedOperand instanceof UnaryExpression(
                    UnaryOperators innerOperator, Expression innerOperand
            )
                    && isAxisDerivative(innerOperator)) {

                Optional<UnaryOperators> composed = composeAxisDerivative(operator, innerOperator);
                if (composed.isPresent()) {
                    return new UnaryExpression(composed.get(), innerOperand);
                }
            }

            // Product rule: grad(a*b) -> (grad(a) * b) + (a * grad(b))
            if (operator == UnaryOperators.GRAD
                    && distributedOperand instanceof BinaryExpression(
                    BinaryOperators innerOp, Expression innerLeft, Expression innerRight
            )
                    && innerOp == BinaryOperators.MULTIPLY) {

                return distribute(gradientOfProduct(innerLeft, innerRight));
            }

            return new UnaryExpression(operator, distributedOperand);
        }

        return expression;
    }

    /**
     * distributes vector based operators
     */
    public static Expression distribute(Expression expression, int dimensions) {
        Expression scalarDistributed = distribute(expression); // existing zero-arg pass: linearity, product rule, axis/composition identities
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
            if (op == UnaryOperators.DIV && operandType == FieldType.TENSOR) {
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
                return new TensorExpression(rows);
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
                    && l.resultType() == FieldType.TENSOR) {
                List<List<Expression>> rows = new ArrayList<>();
                for (int i = 0; i < dimensions; i++) {
                    List<Expression> row = new ArrayList<>();
                    for (int j = 0; j < dimensions; j++) {
                        row.add(new BinaryExpression(op, componentOfTensor(l, i, j), componentOfTensor(r, i, j)));
                    }
                    rows.add(row);
                }
                return new TensorExpression(rows);
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
        return new TensorExpression(rows);
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
        return new TensorExpression(rows);
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

    private static @Nullable Expression laplacianOfScalar(Expression scalar, int dimensions) {
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
            throw new FieldType.TypeMismatchException(
                    "componentOf() requires a VECTOR expression, got " + type + " for: " + vectorLike);
        }
        if (vectorLike instanceof VectorExpression ve) {
            return ve.component(axis);
        }
        return ComponentExpression.ofVector(vectorLike, axis);
    }
    /** Extracts value at row `row` and column 'col' of a TENSOR-valued expression as a SCALAR-valued expression. */
    private static Expression componentOfTensor(Expression tensorLike, int row, int col) {
        FieldType type = tensorLike.resultType();
        if (type != FieldType.TENSOR) {
            throw new FieldType.TypeMismatchException(
                    "componentOfTensor() requires a TENSOR expression, got " + type + " for: " + tensorLike);
        }
        if (tensorLike instanceof TensorExpression te) {
            return te.component(row, col);
        }
        return ComponentExpression.ofTensor(tensorLike, row, col);
    }

    /** Extracts row `row` of a TENSOR-valued expression as a VECTOR-valued expression. */
    private static Expression rowOf(Expression tensorLike, int row, int dimensions) {
        FieldType type = tensorLike.resultType();
        if (type != FieldType.TENSOR) {
            throw new FieldType.TypeMismatchException(
                    "rowOf() requires a TENSOR expression, got " + type + " for: " +tensorLike);
        }
        if (tensorLike instanceof TensorExpression te) {
            List<Expression> components = new ArrayList<>();
            for (int col = 0; col < dimensions; col++) {
                components.add(te.component(row, col));
            }
            return new VectorExpression(components);
        }
        return ComponentExpression.rowOfTensor(tensorLike, row);
    }

    private static Expression mul(Expression a, Expression b) {
        return new BinaryExpression(BinaryOperators.MULTIPLY, a, b);
    }

    private static Expression sub(Expression a, Expression b) {
        return new BinaryExpression(BinaryOperators.SUBTRACT, a, b);
    }


    private static Expression gradientOfProduct(Expression left, Expression right) {

        boolean leftIsConstant  = left instanceof ConstantExpression;
        boolean rightIsConstant = right instanceof ConstantExpression;

        if (leftIsConstant && rightIsConstant) {
            return new ConstantExpression(0.0);
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

    /**
     * Flattens an additive expression into individual algebraic terms.
     *
     * <p>Addition and subtraction are converted into signed coefficients while
     * multiplication is kept inside each resulting {@link Term}.
     *
     * <pre>
     * a - b + c -> [a, -b, c]
     * </pre>
     *
     * @param expression  expression to flatten
     * @param coefficient accumulated sign/coefficient from parent subtraction
     * @param out         destination list of terms
     */
    private static void flattenTerms(Expression expression,
                                     double coefficient,
                                     List<Term> out) {

        if (expression instanceof BinaryExpression(BinaryOperators op, Expression left, Expression right)) {

            if (op == BinaryOperators.ADD) {
                flattenTerms(left, coefficient, out);
                flattenTerms(right, coefficient, out);
                return;
            }

            if (op == BinaryOperators.SUBTRACT) {
                flattenTerms(left, coefficient, out);
                flattenTerms(right, -coefficient, out);
                return;
            }
        }

        Term term = flattenFactors(expression);

        out.add(new Term(
                coefficient * term.coefficient(),
                term.factors()));
    }

    /**
     * Flattens a multiplication tree into a coefficient and a list of factors.
     *
     * <p>Examples:
     *
     * <pre>
     * 2*a*b -> coefficient = 2, factors = [a,b]
     * a*b*c -> coefficient = 1, factors = [a,b,c]
     * </pre>
     *
     * @param expression multiplicative expression
     * @return the extracted coefficient and factors
     */
    private static Term flattenFactors(Expression expression) {

        if (expression instanceof BinaryExpression(
                BinaryOperators op, Expression left, Expression right
        ) && op == BinaryOperators.MULTIPLY) {

            Term a = flattenFactors(left);
            Term b = flattenFactors(right);

            List<Expression> factors = new ArrayList<>(a.factors());
            factors.addAll(b.factors());

            return new Term(
                    a.coefficient() * b.coefficient(),
                    factors);
        }

        if (expression instanceof ConstantExpression(double value)) {
            return new Term(value, List.of());
        }

        return new Term(1.0, List.of(expression));
    }

    private static String factorSignature(List<Expression> factors) {
        return factors.stream().map(Expression::toString).sorted().collect(Collectors.joining("*"));
    }

    private static Expression termToExpression(Term term) {
        if (term.factors().isEmpty()) {
            return new ConstantExpression(term.coefficient());
        }
        Expression product = null;
        for (Expression factor : term.factors()) {
            product = (product == null) ? factor : new BinaryExpression(BinaryOperators.MULTIPLY, product, factor);
        }
        return term.coefficient() == 1.0 ? product : new BinaryExpression(BinaryOperators.MULTIPLY, new ConstantExpression(term.coefficient()), product);
    }

    private static Expression sumTerms(List<Term> terms) {
        Expression sum = null;
        for (Term term : terms) {
            Expression termExpr = termToExpression(term);
            sum = (sum == null) ? termExpr : new BinaryExpression(BinaryOperators.ADD, sum, termExpr);
        }
        return sum == null ? new ConstantExpression(0.0) : sum;
    }

    /**
     * Extracts and groups terms from an additive expression.
     *
     * <p>Terms with identical factor signatures are merged by summing their
     * coefficients. A recursive transformation can be applied to each factor
     * before grouping.
     *
     * @param expression         expression containing additive terms
     * @param recursiveTransform transformation applied recursively to factors
     * @return normalized list of non-zero terms
     */
    private static List<Term> collectTerms(
            Expression expression,
            UnaryOperator<Expression> recursiveTransform) {

        List<Term> terms = new ArrayList<>();
        flattenTerms(expression, 1.0, terms);

        LinkedHashMap<String, Term> grouped = new LinkedHashMap<>();

        for (Term term : terms) {
            List<Expression> transformed =
                    term.factors().stream()
                            .map(recursiveTransform)
                            .toList();

            Term normalized = new Term(term.coefficient(), transformed);

            grouped.merge(
                    factorSignature(normalized.factors()),
                    normalized,
                    (a, b) -> new Term(
                            a.coefficient() + b.coefficient(),
                            a.factors()));
        }

        return grouped.values()
                .stream()
                .filter(t -> t.coefficient() != 0.0)
                .toList();
    }

    /**
     * Combines additive terms having identical non-constant factors.
     *
     * <p>The expression is flattened into a list of terms, numerical coefficients
     * are accumulated, and equivalent terms are merged.
     *
     * <p>Examples:
     *
     * <pre>
     * a + a          -> 2 * a
     * a + 2*a + b    -> 3 * a + b
     * a*c + 2*a*c    -> 3 * (a*c)
     * </pre>
     *
     * <p>The transformation is recursively applied inside binary and unary
     * expressions.
     *
     * @param expression expression to simplify
     * @return equivalent expression with combined like terms
     */
    public static Expression combineLikeTerms(Expression expression) {

        if (expression instanceof BinaryExpression(BinaryOperators operator, Expression left, Expression right)
                && (operator == BinaryOperators.ADD || operator == BinaryOperators.SUBTRACT)) {

            List<Term> terms = collectTerms(expression, ExpressionAlgebra::combineLikeTerms);

            return sumTerms(terms);
        }

        if (expression instanceof BinaryExpression(BinaryOperators operator, Expression left, Expression right)) {
            return new BinaryExpression(operator, combineLikeTerms(left), combineLikeTerms(right));
        }

        if (expression instanceof UnaryExpression(UnaryOperators operator, Expression operand)) {
            return new UnaryExpression(operator, combineLikeTerms(operand));
        }

        return expression;
    }

    /**
     * Extracts common multiplicative factors from additive expressions.
     *
     * <p>This transformation applies the reverse of distribution:
     *
     * <pre>
     * a*b + a*c      -> a * (b + c)
     * a*c + a        -> a * (c + 1)
     * </pre>
     *
     * <p>Before extracting factors, identical terms are combined. A factor is
     * extracted only when it appears in every term of the sum.
     *
     * <p>The transformation is recursively applied inside binary and unary
     * expressions.
     *
     * @param expression expression to factor
     * @return equivalent expression with common factors extracted when possible
     */
    public static Expression factorCommonFactor(Expression expression) {

        if (expression instanceof BinaryExpression(BinaryOperators operator, Expression left, Expression right)
                && (operator == BinaryOperators.ADD || operator == BinaryOperators.SUBTRACT)) {

            List<Term> terms = collectTerms(expression, ExpressionAlgebra::factorCommonFactor);

            if (terms.size() < 2) {
                return sumTerms(terms);
            }

            for (Expression candidate : terms.getFirst().factors()) {
                String candidateKey = candidate.toString();

                boolean sharedByAll = terms.stream()
                        .allMatch(t -> t.factors().stream()
                                .anyMatch(f -> f.toString().equals(candidateKey)));

                if (sharedByAll) {
                    List<Term> remainders = terms.stream()
                            .map(t -> {
                                List<Expression> remainingFactors = new ArrayList<>(t.factors());
                                removeFirstMatching(remainingFactors, candidateKey);
                                return new Term(t.coefficient(), remainingFactors);
                            })
                            .toList();

                    Expression remainderSum = combineLikeTerms(sumTerms(remainders));

                    return new BinaryExpression(
                            BinaryOperators.MULTIPLY,
                            candidate,
                            remainderSum);
                }
            }

            return sumTerms(terms);
        }

        if (expression instanceof BinaryExpression(BinaryOperators operator, Expression left, Expression right)) {
            return new BinaryExpression(operator,
                    factorCommonFactor(left),
                    factorCommonFactor(right));
        }

        if (expression instanceof UnaryExpression(UnaryOperators operator, Expression operand)) {
            return new UnaryExpression(operator,
                    factorCommonFactor(operand));
        }

        return expression;
    }

    private static void removeFirstMatching(List<Expression> factors, String key) {
        Iterator<Expression> it = factors.iterator();
        while (it.hasNext()) {
            if (it.next().toString().equals(key)) {
                it.remove();
                return;
            }
        }
    }

    private record Term(double coefficient, List<Expression> factors) {
    }
}