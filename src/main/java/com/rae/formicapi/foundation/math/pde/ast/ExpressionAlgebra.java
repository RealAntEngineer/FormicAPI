package com.rae.formicapi.foundation.math.pde.ast;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
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
            return new UnaryExpression(operator, distribute(operand));
        }

        return expression; // ConstantExpression, VariableExpression, DiscretizedVariableExpression
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
     * @param expression expression to flatten
     * @param coefficient accumulated sign/coefficient from parent subtraction
     * @param out destination list of terms
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
        return factors.stream().map(Expression::print).sorted().collect(Collectors.joining("*"));
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
     * @param expression expression containing additive terms
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
                String candidateKey = Expression.print(candidate);

                boolean sharedByAll = terms.stream()
                        .allMatch(t -> t.factors().stream()
                                .anyMatch(f -> Expression.print(f).equals(candidateKey)));

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
            if (Expression.print(it.next()).equals(key)) {
                it.remove();
                return;
            }
        }
    }

    private record Term(double coefficient, List<Expression> factors) {
    }
}