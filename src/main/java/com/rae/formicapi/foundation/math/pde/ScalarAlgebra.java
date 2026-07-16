package com.rae.formicapi.foundation.math.pde;


import com.rae.formicapi.foundation.math.pde.ast.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Provides scalar symbolic algebra transformations on {@link Expression} trees:
 * distribution, term combination, and common-factor extraction. Purely structural —
 * has no notion of spatial dimensionality. For lowering vector/tensor-valued
 * operators (grad, div, dot/cross/outer product) into per-component scalar
 * expressions.
 */
@Deprecated
public final class ScalarAlgebra {


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
    private static void flattenTerms(Expression expression, double coefficient, List<Term> out) {

        if (expression instanceof BinaryExpression be) {

            if (be.getOperator() == BinaryOperators.ADD) {
                flattenTerms(be.getLeft(), coefficient, out);
                flattenTerms(be.getRight(), coefficient, out);
                return;
            }

            if (be.getOperator() == BinaryOperators.SUBTRACT) {
                flattenTerms(be.getLeft(), coefficient, out);
                flattenTerms(be.getRight(), -coefficient, out);
                return;
            }
        }

        Term term = flattenFactors(expression);

        out.add(new Term(coefficient * term.coefficient(), term.factors()));
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

        if (expression instanceof BinaryExpression be && be.getOperator() == BinaryOperators.MULTIPLY) {

            Term a = flattenFactors(be.getLeft());
            Term b = flattenFactors(be.getRight());

            List<Expression> factors = new ArrayList<>(a.factors());
            factors.addAll(b.factors());

            return new Term(a.coefficient() * b.coefficient(), factors);
        }

        if (expression instanceof ConstantExpression ce) {
            return new Term(ce.getValue(), List.of());
        }

        return new Term(1.0, List.of(expression));
    }

    private static List<String> factorSignature(List<Expression> factors) {
        return factors.stream().map(Expression::toString).sorted().toList();
    }

    private static Expression termToExpression(Term term, int dimensions) {
        if (term.factors().isEmpty()) {
            return new ConstantExpression(term.coefficient(), dimensions);
        }
        Expression product = null;
        for (Expression factor : term.factors()) {
            product = (product == null) ? factor : new BinaryExpression(BinaryOperators.MULTIPLY, product, factor);
        }
        return term.coefficient() == 1.0 ? product : new BinaryExpression(BinaryOperators.MULTIPLY, new ConstantExpression(term.coefficient(), dimensions), product);
    }

    private static Expression sumTerms(List<Term> terms, int dimensions) {
        Expression sum = null;
        for (Term term : terms) {
            Expression termExpr = termToExpression(term, dimensions);
            sum = (sum == null) ? termExpr : new BinaryExpression(BinaryOperators.ADD, sum, termExpr);
        }
        return sum == null ? new ConstantExpression(0.0, dimensions) : sum;
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
    private static List<Term> collectTerms(Expression expression, UnaryOperator<Expression> recursiveTransform) {

        List<Term> terms = new ArrayList<>();
        flattenTerms(expression, 1.0, terms);

        LinkedHashMap<List<String>, Term> grouped = new LinkedHashMap<>();

        for (Term term : terms) {
            List<Expression> transformed = term.factors().stream().map(recursiveTransform).toList();
            Term             normalized  = new Term(term.coefficient(), transformed);

            grouped.merge(
                    factorSignature(normalized.factors()),
                    normalized,
                    (a, b) -> new Term(a.coefficient() + b.coefficient(), a.factors()));
        }

        return grouped.values().stream().filter(t -> t.coefficient() != 0.0).toList();
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

        switch (expression) {
            case BinaryExpression be when (be.getOperator() == BinaryOperators.ADD || be.getOperator() == BinaryOperators.SUBTRACT) -> {

                List<Term> terms = collectTerms(expression, ScalarAlgebra::combineLikeTerms);
                return sumTerms(terms, be.dimensions());
            }
            case BinaryExpression be -> {
                return new BinaryExpression(be.getOperator(), combineLikeTerms(be.getLeft()), combineLikeTerms(be.getRight()));
            }
            case UnaryExpression ue -> {
                return new UnaryExpression(ue.getOperator(), combineLikeTerms(ue.getOperand()));
            }
            case null, default -> {
            }
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

        switch (expression) {
            case BinaryExpression be when (be.getOperator() == BinaryOperators.ADD || be.getOperator() == BinaryOperators.SUBTRACT) -> {

                List<Term> terms = collectTerms(expression, ScalarAlgebra::factorCommonFactor);

                if (terms.size() < 2) {
                    return sumTerms(terms, be.dimensions());
                }

                for (Expression candidate : terms.getFirst().factors()) {
                    String candidateKey = candidate.toString();

                    boolean sharedByAll = terms.stream()
                            .allMatch(t -> t.factors().stream().anyMatch(f -> f.toString().equals(candidateKey)));

                    if (sharedByAll) {
                        List<Term> remainders = terms.stream()
                                .map(t -> {
                                    List<Expression> remainingFactors = new ArrayList<>(t.factors());
                                    removeFirstMatching(remainingFactors, candidateKey);
                                    return new Term(t.coefficient(), remainingFactors);
                                })
                                .toList();

                        Expression remainderSum = combineLikeTerms(sumTerms(remainders, be.dimensions()));

                        return new BinaryExpression(BinaryOperators.MULTIPLY, candidate, remainderSum);
                    }
                }

                return sumTerms(terms, be.dimensions());
            }
            case BinaryExpression be -> {
                return new BinaryExpression(be.getOperator(), factorCommonFactor(be.getLeft()), factorCommonFactor(be.getRight()));
            }
            case UnaryExpression ue -> {
                return new UnaryExpression(ue.getOperator(), factorCommonFactor(ue.getOperand()));
            }
            default -> {
            }
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
