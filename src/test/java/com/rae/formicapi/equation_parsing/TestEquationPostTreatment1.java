package com.rae.formicapi.equation_parsing;

import com.rae.formicapi.foundation.math.pde.Field;
import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import com.rae.formicapi.foundation.math.pde.ast.ConstantExpression;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import com.rae.formicapi.foundation.math.pde.ast.ExpressionAlgebra;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEquationPostTreatment1 {

    private static SymbolBinding variable(String name) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), SymbolRole.COEFFICIENT);
    }

    private static final Map<String, SymbolBinding> SYMBOLS = Map.of(
            "a", variable("a"),
            "b", variable("b"),
            "c", variable("c"),
            "d", variable("d")
    );

    private static Expression parse(String expression) {
        return Expression.parseExpression(expression, SYMBOLS, 0);
    }

    @Test
    void mulDistributionLeft() {
        Expression expr = parse("(a+b)*c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * c) + (b * c))", Expression.print(distributed));
    }

    @Test
    void mulDistributionRight() {
        Expression expr = parse("a*(b+c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * b) + (a * c))", Expression.print(distributed));
    }

    @Test
    void recursiveDistribution() {
        Expression expr = parse("(a+b)*(c+d)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * c) + (a * d)) + ((b * c) + (b * d)))", Expression.print(distributed));
    }

    @Test
    void noDistribution() {
        Expression expr = parse("a*b");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(a * b)", Expression.print(distributed));
    }

    @Test
    void divDistributionLeft() {
        Expression expr = parse("(a+b)/c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a / c) + (b / c))", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // Subtraction
    // ------------------------------------------------------------------

    @Test
    void subMulDistributionLeft() {
        Expression expr = parse("(a-b)*c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * c) - (b * c))", Expression.print(distributed));
    }

    @Test
    void subMulDistributionRight() {
        Expression expr = parse("a*(b-c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * b) - (a * c))", Expression.print(distributed));
    }

    @Test
    void subDivDistributionLeft() {
        Expression expr = parse("(a-b)/c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a / c) - (b / c))", Expression.print(distributed));
    }

    @Test
    void mixedSignRecursiveDistribution() {
        Expression expr = parse("(a-b)*(c-d)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * c) - (a * d)) - ((b * c) - (b * d)))", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // Division must NOT distribute when the sum is the denominator
    // ------------------------------------------------------------------

    @Test
    void divNoDistributionWhenSumIsDenominator() {
        Expression expr = parse("a/(b+c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(a / (b + c))", Expression.print(distributed));
    }

    @Test
    void divNoDistributionWhenDifferenceIsDenominator() {
        Expression expr = parse("a/(b-c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(a / (b - c))", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // POWER never distributes (no rule defined for it)
    // ------------------------------------------------------------------

    @Test
    void powerNeverDistributes() {
        Expression expr = parse("(a+b)^c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a + b) ^ c)", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // Chained multiplication requires the recursive distribute(...) call
    // inside the rewrite branch, not just a single pass
    // ------------------------------------------------------------------

    @Test
    void chainedMultiplicationDistribution() {
        Expression expr = parse("(a+b)*c*d");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * c) * d) + ((b * c) * d))", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // Regression: reconstruction must happen even when the top-level
    // operator itself doesn't distribute (e.g. ADD), otherwise distributed
    // children get silently discarded on the fallthrough path
    // ------------------------------------------------------------------

    @Test
    void additionOfDistributableTermsIsReconstructed() {
        Expression expr = parse("a*(b+c) + d");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * b) + (a * c)) + d)", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // Recursion into unary operators
    // ------------------------------------------------------------------

    @Test
    void distributionInsideUnaryOperator() {
        Expression expr = parse("grad(a*(b+c))");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("grad(((a * b) + (a * c)))", Expression.print(distributed));
    }

    // ------------------------------------------------------------------
    // Leaves are untouched
    // ------------------------------------------------------------------

    @Test
    void constantIsUnchanged() {
        Expression expr = new ConstantExpression(5.0);
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals(expr, distributed);
    }

    @Test
    void variableIsUnchanged() {
        Expression expr = parse("a");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals(expr, distributed);
    }

    // ------------------------------------------------------------------
    // Idempotency: distributing an already-fully-distributed expression
    // is a no-op
    // ------------------------------------------------------------------

    @Test
    void distributionIsIdempotent() {
        Expression expr = parse("(a+b)*(c+d)");
        Expression once = ExpressionAlgebra.distribute(expr);
        Expression twice = ExpressionAlgebra.distribute(once);
        assertEquals(Expression.print(once), Expression.print(twice));
    }
}