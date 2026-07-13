package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.ast.ConstantExpression;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import com.rae.formicapi.foundation.math.pde.ast.ExpressionAlgebra;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEquationPostTreatment1 {

    @Test
    void mulDistributionLeft() {
        Expression expr = parse("(a+b)*c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * c) + (b * c))", distributed.toString());
    }

    @Test
    void mulDistributionRight() {
        Expression expr = parse("a*(b+c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * b) + (a * c))", distributed.toString());
    }

    @Test
    void recursiveDistribution() {
        Expression expr = parse("(a+b)*(c+d)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * c) + (a * d)) + ((b * c) + (b * d)))", distributed.toString());
    }

    @Test
    void noDistribution() {
        Expression expr = parse("a*b");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(a * b)", distributed.toString());
    }

    @Test
    void divDistributionLeft() {
        Expression expr = parse("(a+b)/c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a / c) + (b / c))", distributed.toString());
    }

// ------------------------------------------------------------------
// Subtraction
// ------------------------------------------------------------------

    @Test
    void subMulDistributionLeft() {
        Expression expr = parse("(a-b)*c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * c) - (b * c))", distributed.toString());
    }

    @Test
    void subMulDistributionRight() {
        Expression expr = parse("a*(b-c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a * b) - (a * c))", distributed.toString());
    }

    @Test
    void subDivDistributionLeft() {
        Expression expr = parse("(a-b)/c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a / c) - (b / c))", distributed.toString());
    }

    @Test
    void mixedSignRecursiveDistribution() {
        Expression expr = parse("(a-b)*(c-d)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * c) - (a * d)) - ((b * c) - (b * d)))", distributed.toString());
    }

    // ------------------------------------------------------------------
    // Division must NOT distribute when the sum is the denominator
    // ------------------------------------------------------------------

    @Test
    void divNoDistributionWhenSumIsDenominator() {
        Expression expr = parse("a/(b+c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(a / (b + c))", distributed.toString());
    }

    @Test
    void divNoDistributionWhenDifferenceIsDenominator() {
        Expression expr = parse("a/(b-c)");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(a / (b - c))", distributed.toString());
    }

    // ------------------------------------------------------------------
    // POWER never distributes (no rule defined for it)
    // ------------------------------------------------------------------

    @Test
    void powerNeverDistributes() {
        Expression expr = parse("(a+b)^c");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("((a + b) ^ c)",distributed.toString());
    }

    // ------------------------------------------------------------------
    // Chained multiplication requires the recursive distribute(...) call
    // inside the rewrite branch, not just a single pass
    // ------------------------------------------------------------------

    @Test
    void chainedMultiplicationDistribution() {
        Expression expr = parse("(a+b)*c*d");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * c) * d) + ((b * c) * d))", distributed.toString());
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
        assertEquals("(((a * b) + (a * c)) + d)", distributed.toString());
    }

    // ------------------------------------------------------------------
    // Recursion into unary operators
    // ------------------------------------------------------------------

    @Test
    void distributionInsideUnaryOperator() {
        Expression expr = parse("(a*(b+c))^2");
        Expression distributed = ExpressionAlgebra.distribute(expr);
        assertEquals("(((a * b) + (a * c)) ^ 2.0)", distributed.toString());
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
        assertEquals(once.toString(), twice.toString());
    }
}