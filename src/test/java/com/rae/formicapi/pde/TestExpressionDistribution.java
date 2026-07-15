package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.ast.ConstantExpression;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link Expression#expand()} distributing multiplication/division over
 * addition/subtraction for plain scalar arithmetic. Vector-calculus-specific
 * distribution (grad/div/lap over sums, product rule) lives in
 * {@link TestDifferentialOperatorLinearity} instead.
 */
public class TestExpressionDistribution {

    // ------------------------------------------------------------------
    // Multiplication / division over addition
    // ------------------------------------------------------------------

    @Test
    void multiplicationDistributesOverAdditionOnLeft() {
        Expression expr = parse("(a+b)*c");
        assertEquals("a * c + b * c", expr.expand().prettyPrint());
    }

    @Test
    void multiplicationDistributesOverAdditionOnRight() {
        Expression expr = parse("a*(b+c)");
        assertEquals("a * b + a * c", expr.expand().prettyPrint());
    }

    @Test
    void distributionRecursesIntoBothFactorsOfNestedSums() {
        Expression expr = parse("(a+b)*(c+d)");
        assertEquals("a * c + a * d + b * c + b * d", expr.expand().prettyPrint());
    }

    @Test
    void multiplicationOfTwoPlainVariablesIsUnchanged() {
        Expression expr = parse("a*b");
        assertEquals("a * b", expr.expand().prettyPrint());
    }

    @Test
    void divisionDistributesOverAdditionInNumerator() {
        Expression expr = parse("(a+b)/c");
        assertEquals("a / c + b / c", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Subtraction
    // ------------------------------------------------------------------

    @Test
    void multiplicationDistributesOverSubtractionOnLeft() {
        Expression expr = parse("(a-b)*c");
        assertEquals("a * c - b * c", expr.expand().prettyPrint());
    }

    @Test
    void multiplicationDistributesOverSubtractionOnRight() {
        Expression expr = parse("a*(b-c)");
        assertEquals("a * b - a * c", expr.expand().prettyPrint());
    }

    @Test
    void divisionDistributesOverSubtractionInNumerator() {
        Expression expr = parse("(a-b)/c");
        assertEquals("a / c - b / c", expr.expand().prettyPrint());
    }

    @Test
    void differenceOfSumsDistributesWithParenthesizedResult() {
        Expression expr = parse("(a-b)*(c-d)");
        assertEquals("(a * c - a * d) - (b * c - b * d)", expr.expand().prettyPrint());
        //assertEquals("a * c - a * d - b * c + b * d", expr.expand().prettyPrint()); -> in the future
    }

    // ------------------------------------------------------------------
    // Division must NOT distribute when the sum is the denominator
    // ------------------------------------------------------------------

    @Test
    void divisionDoesNotDistributeWhenSumIsDenominator() {
        Expression expr = parse("a/(b+c)");
        assertEquals("a / (b + c)", expr.expand().prettyPrint());
    }

    @Test
    void divisionDoesNotDistributeWhenDifferenceIsDenominator() {
        Expression expr = parse("a/(b-c)");
        assertEquals("a / (b - c)", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // POWER never distributes (no rule defined for it)
    // ------------------------------------------------------------------

    @Test
    void powerNeverDistributesOverItsBase() {
        Expression expr = parse("(a+b)^c");
        assertEquals("(a + b) ^ c", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Chained multiplication requires the recursive distribute(...) call
    // inside the rewrite branch, not just a single pass
    // ------------------------------------------------------------------

    @Test
    void chainedMultiplicationDistributesAcrossAllFactors() {
        Expression expr = parse("(a+b)*c*d");
        assertEquals("a * c * d + b * c * d", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Regression: reconstruction must happen even when the top-level
    // operator itself doesn't distribute (e.g. ADD), otherwise distributed
    // children get silently discarded on the fallthrough path
    // ------------------------------------------------------------------

    @Test
    void siblingAdditionIsPreservedWhenOnlyOneSideDistributes() {
        Expression expr = parse("a*(b+c) + d");
        assertEquals("a * b + a * c + d", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Recursion into unary operators
    // ------------------------------------------------------------------

    @Test
    void distributionRecursesInsideUnaryOperator() {
        Expression expr = parse("(a*(b+c))^2");
        assertEquals("(a * b + a * c) ^ 2.0", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Leaves are untouched
    // ------------------------------------------------------------------

    @Test
    void constantLeafIsUnchangedByExpand() {
        Expression expr = new ConstantExpression(5.0, 3);
        assertEquals(expr, expr.expand());
    }

    @Test
    void variableLeafIsUnchangedByExpand() {
        Expression expr = parse("a");
        assertEquals(expr, expr.expand());
    }

    // ------------------------------------------------------------------
    // Idempotency: distributing an already-fully-distributed expression
    // is a no-op
    // ------------------------------------------------------------------
    //TODO add more complex function to verify said property
    @Test
    void expandingAnAlreadyDistributedExpressionIsANoOp() {
        Expression expr = parse("(a+b)*(c+d)");
        Expression once = expr.expand();
        Expression twice = once.expand();
        assertEquals(once.toString(), twice.toString());
    }
}