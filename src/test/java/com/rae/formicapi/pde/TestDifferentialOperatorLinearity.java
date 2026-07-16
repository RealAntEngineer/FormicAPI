package com.rae.formicapi.pde;


import com.rae.formicapi.foundation.math.pde.ast.Expression;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Linearity of grad/div/lap/ddt: distributing over sums and differences,
 * the product rule for grad, and constant folding. Tests where two
 * differential operators are *composed* (e.g. div(grad(...)) collapsing to
 * a Laplacian, or ddx(ddx(...)) collapsing to a second derivative) live in
 * {@link TestDerivativeComposition} instead — that's a distinct rewrite
 * concern from plain linearity.
 */
public class TestDifferentialOperatorLinearity {

    @Test
    void gradDistributesOverAddition() {
        Expression expr = parse("grad(a+b)");
        assertEquals("(ddx(a) + ddx(b), ddy(a) + ddy(b), ddz(a) + ddz(b))", expr.expand().prettyPrint());
    }

    @Test
    void divDistributesOverSubtraction() {
        Expression expr = parse("div(e1-e2)");
        Expression simplified = expr.expand();
        assertEquals("(ddx(e1[0]) + ddy(e1[1]) + ddz(e1[2])) - (ddx(e2[0]) + ddy(e2[1]) + ddz(e2[2]))", simplified.prettyPrint());
    }

    @Test
    void laplacianDistributesOverAddition() {
        Expression expr = parse("lap(a+b)");
        assertEquals("d2dx2(a) + d2dy2(a) + d2dz2(a) + d2dx2(b) + d2dy2(b) + d2dz2(b)", expr.expand().prettyPrint());
    }

    @Test
    void gradientProductRuleAppliesWhenBothFactorsAreNonConstant() {
        Expression expr = parse("grad(x*y)");
        assertEquals("(y * ddx(x) + x * ddx(y), y * ddy(x) + x * ddy(y), y * ddz(x) + x * ddz(y))", expr.expand().prettyPrint());
    }

    @Test
    void gradientOfConstantTimesFieldPullsConstantOutOfDerivative() {
        Expression expr = parse("grad(2*x)");
        assertEquals("(2.0 * ddx(x), 2.0 * ddy(x), 2.0 * ddz(x))", expr.expand().prettyPrint());
    }

    @Test
    void gradientOfPureConstantIsZero() {
        Expression expr = parse("grad(5)");
        assertEquals("0.0", expr.expand().prettyPrint());
    }

    @Test
    void ddtDistributesOverAddition() {
        Expression expr = parse("ddt(a+b)");
        assertEquals("ddt(a) + ddt(b)", expr.expand().prettyPrint());
    }
}
