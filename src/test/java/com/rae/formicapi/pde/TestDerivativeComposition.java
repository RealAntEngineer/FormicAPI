package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.ast.Expression;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Rewrite rules that collapse a *composition* of two differential operators
 * into a single simpler operator: div(grad(x)) -> lap(x), ddx(ddx(x)) ->
 * d2dx2(x), mixed partials -> d2dxdy(x), etc. Distinct from plain linearity
 * (operators distributing over +/-), which lives in
 * {@link TestDifferentialOperatorLinearity}.
 */
public class TestDerivativeComposition {

    // ------------------------------------------------------------------
    // div(grad(...)) -> Laplacian
    // ------------------------------------------------------------------

    @Test
    void divOfGradCollapsesToLaplacianInASinglePass() {
        Expression expr = parse("div(grad(T))");
        // a single expand() suffices here: no wrapping multiplication forces a
        // second pass, unlike divOfCoefficientTimesGradCollapsesToCoefficientTimesLaplacian below.
        assertEquals("d2dx2(T) + d2dy2(T) + d2dz2(T)", expr.expand().prettyPrint());
    }

    @Test
    void divOfGradCollapsesAfterFirstDistributingASumInsideGrad() {
        // grad(a+b) distributes first, then div distributes over the sum,
        // then each div(grad(...)) term individually collapses to lap(...)
        Expression expr = parse("div(grad(a+b))");
        assertEquals("d2dx2(a) + d2dy2(a) + d2dz2(a) + d2dx2(b) + d2dy2(b) + d2dz2(b)", expr.expand().expand().prettyPrint());
    }

    @Test
    void divOfCoefficientTimesGradCollapsesToCoefficientTimesLaplacian() {
        // Equivalent to TestVectorTensorAlgebra#divOfKTimesGradExpandsToCoefficientTimesLaplacian,
        // but built via the string parser rather than a hand-built AST; kept here
        // as the canonical parser-facing regression test for this rewrite.
        Expression expr = parse("div(k*grad(T))");
        assertEquals("k * d2dx2(T) + k * d2dy2(T) + k * d2dz2(T)", expr.expand().expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Same-axis second derivatives
    // ------------------------------------------------------------------

    @Test
    void sameAxisSecondDerivativeCollapsesOnX() {
        Expression expr = parse("ddx(ddx(T))");
        assertEquals("d2dx2(T)", expr.expand().prettyPrint());
    }

    @Test
    void sameAxisSecondDerivativeCollapsesOnY() {
        Expression expr = parse("ddy(ddy(T))");
        assertEquals("d2dy2(T)", expr.expand().prettyPrint());
    }

    @Test
    void sameAxisSecondDerivativeCollapsesOnZ() {
        Expression expr = parse("ddz(ddz(T))");
        assertEquals("d2dz2(T)", expr.expand().prettyPrint());
    }

    @Test
    void axisSecondDerivativeCollapsesAfterLinearityDistributesFirst() {
        // ddx(ddx(a) + ddx(b)) shape reached via linearity first, then each collapses.
        // distribute() alone does not run combineLikeTerms, so both collapse
        // individually but stay summed rather than merging into 2 * d2dx2(T).
        Expression expr = parse("ddx(ddx(T)+ddx(T))");
        assertEquals("d2dx2(T) + d2dx2(T)", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Mixed partials
    // ------------------------------------------------------------------

    @Test
    void mixedPartialXyCollapses() {
        Expression expr = parse("ddx(ddy(T))");
        assertEquals("d2dxdy(T)", expr.expand().prettyPrint());
    }

    @Test
    void mixedPartialCollapsesRegardlessOfOperatorOrder() {
        // ddy(ddx(T)) should collapse the same way as ddx(ddy(T)) — mixed partials commute
        Expression expr = parse("ddy(ddx(T))");
        assertEquals("d2dxdy(T)", expr.expand().prettyPrint());
    }

    @Test
    void mixedPartialXzCollapses() {
        Expression expr = parse("ddx(ddz(T))");
        assertEquals("d2dxdz(T)", expr.expand().prettyPrint());
    }

    @Test
    void mixedPartialYzCollapses() {
        Expression expr = parse("ddz(ddy(T))");
        assertEquals("d2dydz(T)", expr.expand().prettyPrint());
    }

    // ------------------------------------------------------------------
    // Operators that must NOT collapse
    // ------------------------------------------------------------------

    @Test
    void ddtDoesNotCollapseWhenNestedWithAnAxisDerivative() {
        // ddt is not an axis derivative; nesting it with ddx should NOT collapse into anything
        Expression expr = parse("ddx(ddt(T))");
        assertEquals("ddx(ddt(T))", expr.expand().prettyPrint());
    }
}
