package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.Field;
import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import com.rae.formicapi.foundation.math.pde.ast.ExpressionAlgebra;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEquationPostTreatment4 {

    @Test
    void sameAxisSecondDerivativeCollapses() {
        Expression expr = parse("ddx(ddx(T))");
        assertEquals("d2dx2(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void yAxisSecondDerivativeCollapses() {
        Expression expr = parse("ddy(ddy(T))");
        assertEquals("d2dy2(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void zAxisSecondDerivativeCollapses() {
        Expression expr = parse("ddz(ddz(T))");
        assertEquals("d2dz2(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void mixedPartialXyCollapses() {
        Expression expr = parse("ddx(ddy(T))");
        assertEquals("d2dxdy(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void mixedPartialCollapsesRegardlessOfOrder() {
        // ddy(ddx(T)) should collapse the same way as ddx(ddy(T)) — mixed partials commute
        Expression expr = parse("ddy(ddx(T))");
        assertEquals("d2dxdy(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void mixedPartialXzCollapses() {
        Expression expr = parse("ddx(ddz(T))");
        assertEquals("d2dxdz(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void mixedPartialYzCollapses() {
        Expression expr = parse("ddz(ddy(T))");
        assertEquals("d2dydz(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void divOfGradStillCollapsesToLaplacianNotAxisDerivative() {
        // sanity check the two composition rules don't interfere with each other
        Expression expr = parse("div(grad(T))");
        assertEquals("lap(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void ddtDoesNotCollapseWithAxisDerivative() {
        // ddt is not an axis derivative; nesting it with ddx should NOT collapse into anything
        Expression expr = parse("ddx(ddt(T))");
        assertEquals("ddx(ddt(T))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void axisSecondDerivativeCollapsesAfterLinearityDistribution() {
        // ddx(ddx(a) + ddx(b)) shape reached via linearity first, then each collapses
        Expression expr = parse("ddx(ddx(T)+ddx(T))");
        // NOTE: this combines two identical ddx(ddx(T)) terms via ADD; distribute() alone
        // does not run combineLikeTerms, so both collapse individually but stay summed
        assertEquals("(d2dx2(T) + d2dx2(T))", ExpressionAlgebra.distribute(expr).toString());
    }
}
