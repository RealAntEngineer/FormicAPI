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

public class TestEquationPostTreatment3 {

    @Test
    void gradDistributesOverAddition() {
        Expression expr = parse("grad(a+b)");
        assertEquals("(grad(a) + grad(b))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void divDistributesOverSubtraction() {
        Expression expr = parse("div(e1-e2)");
        assertEquals("(div(e1) - div(e2))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void laplacianDistributesOverAddition() {
        Expression expr = parse("lap(a+b)");
        assertEquals("(lap(a) + lap(b))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void divOfGradCollapsesToLaplacian() {
        Expression expr = parse("div(grad(T))");
        assertEquals("lap(T)", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void divOfGradCollapsesAfterDistributingASum() {
        // grad(a+b) distributes first, then div distributes over the sum,
        // then each div(grad(...)) term individually collapses to lap(...)
        Expression expr = parse("div(grad(a+b))");
        assertEquals("(lap(a) + lap(b))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void gradientProductRuleBothNonConstant() {
        Expression expr = parse("grad(a*b)");
        assertEquals("((grad(a) * b) + (a * grad(b)))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void gradientOfConstantTimesFieldPullsConstantOut() {
        Expression expr = parse("grad(2*a)");
        assertEquals("(2.0 * grad(a))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void gradientOfConstantIsZero() {
        Expression expr = parse("grad(5)");
        assertEquals("0.0", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void diffusionTermIsLeftIntactByDistribution() {
        // div(k * grad(T)) is a recognized compound pattern (DiffusionTermMatcher) and
        // should NOT be torn apart by a div product rule we deliberately don't implement
        Expression expr = parse("div(k*grad(T))");
        assertEquals("div((k * grad(T)))", ExpressionAlgebra.distribute(expr).toString());
    }

    @Test
    void ddtDistributesOverAddition() {
        Expression expr = parse("ddt(a+b)");
        assertEquals("(ddt(a) + ddt(b))", ExpressionAlgebra.distribute(expr).toString());
    }
}
