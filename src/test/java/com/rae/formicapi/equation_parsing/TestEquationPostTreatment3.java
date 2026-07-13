package com.rae.formicapi.equation_parsing;

import com.rae.formicapi.foundation.math.pde.Field;
import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import com.rae.formicapi.foundation.math.pde.ast.ExpressionAlgebra;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEquationPostTreatment3 {

    private static SymbolBinding variable(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), role);
    }

    private static final Map<String, SymbolBinding> SYMBOLS = Map.of(
            "a", variable("a", SymbolRole.COEFFICIENT),
            "b", variable("b", SymbolRole.COEFFICIENT),
            "k", variable("k", SymbolRole.COEFFICIENT),
            "T", variable("T", SymbolRole.UNKNOWN)
    );

    private static Expression parse(String expression) {
        return Expression.parseExpression(expression, SYMBOLS, 0);
    }

    @Test
    void gradDistributesOverAddition() {
        Expression expr = parse("grad(a+b)");
        assertEquals("(grad(a) + grad(b))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void divDistributesOverSubtraction() {
        Expression expr = parse("div(a-b)");
        assertEquals("(div(a) - div(b))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void laplacianDistributesOverAddition() {
        Expression expr = parse("lap(a+b)");
        assertEquals("(lap(a) + lap(b))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void divOfGradCollapsesToLaplacian() {
        Expression expr = parse("div(grad(T))");
        assertEquals("lap(T)", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void divOfGradCollapsesAfterDistributingASum() {
        // grad(a+b) distributes first, then div distributes over the sum,
        // then each div(grad(...)) term individually collapses to lap(...)
        Expression expr = parse("div(grad(a+b))");
        assertEquals("(lap(a) + lap(b))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void gradientProductRuleBothNonConstant() {
        Expression expr = parse("grad(a*b)");
        assertEquals("((grad(a) * b) + (a * grad(b)))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void gradientOfConstantTimesFieldPullsConstantOut() {
        Expression expr = parse("grad(2*a)");
        assertEquals("(2.0 * grad(a))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void gradientOfConstantIsZero() {
        Expression expr = parse("grad(5)");
        assertEquals("0.0", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void diffusionTermIsLeftIntactByDistribution() {
        // div(k * grad(T)) is a recognized compound pattern (DiffusionTermMatcher) and
        // should NOT be torn apart by a div product rule we deliberately don't implement
        Expression expr = parse("div(k*grad(T))");
        assertEquals("div((k * grad(T)))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }

    @Test
    void ddtDistributesOverAddition() {
        Expression expr = parse("ddt(a+b)");
        assertEquals("(ddt(a) + ddt(b))", Expression.print(ExpressionAlgebra.distribute(expr)));
    }
}
