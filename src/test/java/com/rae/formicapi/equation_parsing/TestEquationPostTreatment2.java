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

public class TestEquationPostTreatment2 {

    private static final Map<String, SymbolBinding> SYMBOLS = Map.of(
            "a", variable("a"),
            "b", variable("b"),
            "c", variable("c"),
            "d", variable("d")
    );

    private static SymbolBinding variable(String name) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), SymbolRole.COEFFICIENT);
    }

    @Test
    void combineIdenticalTerms() {
        Expression expr = parse("a+a");
        assertEquals("(2.0 * a)", Expression.print(ExpressionAlgebra.combineLikeTerms(expr)));
    }

    private static Expression parse(String expression) {
        return Expression.parseExpression(expression, SYMBOLS, 0);
    }

    @Test
    void combineThreeIdenticalTerms() {
        Expression expr = parse("a+a+a");
        assertEquals("(3.0 * a)", Expression.print(ExpressionAlgebra.combineLikeTerms(expr)));
    }

    @Test
    void cancellingTermsBecomeZero() {
        Expression expr = parse("a-a");
        assertEquals("0.0", Expression.print(ExpressionAlgebra.combineLikeTerms(expr)));
    }

    @Test
    void combineLikeTermsLeavesDistinctTermsAlone() {
        Expression expr = parse("a+b");
        assertEquals("(a + b)", Expression.print(ExpressionAlgebra.combineLikeTerms(expr)));
    }

    @Test
    void combineLikeTermsRecursesIntoMultiplication() {
        Expression expr = parse("a*(b+b)");
        assertEquals("(a * (2.0 * b))", Expression.print(ExpressionAlgebra.combineLikeTerms(expr)));
    }

    @Test
    void combineLikeTermsRecursesIntoUnaryOperator() {
        Expression expr = parse("grad(a+a)");
        assertEquals("grad((2.0 * a))", Expression.print(ExpressionAlgebra.combineLikeTerms(expr)));
    }


    @Test
    void factorSimpleCommonFactor() {
        Expression expr = parse("a*c+a");
        assertEquals("(a * (c + 1.0))", Expression.print(ExpressionAlgebra.factorCommonFactor(expr)));
    }

    @Test
    void factorReducesToCombineWhenTermsIdentical() {
        Expression expr = parse("a+a");
        assertEquals("(2.0 * a)", Expression.print(ExpressionAlgebra.factorCommonFactor(expr)));
    }

    @Test
    void factorThreeTermsWithSharedFactor() {
        Expression expr = parse("a*c+a*d+a");
        assertEquals("(a * ((c + d) + 1.0))", Expression.print(ExpressionAlgebra.factorCommonFactor(expr)));
    }

    @Test
    void noCommonFactorLeavesExpressionAlone() {
        Expression expr = parse("a+b");
        assertEquals("(a + b)", Expression.print(ExpressionAlgebra.factorCommonFactor(expr)));
    }

    @Test
    void partialSharedFactorIsNotExtracted() {
        // only 'a' and 'c' share a factor; 'b' doesn't participate -> out of scope, left uncombined
        Expression expr   = parse("a*c+a*c+b");
        Expression result = ExpressionAlgebra.factorCommonFactor(expr);
        // a*c and a*c ARE identical terms, so they still combine via the grouping step
        assertEquals("((2.0 * (a * c)) + b)", Expression.print(result));
    }

    @Test
    void factorRecursesIntoUnaryOperator() {
        Expression expr = parse("grad(a*c+a)");
        assertEquals("grad((a * (c + 1.0)))", Expression.print(ExpressionAlgebra.factorCommonFactor(expr)));
    }

    @Test
    void factorCommonFactorRecursesIntoMultiplication() {
        Expression expr = parse("b*(a*c+a)");
        assertEquals("(b * (a * (c + 1.0)))", Expression.print(ExpressionAlgebra.factorCommonFactor(expr)));
    }
}