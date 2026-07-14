package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.ast.Expression;
import com.rae.formicapi.foundation.math.pde.ScalarAlgebra;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEquationPostTreatment2 {

    @Test
    void combineIdenticalTerms() {
        Expression expr = parse("a+a");
        assertEquals("(2.0 * a)", ScalarAlgebra.combineLikeTerms(expr).toString());
    }

    @Test
    void combineThreeIdenticalTerms() {
        Expression expr = parse("a+a+a");
        assertEquals("(3.0 * a)", ScalarAlgebra.combineLikeTerms(expr).toString());
    }

    @Test
    void cancellingTermsBecomeZero() {
        Expression expr = parse("a-a");
        assertEquals("0.0", ScalarAlgebra.combineLikeTerms(expr).toString());
    }

    @Test
    void combineLikeTermsLeavesDistinctTermsAlone() {
        Expression expr = parse("a+b");
        assertEquals("(a + b)", ScalarAlgebra.combineLikeTerms(expr).toString());
    }

    @Test
    void combineLikeTermsRecursesIntoMultiplication() {
        Expression expr = parse("a*(b+b)");
        assertEquals("(a * (2.0 * b))", ScalarAlgebra.combineLikeTerms(expr).toString());
    }

    @Test
    void combineLikeTermsRecursesIntoUnaryOperator() {
        Expression expr = parse("grad(a+a)");
        assertEquals("grad((2.0 * a))", ScalarAlgebra.combineLikeTerms(expr).toString());
    }


    @Test
    void factorSimpleCommonFactor() {
        Expression expr = parse("a*c+a");
        assertEquals("(a * (c + 1.0))", ScalarAlgebra.factorCommonFactor(expr).toString());
    }

    @Test
    void factorReducesToCombineWhenTermsIdentical() {
        Expression expr = parse("a+a");
        assertEquals("(2.0 * a)", ScalarAlgebra.factorCommonFactor(expr).toString());
    }

    @Test
    void factorThreeTermsWithSharedFactor() {
        Expression expr = parse("a*c+a*d+a");
        assertEquals("(a * ((c + d) + 1.0))", ScalarAlgebra.factorCommonFactor(expr).toString());
    }

    @Test
    void noCommonFactorLeavesExpressionAlone() {
        Expression expr = parse("a+b");
        assertEquals("(a + b)", ScalarAlgebra.factorCommonFactor(expr).toString());
    }

    @Test
    void partialSharedFactorIsNotExtracted() {
        // only 'a' and 'c' share a factor; 'b' doesn't participate -> out of scope, left uncombined
        Expression expr   = parse("a*c+a*c+b");
        Expression result = ScalarAlgebra.factorCommonFactor(expr);
        // a*c and a*c ARE identical terms, so they still combine via the grouping step
        assertEquals("((2.0 * (a * c)) + b)", result.toString());
    }

    @Test
    void factorRecursesIntoUnaryOperator() {
        Expression expr = parse("grad(a*c+a)");
        assertEquals("grad((a * (c + 1.0)))", ScalarAlgebra.factorCommonFactor(expr).toString());
    }

    @Test
    void factorCommonFactorRecursesIntoMultiplication() {
        Expression expr = parse("b*(a*c+a)");
        assertEquals("(b * (a * (c + 1.0)))", ScalarAlgebra.factorCommonFactor(expr).toString());
    }
}