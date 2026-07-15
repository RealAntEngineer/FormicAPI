package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.ScalarAlgebra;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link ScalarAlgebra#combineLikeTerms} and {@link ScalarAlgebra#factorCommonFactor}.
 */
public class TestScalarAlgebra {

    // ------------------------------------------------------------------
    // combineLikeTerms
    // ------------------------------------------------------------------

    @Test
    void combineLikeTermsMergesTwoIdenticalTerms() {
        Expression expr = parse("a+a");
        assertEquals("2.0 * a", ScalarAlgebra.combineLikeTerms(expr).prettyPrint());
    }

    @Test
    void combineLikeTermsMergesThreeIdenticalTerms() {
        Expression expr = parse("a+a+a");
        assertEquals("3.0 * a", ScalarAlgebra.combineLikeTerms(expr).prettyPrint());
    }

    @Test
    void combineLikeTermsCancelsOppositeSignedIdenticalTerms() {
        Expression expr = parse("a-a");
        assertEquals("0.0", ScalarAlgebra.combineLikeTerms(expr).prettyPrint());
    }

    @Test
    void combineLikeTermsLeavesDistinctTermsAlone() {
        Expression expr = parse("a+b");
        assertEquals("a + b", ScalarAlgebra.combineLikeTerms(expr).prettyPrint());
    }

    @Test
    void combineLikeTermsRecursesIntoMultiplication() {
        Expression expr = parse("a*(b+b)");
        assertEquals("a * 2.0 * b", ScalarAlgebra.combineLikeTerms(expr).prettyPrint());
    }

    @Test
    void combineLikeTermsRecursesIntoUnaryOperator() {
        Expression expr = parse("grad(a+a)");
        assertEquals("grad(2.0 * a)", ScalarAlgebra.combineLikeTerms(expr).prettyPrint());
    }

    // ------------------------------------------------------------------
    // factorCommonFactor
    // ------------------------------------------------------------------

    @Test
    void factorCommonFactorExtractsSharedVariable() {
        Expression expr = parse("a*c+a");
        assertEquals("a * (1.0 + c)", ScalarAlgebra.factorCommonFactor(expr).prettyPrint());
    }

    @Test
    void factorCommonFactorFallsBackToCombineWhenTermsAreIdentical() {
        Expression expr = parse("a+a");
        assertEquals("2.0 * a", ScalarAlgebra.factorCommonFactor(expr).prettyPrint());
    }

    @Test
    void factorCommonFactorExtractsFromThreeTerms() {
        Expression expr = parse("a*c+a*d+a");
        assertEquals("a * (1.0 + c + d)", ScalarAlgebra.factorCommonFactor(expr).prettyPrint());
    }

    @Test
    void factorCommonFactorLeavesExpressionAloneWhenNoFactorIsShared() {
        Expression expr = parse("a+b");
        assertEquals("a + b", ScalarAlgebra.factorCommonFactor(expr).prettyPrint());
    }

    @Test
    void factorCommonFactorDoesNotExtractAPartiallySharedFactor() {
        // only 'a' and 'c' share a factor; 'b' doesn't participate -> out of scope, left uncombined
        Expression expr = parse("a*c+a*c+b");
        Expression result = ScalarAlgebra.factorCommonFactor(expr);
        // a*c and a*c ARE identical terms, so they still combine via the grouping step
        assertEquals("2.0 * a * c + b", result.prettyPrint());
    }

    @Test
    void factorCommonFactorRecursesIntoUnaryOperator() {
        Expression expr = parse("grad(a*c+a)");
        assertEquals("grad(a * (1.0 + c))", ScalarAlgebra.factorCommonFactor(expr).prettyPrint());
    }

    @Test
    void factorCommonFactorRecursesIntoMultiplication() {
        Expression expr = parse("b*(a*c+a)");
        assertEquals("b * a * (1.0 + c)", ScalarAlgebra.factorCommonFactor(expr).prettyPrint());
    }
}
