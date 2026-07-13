package com.rae.formicapi.equation_parsing;

import com.rae.formicapi.foundation.math.pde.Field;
import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import com.rae.formicapi.foundation.math.pde.ast.BinaryExpression;
import com.rae.formicapi.foundation.math.pde.ast.BinaryOperators;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import com.rae.formicapi.foundation.math.pde.ast.ExpressionAlgebra;
import com.rae.formicapi.foundation.math.pde.stencil.DiscretizedVariableExpression;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestDiscretizedVariableExpression {

    private static SymbolBinding variable(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), role);
    }

    private static final SymbolBinding T = variable("T", SymbolRole.UNKNOWN);

    // ------------------------------------------------------------------
    // Printing across dimensionalities
    // ------------------------------------------------------------------

    @Test
    void printsCentralNodeAsBareSymbolNameRegardlessOfDimension() {
        assertEquals("T", Expression.print(DiscretizedVariableExpression.atCentralNode(T, 1)));
        assertEquals("T", Expression.print(DiscretizedVariableExpression.atCentralNode(T, 2)));
        assertEquals("T", Expression.print(DiscretizedVariableExpression.atCentralNode(T, 3)));
    }

    @Test
    void printsOneDimensionalOffset() {
        Expression e = DiscretizedVariableExpression.atCentralNode(T, 1).withSpatialOffset(1);
        assertEquals("T[1]", Expression.print(e));
    }

    @Test
    void printsTwoDimensionalOffset() {
        Expression e = DiscretizedVariableExpression.atCentralNode(T, 2).withSpatialOffset(1, -1);
        assertEquals("T[1,-1]", Expression.print(e));
    }

    @Test
    void printsThreeDimensionalOffset() {
        Expression e = DiscretizedVariableExpression.atCentralNode(T, 3).withSpatialOffset(-1, 0, 0);
        assertEquals("T[-1,0,0]", Expression.print(e));
    }

    @Test
    void printsCombinedSpatialAndTemporalOffset() {
        Expression e = DiscretizedVariableExpression.atCentralNode(T, 3, -1).withSpatialOffset(-1, 0, 0);
        assertEquals("T[-1,0,0]{-1}", Expression.print(e));
    }

    // ------------------------------------------------------------------
    // withSpatialOffset dimension validation
    // ------------------------------------------------------------------

    @Test
    void withSpatialOffsetRejectsMismatchedArity() {
        DiscretizedVariableExpression twoD = DiscretizedVariableExpression.atCentralNode(T, 2);
        assertThrows(IllegalArgumentException.class, () -> twoD.withSpatialOffset(1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> twoD.withSpatialOffset(1));
    }

    // ------------------------------------------------------------------
    // equals/hashCode — the array pitfall, tested explicitly
    // ------------------------------------------------------------------

    @Test
    void sameOffsetsFromIndependentArraysAreEqual() {
        // Deliberately two separate int[] instances with equal contents, to catch
        // reference-equality regressions on the array field.
        Expression a = new DiscretizedVariableExpression(T, new int[]{1, 0, 0}, 0);
        Expression b = new DiscretizedVariableExpression(T, new int[]{1, 0, 0}, 0);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void differentSpatialOffsetsAreNotEqual() {
        Expression a = new DiscretizedVariableExpression(T, new int[]{1, 0, 0}, 0);
        Expression b = new DiscretizedVariableExpression(T, new int[]{-1, 0, 0}, 0);
        assertNotEquals(a, b);
    }

    @Test
    void differentTemporalOffsetsAreNotEqual() {
        Expression a = DiscretizedVariableExpression.atCentralNode(T, 3, 0);
        Expression b = DiscretizedVariableExpression.atCentralNode(T, 3, -1);
        assertNotEquals(a, b);
    }

    @Test
    void mutatingReturnedArrayDoesNotAffectInstance() {
        DiscretizedVariableExpression e = DiscretizedVariableExpression.atCentralNode(T, 3).withSpatialOffset(1, 0, 0);
        int[] returned = e.spatialOffset();
        returned[0] = 999;
        assertEquals(1, e.offset(0));
    }

    // ------------------------------------------------------------------
    // Interaction with ExpressionAlgebra
    // ------------------------------------------------------------------

    @Test
    void combineLikeTermsMergesIndependentlyBuiltSameOffsetReferences() {
        Expression a = new DiscretizedVariableExpression(T, new int[]{0, 0, 0}, 0);
        Expression b = new DiscretizedVariableExpression(T, new int[]{0, 0, 0}, 0);
        Expression sum = new BinaryExpression(BinaryOperators.ADD, a, b);

        assertEquals("(2.0 * T)", Expression.print(ExpressionAlgebra.combineLikeTerms(sum)));
    }

    @Test
    void combineLikeTermsKeepsDifferentOffsetsSeparate() {
        Expression plus  = DiscretizedVariableExpression.atCentralNode(T, 3).withSpatialOffset(1, 0, 0);
        Expression minus = DiscretizedVariableExpression.atCentralNode(T, 3).withSpatialOffset(-1, 0, 0);
        Expression sum = new BinaryExpression(BinaryOperators.ADD, plus, minus);

        assertEquals("(T[1,0,0] + T[-1,0,0])", Expression.print(ExpressionAlgebra.combineLikeTerms(sum)));
    }

    @Test
    void isCentralAndIsCurrentTimeLevel() {
        DiscretizedVariableExpression central = DiscretizedVariableExpression.atCentralNode(T, 3);
        assertTrue(central.isCentral());
        assertTrue(central.isCurrentTimeLevel());

        assertFalse(central.withSpatialOffset(0, 1, 0).isCentral());
        assertFalse(central.withTemporalOffset(-1).isCurrentTimeLevel());
    }
}