package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.Equation;
import com.rae.formicapi.foundation.math.pde.ast.Expression;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for {@link Expression#groupByParenthesis(String)}, the low-level
 * tokenizer helper that splits a string into top-level parenthesis groups.
 * Previously buried inside TestEquation even though it has nothing to do
 * with {@link Equation} construction.
 */
public class TestExpressionParenthesisGrouping {

    @Test
    void splitsLeadingGroupFromTrailingRemainder() {
        String[] grouped = Expression.groupByParenthesis("(a + b * c ((()))) + 1");
        assertArrayEquals(new String[]{"a + b * c ((()))", " + 1"}, grouped);
    }

    @Test
    void unmatchedOpeningParenthesisThrows() {
        assertThrows(RuntimeException.class, () -> Expression.groupByParenthesis("(a + b"));
    }

    @Test
    void missingLeadingParenthesisThrows() {
        assertThrows(RuntimeException.class, () -> Expression.groupByParenthesis("a + b)"));
    }
}