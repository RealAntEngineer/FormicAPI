package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.*;
import com.rae.formicapi.foundation.math.pde.ast.*;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.pde.PDEUtil.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Everything that exercises {@link Equation}'s string-parsing pipeline: turning
 * a textual expression into an AST. Grouping-only concerns (parenthesis matching)
 * live in {@link TestExpressionParenthesisGrouping} since that's a static
 * {@link Expression} helper, not an {@link Equation} concern.
 */
public class TestEquationParsing {

    // ------------------------------------------------------------------
    // Real-world expression parsing
    // ------------------------------------------------------------------

    @Test
    void parsesDiffusionEquationIntoExpectedAst() {

        SymbolBinding k   = variable("k", SymbolRole.COEFFICIENT);
        SymbolBinding res = variable("res", SymbolRole.COEFFICIENT);
        SymbolBinding td  = variable("Td", SymbolRole.COEFFICIENT);
        SymbolBinding t   = variable("T", SymbolRole.UNKNOWN);

        Equation eq = new Equation(
                "div(k * grad(T)) + res * (Td - T) = 0",
                3, k, res, td, t
        );

        Expression expectedLeft =
                new BinaryExpression(
                        BinaryOperators.ADD,
                        new UnaryExpression(
                                UnaryOperators.DIV,
                                new BinaryExpression(
                                        BinaryOperators.MULTIPLY,
                                        new VariableExpression(k, 3),
                                        new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(t, 3))
                                )
                        ),
                        new BinaryExpression(
                                BinaryOperators.MULTIPLY,
                                new VariableExpression(res, 3),
                                new BinaryExpression(
                                        BinaryOperators.SUBTRACT,
                                        new VariableExpression(td, 3),
                                        new VariableExpression(t, 3)
                                )
                        )
                );

        Expression expectedRight = new ConstantExpression(0.0, 3);

        assertEquals(expectedLeft, eq.getLeft());
        assertEquals(expectedRight, eq.getRight());
    }

    @Test
    void parsesVectorEquationInvolvingGradOfDiv() {

        SymbolBinding k   = variable("k", SymbolRole.COEFFICIENT);
        SymbolBinding res = variable("res", SymbolRole.COEFFICIENT);
        SymbolBinding td  = variable("Td", SymbolRole.COEFFICIENT);
        SymbolBinding V   = vector("V", SymbolRole.UNKNOWN);

        Equation eq = new Equation(
                "grad(div(V))  = 0",
                3, k, res, td, V
        );

        Expression leftDistributed = eq.getLeft().expand();
        Expression leftSimplified = leftDistributed.expand();

        // No assertion beyond "doesn't throw" here: this test only pins down
        // that a vector-valued equation with a nested grad(div(...)) parses
        // and expands without error. Semantic checks for grad/div behavior
        // live in TestDifferentialOperatorLinearity / TestDerivativeComposition.
        assertNotNull(leftSimplified);
    }

    // ------------------------------------------------------------------
    // Operator precedence
    // ------------------------------------------------------------------

    @Test
    void multiplicationBindsTighterThanAdditionAndSubtraction() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq1 = new Equation("a * b + c = 0", 3, a, b, c);

        Expression expected1 =
                new BinaryExpression(
                        BinaryOperators.ADD,
                        new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(a, 3), new VariableExpression(b, 3)),
                        new VariableExpression(c, 3)
                );

        assertEquals(expected1, eq1.getLeft());

        Equation eq2 = new Equation("a*b-c = 0", 3, a, b, c);

        Expression expected2 =
                new BinaryExpression(
                        BinaryOperators.SUBTRACT,
                        new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(a, 3), new VariableExpression(b, 3)),
                        new VariableExpression(c, 3)
                );

        assertEquals(expected2, eq2.getLeft());
    }

    @Test
    void additionDoesNotStealOperandFromMultiplication() {
        // regression test: '+' must not bind '4' before '*' does, i.e. 3*4+2 != 3*(4+2)

        SymbolBinding x = variable("x", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("3 * 4 + 2 = 0", 3, x);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.ADD,
                        new BinaryExpression(BinaryOperators.MULTIPLY, new ConstantExpression(3, 3), new ConstantExpression(4, 3)),
                        new ConstantExpression(2, 3)
                );

        assertEquals(expected, eq.getLeft());
    }

    @Test
    void powerHasHigherPriorityThanMultiplication() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("a * b ^ c = 0", 3, a, b, c);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.MULTIPLY,
                        new VariableExpression(a, 3),
                        new BinaryExpression(BinaryOperators.POWER, new VariableExpression(b, 3), new VariableExpression(c, 3))
                );

        assertEquals(expected, eq.getLeft());
    }

    // ------------------------------------------------------------------
    // Associativity
    // ------------------------------------------------------------------

    @Test
    void subtractionIsLeftAssociative() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("a-b-c=0", 3, a, b, c);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.SUBTRACT,
                        new BinaryExpression(BinaryOperators.SUBTRACT, new VariableExpression(a, 3), new VariableExpression(b, 3)),
                        new VariableExpression(c, 3)
                );

        assertEquals(expected, eq.getLeft());
    }

    @Test
    void powerIsRightAssociative() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("a^b^c=0", 3, a, b, c);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.POWER,
                        new VariableExpression(a, 3),
                        new BinaryExpression(BinaryOperators.POWER, new VariableExpression(b, 3), new VariableExpression(c, 3))
                );

        assertEquals(expected, eq.getLeft());
    }

    // ------------------------------------------------------------------
    // Implicit multiplication
    // ------------------------------------------------------------------

    @Test
    void numberDirectlyBeforeParenthesisImpliesMultiplication() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("2(a + b) = 0", 3, a, b);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.MULTIPLY,
                        new ConstantExpression(2, 3),
                        new BinaryExpression(BinaryOperators.ADD, new VariableExpression(a, 3), new VariableExpression(b, 3))
                );

        assertEquals(expected, eq.getLeft());
    }

    @Test
    void twoAdjacentParenthesisGroupsImplyMultiplication() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("(a)(b) = 0", 3, a, b);

        Expression expected =
                new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(a, 3),
                        new VariableExpression(b, 3));

        assertEquals(expected, eq.getLeft());
    }

    // ------------------------------------------------------------------
    // Error cases
    // ------------------------------------------------------------------

    @Test
    void referencingUndeclaredSymbolThrows() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);

        assertThrows(RuntimeException.class, () -> new Equation("a + unknownSymbol = 0", 3, a));
    }

    @Test
    void unsupportedBinaryOperatorCharacterThrows() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        assertThrows(RuntimeException.class, () -> new Equation("a % b = 0", 3, a, b));
    }

    @Test
    void emptyLeftHandSideThrows() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);

        assertThrows(RuntimeException.class, () -> new Equation(" = 0", 3, a));
    }

    // ------------------------------------------------------------------
    // toString formatting
    // ------------------------------------------------------------------

    @Test
    void toStringPreservesSymbolDeclarationOrder() {

        SymbolBinding k   = variable("k", SymbolRole.COEFFICIENT);
        SymbolBinding res = variable("res", SymbolRole.COEFFICIENT);
        SymbolBinding td  = variable("Td", SymbolRole.COEFFICIENT);
        SymbolBinding t   = variable("T", SymbolRole.UNKNOWN);

        Equation eq = new Equation("div(k * grad(T)) + res * (Td - T) = 0", 3, k, res, td, t);

        assertEquals(
                "Equation{div(k * grad(T)) + res * (Td - T) = 0.0, symbols=[k:SCALAR:COEFFICIENT, res:SCALAR:COEFFICIENT, Td:SCALAR:COEFFICIENT, T:SCALAR:UNKNOWN]}",
                eq.toString()
        );
    }
}