package com.rae.formicapi;

import com.rae.formicapi.foundation.math.pde.*;
import com.rae.formicapi.foundation.math.pde.ast.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestEquation {

    private static SymbolBinding variable(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), role);
    }

    // ------------------------------------------------------------------
    // Real-world expression parsing
    // ------------------------------------------------------------------

    @Test
    void testParsingDiffusionEquation() {

        SymbolBinding k   = variable("k", SymbolRole.COEFFICIENT);
        SymbolBinding res = variable("res", SymbolRole.COEFFICIENT);
        SymbolBinding td  = variable("Td", SymbolRole.COEFFICIENT);
        SymbolBinding t   = variable("T", SymbolRole.UNKNOWN);

        Equation eq = new Equation(
                "div(k * grad(T)) + res * (Td - T) = 0",
                k, res, td, t
        );

        Expression expectedLeft =
                new BinaryExpression(
                        BinaryOperators.ADD,
                        new UnaryExpression(
                                UnaryOperators.DIV,
                                new BinaryExpression(
                                        BinaryOperators.MULTIPLY,
                                        new VariableExpression(k),
                                        new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(t))
                                )
                        ),
                        new BinaryExpression(
                                BinaryOperators.MULTIPLY,
                                new VariableExpression(res),
                                new BinaryExpression(
                                        BinaryOperators.SUBTRACT,
                                        new VariableExpression(td),
                                        new VariableExpression(t)
                                )
                        )
                );

        Expression expectedRight = new ConstantExpression(0.0);

        assertEquals(expectedLeft, eq.getLeft());
        assertEquals(expectedRight, eq.getRight());
    }

    // ------------------------------------------------------------------
    // Operator precedence
    // ------------------------------------------------------------------

    @Test
    void parsingBinaryCombinationTest() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq1 = new Equation("a * b + c = 0", a, b, c);

        Expression expected1 =
                new BinaryExpression(
                        BinaryOperators.ADD,
                        new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(a), new VariableExpression(b)),
                        new VariableExpression(c)
                );

        assertEquals(expected1, eq1.getLeft());

        Equation eq2 = new Equation("a*b-c = 0", a, b, c);

        Expression expected2 =
                new BinaryExpression(
                        BinaryOperators.SUBTRACT,
                        new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(a), new VariableExpression(b)),
                        new VariableExpression(c)
                );

        assertEquals(expected2, eq2.getLeft());
    }

    @Test
    void additionDoesNotStealOperandFromMultiplication() {
        // regression test: '+' must not bind '4' before '*' does, i.e. 3*4+2 != 3*(4+2)

        SymbolBinding x = variable("x", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("3 * 4 + 2 = 0", x);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.ADD,
                        new BinaryExpression(BinaryOperators.MULTIPLY, new ConstantExpression(3), new ConstantExpression(4)),
                        new ConstantExpression(2)
                );

        assertEquals(expected, eq.getLeft());
    }

    @Test
    void powerHasHigherPriorityThanMultiplication() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("a * b ^ c = 0", a, b, c);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.MULTIPLY,
                        new VariableExpression(a),
                        new BinaryExpression(BinaryOperators.POWER, new VariableExpression(b), new VariableExpression(c))
                );

        assertEquals(expected, eq.getLeft());
    }

    // ------------------------------------------------------------------
    // Associativity
    // ------------------------------------------------------------------

    @Test
    void parsingAssociativityTest() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("a-b-c=0", a, b, c);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.SUBTRACT,
                        new BinaryExpression(BinaryOperators.SUBTRACT, new VariableExpression(a), new VariableExpression(b)),
                        new VariableExpression(c)
                );

        assertEquals(expected, eq.getLeft());
    }

    @Test
    void powerIsRightAssociative() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("a^b^c=0", a, b, c);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.POWER,
                        new VariableExpression(a),
                        new BinaryExpression(BinaryOperators.POWER, new VariableExpression(b), new VariableExpression(c))
                );

        assertEquals(expected, eq.getLeft());
    }

    // ------------------------------------------------------------------
    // Implicit multiplication
    // ------------------------------------------------------------------

    @Test
    void implicitMultiplicationBeforeParenthesis() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("2(a + b) = 0", a, b);

        Expression expected =
                new BinaryExpression(
                        BinaryOperators.MULTIPLY,
                        new ConstantExpression(2),
                        new BinaryExpression(BinaryOperators.ADD, new VariableExpression(a), new VariableExpression(b))
                );

        assertEquals(expected, eq.getLeft());
    }

    @Test
    void implicitMultiplicationBetweenTwoParenthesisGroups() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        Equation eq = new Equation("(a)(b) = 0", a, b);

        Expression expected =
                new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(a), new VariableExpression(b));

        assertEquals(expected, eq.getLeft());
    }

    // ------------------------------------------------------------------
    // Parenthesis grouping helper
    // ------------------------------------------------------------------

    @Test
    void parenthesisGroupingTest() {
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

    // ------------------------------------------------------------------
    // Error cases
    // ------------------------------------------------------------------

    @Test
    void unknownIdentifierThrows() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);

        assertThrows(RuntimeException.class, () -> new Equation("a + unknownSymbol = 0", a));
    }

    @Test
    void unknownBinaryOperatorCharacterThrows() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        assertThrows(RuntimeException.class, () -> new Equation("a % b = 0", a, b));
    }

    @Test
    void emptyExpressionThrows() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);

        assertThrows(RuntimeException.class, () -> new Equation(" = 0", a));
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

        Equation eq = new Equation("div(k * grad(T)) + res * (Td - T) = 0", k, res, td, t);

        assertEquals(
                "Equation{(div((k * grad(T))) + (res * (Td - T))) = 0.0, symbols=[k:SCALAR:COEFFICIENT, res:SCALAR:COEFFICIENT, Td:SCALAR:COEFFICIENT, T:SCALAR:UNKNOWN]}",
                eq.toString()
        );
    }
}