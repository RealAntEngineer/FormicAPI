package com.rae.formicapi;

import com.rae.formicapi.foundation.math.pde.*;
import com.rae.formicapi.foundation.math.pde.ast.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestExpression {



    @Test()
    void testParsingDiffusionEquation() {

        SymbolBinding k   = variable("k", SymbolRole.COEFFICIENT);
        SymbolBinding res = variable("res", SymbolRole.COEFFICIENT);
        SymbolBinding td = variable("Td", SymbolRole.COEFFICIENT);
        SymbolBinding t = variable("T", SymbolRole.UNKNOWN);

        Equation eq = new Equation(
                "div(k * grad(T)) + res * (Td - T) = 0",
                k,
                res,
                td,
                t
        );

        Expression expectedLeft =
                new BinaryExpression(
                        BinaryOperator.ADD,
                        new UnaryExpression(
                                UnaryOperator.DIV,
                                new BinaryExpression(
                                        BinaryOperator.MULTIPLY,
                                        new VariableExpression(k),
                                        new UnaryExpression(
                                                UnaryOperator.GRAD,
                                                new VariableExpression(t)
                                        )
                                )
                        ),
                        new BinaryExpression(
                                BinaryOperator.MULTIPLY,
                                new VariableExpression(res),
                                new BinaryExpression(
                                        BinaryOperator.SUBTRACT,
                                        new VariableExpression(td),
                                        new VariableExpression(t)
                                )
                        )
                );

        Expression expectedRight =
                new ConstantExpression(0.0);

        assertEquals(expectedLeft, eq.getLeft());
        assertEquals(expectedRight, eq.getRight());

        System.out.println(eq);
    }

    @Test
    void parsingBinaryCombinationTest() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);

        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);

        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);


        Equation eq1 = new Equation(
                "(a * b) + c = 0",
                a, b, c
        );


        Expression expected1 =
                new BinaryExpression(
                        BinaryOperator.ADD,
                        new BinaryExpression(
                                BinaryOperator.MULTIPLY,
                                new VariableExpression(a),
                                new VariableExpression(b)
                        ),
                        new VariableExpression(c)
                );


        assertEquals(expected1, eq1.getLeft());



        Equation eq2 = new Equation(
                "(a*b)-c = 0",
                a, b, c
        );


        Expression expected2 =
                new BinaryExpression(
                        BinaryOperator.SUBTRACT,
                        new BinaryExpression(
                                BinaryOperator.MULTIPLY,
                                new VariableExpression(a),
                                new VariableExpression(b)
                        ),
                        new VariableExpression(c)
                );


        assertEquals(expected2, eq2.getLeft());
    }

    private static SymbolBinding variable(String a, SymbolRole coefficient) {
        return new SymbolBinding(
                new Field(a, FieldType.SCALAR),
                coefficient
        );
    }

    @Test
    void parsingAssociativityTest() {

        SymbolBinding a = variable("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = variable("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = variable("c", SymbolRole.COEFFICIENT);


        Equation eq =
                new Equation(
                        "a-b-c=0",
                        a,b,c
                );


        Expression expected =
                new BinaryExpression(
                        BinaryOperator.SUBTRACT,
                        new BinaryExpression(
                                BinaryOperator.SUBTRACT,
                                new VariableExpression(a),
                                new VariableExpression(b)
                        ),
                        new VariableExpression(c)
                );


        assertEquals(expected, eq.getLeft());
    }


    @Test
    void parenthesisGroupingTest(){
        String[] grouped = Equation.groupByParenthesis("(a + b * c ((()))) + 1");

        assertArrayEquals(new String[]{"a + b * c ((()))", " + 1"}, grouped);
    }
}
