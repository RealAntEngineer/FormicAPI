package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.SymbolBinding;

public interface Expression {

    static String print(
            Expression expression) {

        String result;

        switch (expression) {
            case ConstantExpression(double value) -> result = Double.toString(value);

            case VariableExpression(SymbolBinding binding) -> result = binding.field().name();

            case UnaryExpression(UnaryOperator operator, Expression child) ->
                    result = operator.representation() + "(" + print(child) + ")";

            case BinaryExpression(BinaryOperator operator, Expression left1, Expression right1) -> {

                String left = print(left1);

                String right = print(right1);

                result = "(" + left + " " + operator.representation + " " + right + ")";
            }
            default -> throw new RuntimeException("Unknown expression type " + expression.getClass()
            );
        }

        return result;
    }
}