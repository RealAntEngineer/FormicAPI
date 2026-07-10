package com.rae.formicapi.foundation.math.pde;


import com.rae.formicapi.foundation.math.pde.ast.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;


public final class Equation {
    private static final int MAX_DEPTH = 100;

    private final Map<String, SymbolBinding> symbols =
            new HashMap<>();

    private final Expression left;
    private final Expression right;


    public Equation(String expression, SymbolBinding... fieldBindings) {
        for (SymbolBinding bind : fieldBindings) {

            symbols.put(bind.field().name(), bind);
        }
        expression = expression.replaceAll("\\s+", " ");
        String[] split = expression.split("=");

        if (split.length != 2)
            throw new RuntimeException("Failed to parse equation, wrong number of equality : "
                    + split.length + " for string" + expression);

        left = parseExpression(split[0], 0);
        right = parseExpression(split[1], 0);
    }

    //split the text in 2 so that the
    public static String[] groupByParenthesis(String text) {

        if (!text.startsWith("(") || text.length() == 1) {
            throw new RuntimeException("Unmatched parenthesis at start");
        }

        int parenthesisCount = 1;
        for (int i = 1; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '(') {
                parenthesisCount++;
            } else if (c == ')') {
                parenthesisCount--;
            }

            if (parenthesisCount <= 0) {
                if (i <= text.length() - 1) {
                    return new String[]{text.substring(1, i), text.substring(i + 1)};
                } else {
                    return new String[]{text.substring(1, text.length() - 1), ""};
                }
            }

        }

        throw new RuntimeException("Unmatched parenthesis");
    }

    public Expression parseExpression(String text, int depth) {

        if (depth > MAX_DEPTH)
            throw new RuntimeException("Max depth reached");

        //0. cleaning

        text = text.trim();

        //1. march through characters until delimitation is hit  : ( or space.
        // expression such as k_3i, k*i will be recognized as 1 identifier

        String identifier      = text;
        String remainingString = "";

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == ' ') {
                identifier = text.substring(0, i);
                remainingString = text.substring(i + 1);//we remove the char if it's a " "
                break;
            } else if (c == '(' || c == '+' || c == '-' || c == '/' || c == '*') {
                identifier = text.substring(0, i);
                remainingString = text.substring(i);//we keep the parenthesis
                break;
            }
        }

        assert !identifier.contains(" ");
        assert !identifier.contains("(");

        Expression left;

        if (identifier.isEmpty()) {
            if (!remainingString.startsWith("(")) {
                throw new RuntimeException("Expression is empty");//we can't begin with a ' ' since we trim
            } else {
                String[] splited = groupByParenthesis(remainingString);

                left = parseExpression(splited[0], depth + 1);
                remainingString = splited[1];

            }
        } else {
            //2. once identifier is found we try to know what it is in this order : (value, Symbol, Operator)

            if (identifier.startsWith("0") || identifier.startsWith("1") || identifier.startsWith("2")
                    || identifier.startsWith("3") || identifier.startsWith("4") || identifier.startsWith("5")
                    || identifier.startsWith("6") || identifier.startsWith("7") || identifier.startsWith("8")
                    || identifier.startsWith("9")) {
                try {
                    double value = Double.parseDouble(identifier);
                    left = new ConstantExpression(value);

                } catch (NumberFormatException formatException) {
                    throw new RuntimeException("Failed to parse value " + identifier + " in expression " + text, formatException);
                }
            } else if (symbols.containsKey(identifier)) {
                left = new VariableExpression(symbols.get(identifier));
            } else {
                Optional<BinaryOperator> optionalBinary = BinaryOperator.parse(identifier);

                if (optionalBinary.isPresent())
                    throw new RuntimeException("Binary with no left expression is illegal");

                Optional<UnaryOperator> optionalUnary = UnaryOperator.parse(identifier);

                if (optionalUnary.isPresent()) {
                    //same treatment as the no identifier
                    String[] splited = groupByParenthesis(remainingString);

                    left = new UnaryExpression(optionalUnary.get(), parseExpression(splited[0], depth + 1));
                    remainingString = splited[1];

                } else {
                    throw new RuntimeException("Unable to parse identifier : " + identifier);
                }
            }
        }


        while (!remainingString.isEmpty()) {
            // remove useless spaces after identifier
            remainingString = remainingString.trim();


            if (remainingString.isEmpty()) {
                return left;
            }

            //search for a binary operator, either explicit or implicit

            if (remainingString.startsWith("(")) {// implicit multiplication


                //group by parenthesis first
                String[] splited = groupByParenthesis(remainingString);

                left = new BinaryExpression(
                        BinaryOperator.MULTIPLY,
                        left, parseExpression(splited[0], depth + 1));
                remainingString = splited[1];

            } else {//explicit
                //the binary operator is supposed to be only 1 character

                Optional<BinaryOperator> optionalBinary = BinaryOperator.parse(String.valueOf(remainingString.charAt(0)));

                if (optionalBinary.isEmpty()) {
                    throw new RuntimeException("Unable to parse binary operator : " + remainingString.charAt(0));

                }
                //seems like this could cause priority issues. -> this creates implicit parenthesis so we could
                // have issue depending on the priority level of the operator

                String[] splited = groupByParenthesis("(" + remainingString.substring(1) + ")");

                left = new BinaryExpression(
                        optionalBinary.get(),
                        left, parseExpression(splited[0], depth + 1));
                remainingString = splited[1];
            }
        }

        return left;
    }

    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }

    @Override
    public String toString() {
        return "Equation{" +
                "symbols=" + symbols +
                ", " + Expression.print(left) +
                " = " + Expression.print(right) +
                '}';
    }
}
