package com.rae.formicapi.foundation.math.pde;


import com.rae.formicapi.foundation.math.pde.ast.*;

import java.util.*;


public final class Equation {
    private static final int MAX_DEPTH = 100;

    private final Map<String, SymbolBinding> symbols = new LinkedHashMap<>();

    private final Expression left;
    private final Expression right;


    public Equation(String expression, SymbolBinding... fieldBindings) {
        for (SymbolBinding bind : fieldBindings) {
            SymbolBinding previous = symbols.put(bind.field().name(), bind);
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate symbol: " + bind.field().name());
            }
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

        text = text.trim();

        List<Expression>     operands  = new ArrayList<>();
        List<BinaryOperator> operators = new ArrayList<>();

        String remaining = text;

        while (true) {
            remaining = remaining.trim();

            Expression operand;

            if (remaining.startsWith("(")) {
                String[] splited = groupByParenthesis(remaining);
                operand = parseExpression(splited[0], depth + 1);
                remaining = splited[1];

            } else {
                String identifier = remaining;
                String rest = "";

                for (int i = 0; i < remaining.length(); i++) {
                    char c = remaining.charAt(i);
                    if (c == ' ') {
                        identifier = remaining.substring(0, i);
                        rest = remaining.substring(i + 1);
                        break;
                    } else if (c == '(' || BinaryOperator.isOperatorChar(c)) {
                        identifier = remaining.substring(0, i);
                        rest = remaining.substring(i);
                        break;
                    }
                }

                if (identifier.isEmpty())
                    throw new RuntimeException("Expression is empty");

                if (Character.isDigit(identifier.charAt(0))) {
                    try {
                        operand = new ConstantExpression(Double.parseDouble(identifier));
                    } catch (NumberFormatException e) {
                        throw new RuntimeException("Failed to parse value " + identifier + " in expression " + text, e);
                    }

                } else if (symbols.containsKey(identifier)) {
                    operand = new VariableExpression(symbols.get(identifier));

                } else {
                    Optional<UnaryOperator> optionalUnary = UnaryOperator.parse(identifier);
                    if (optionalUnary.isPresent()) {
                        String[] splited = groupByParenthesis(rest);
                        operand = new UnaryExpression(optionalUnary.get(), parseExpression(splited[0], depth + 1));
                        rest = splited[1];
                    } else {
                        throw new RuntimeException("Unable to parse identifier : " + identifier);
                    }
                }
                remaining = rest;
            }

            // implicit multiplication: operand directly followed by '(' , e.g. "2(3+4)"
            remaining = remaining.trim();
            while (remaining.startsWith("(")) {
                String[] splited = groupByParenthesis(remaining);
                Expression right = parseExpression(splited[0], depth + 1);
                operand = new BinaryExpression(BinaryOperator.MULTIPLY, operand, right);
                remaining = splited[1].trim();
            }

            operands.add(operand);

            if (remaining.isEmpty())
                break;

            Optional<BinaryOperator> optionalBinary = BinaryOperator.parse(remaining.charAt(0));
            if (optionalBinary.isEmpty())
                throw new RuntimeException("Unable to parse binary operator : " + remaining.charAt(0));

            operators.add(optionalBinary.get());
            remaining = remaining.substring(1);
        }

        return buildTreeByPrecedence(operands, operators);
    }

    private Expression buildTreeByPrecedence(List<Expression> operands, List<BinaryOperator> operators) {
        List<Expression> outOperands = new ArrayList<>(operands);
        List<BinaryOperator> outOperators = new ArrayList<>(operators);

        int maxPriority = 0;
        for (BinaryOperator op : outOperators) {
            maxPriority = Math.max(maxPriority, op.priority);
        }

        for (int priority = maxPriority; priority >= 0; priority--) {

            boolean rightAssociative = isRightAssociativeLevel(outOperators, priority);

            if (rightAssociative) {
                // scan right-to-left, combine as we go
                int i = outOperators.size() - 1;
                while (i >= 0) {
                    if (outOperators.get(i).priority == priority) {
                        Expression combined = new BinaryExpression(
                                outOperators.get(i),
                                outOperands.get(i),
                                outOperands.get(i + 1));

                        outOperands.set(i, combined);
                        outOperands.remove(i + 1);
                        outOperators.remove(i);
                    }
                    i--;
                }
            } else {
                // scan left-to-right, combine as we go
                int i = 0;
                while (i < outOperators.size()) {
                    if (outOperators.get(i).priority == priority) {
                        Expression combined = new BinaryExpression(
                                outOperators.get(i),
                                outOperands.get(i),
                                outOperands.get(i + 1));

                        outOperands.set(i, combined);
                        outOperands.remove(i + 1);
                        outOperators.remove(i);
                        // don't advance: next operator shifted into position i
                    } else {
                        i++;
                    }
                }
            }
        }

        return outOperands.getFirst();
    }

    private boolean isRightAssociativeLevel(List<BinaryOperator> operators, int priority) {
        for (BinaryOperator op : operators) {
            if (op.priority == priority) {
                return op.rightAssociative;
            }
        }
        return false; // doesn't matter, no operator at this level
    }

    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }

    @Override
    public String toString() {
        return "Equation{" +Expression.print(left) +
                " = " + Expression.print(right) +
                ", symbols=" + Arrays.toString(symbols.values().toArray()) +
                '}';
    }
}