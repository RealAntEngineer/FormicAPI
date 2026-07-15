package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import org.checkerframework.common.value.qual.IntRange;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public abstract class Expression {

    public static final int MAX_DEPTH = 100;
    private final       int dimensions;
    private final       int depth;


    Expression(@IntRange(from = 1) int dimensions) {
        this(dimensions, 0);
    }

    Expression(@IntRange(from = 1) int dimensions, @IntRange(from = 0) int depth) {
        this.dimensions = dimensions;
        this.depth = depth;
    }

    public static Expression parseExpression(String text, Map<String, SymbolBinding> symbols, @IntRange(from = 0) int depth, @IntRange(from = 1) int dimensions) {

        if (depth > MAX_DEPTH)
            throw new RuntimeException("Max depth reached");

        text = text.trim();

        List<Expression>      operands  = new ArrayList<>();
        List<BinaryOperators> operators = new ArrayList<>();

        String remaining = text;

        while (true) {
            remaining = remaining.trim();

            Expression operand;

            if (remaining.startsWith("(")) {
                String[] splited = groupByParenthesis(remaining);
                operand = parseExpression(splited[0], symbols, depth + 1, dimensions);
                remaining = splited[1];

            } else {
                String identifier = remaining;
                String rest       = "";

                for (int i = 0; i < remaining.length(); i++) {
                    char c = remaining.charAt(i);
                    if (c == ' ') {
                        identifier = remaining.substring(0, i);
                        rest = remaining.substring(i + 1);
                        break;
                    } else if (c == '(' || BinaryOperators.isOperatorChar(c)) {
                        identifier = remaining.substring(0, i);
                        rest = remaining.substring(i);
                        break;
                    }
                }

                if (identifier.isEmpty())
                    throw new RuntimeException("Expression is empty");

                if (Character.isDigit(identifier.charAt(0))) {
                    try {
                        operand = new ConstantExpression(Double.parseDouble(identifier), dimensions);
                    } catch (NumberFormatException e) {
                        throw new RuntimeException("Failed to parse value " + identifier + " in expression " + text, e);
                    }

                } else if (symbols.containsKey(identifier)) {
                    operand = new VariableExpression(symbols.get(identifier), dimensions);

                } else {
                    Optional<UnaryOperators> optionalUnary = UnaryOperators.parse(identifier);
                    if (optionalUnary.isPresent()) {
                        String[] splited = groupByParenthesis(rest);
                        operand = new UnaryExpression(optionalUnary.get(), parseExpression(splited[0], symbols, depth + 1, dimensions));
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
                String[]   splited = groupByParenthesis(remaining);
                Expression right   = parseExpression(splited[0], symbols, depth + 1, dimensions);
                operand = new BinaryExpression(BinaryOperators.MULTIPLY, operand, right);
                remaining = splited[1].trim();
            }

            operands.add(operand);

            if (remaining.isEmpty())
                break;

            Optional<BinaryOperators> optionalBinary = BinaryOperators.parse(remaining.charAt(0));
            if (optionalBinary.isEmpty())
                throw new RuntimeException("Unable to parse binary operator : " + remaining.charAt(0));

            operators.add(optionalBinary.get());
            remaining = remaining.substring(1);
        }

        return buildTreeByPrecedence(operands, operators);
    }

    private static Expression buildTreeByPrecedence(List<Expression> operands, List<BinaryOperators> operators) {
        List<Expression>      outOperands  = new ArrayList<>(operands);
        List<BinaryOperators> outOperators = new ArrayList<>(operators);

        int maxPriority = 0;
        for (BinaryOperators op : outOperators) {
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

    private static boolean isRightAssociativeLevel(List<BinaryOperators> operators, int priority) {
        for (BinaryOperators op : operators) {
            if (op.priority == priority) {
                return op.rightMerging;
            }
        }
        return false; // doesn't matter, no operator at this level
    }

    public final int dimensions() {
        return dimensions;
    }

    public Expression componentAt(int axis) {
        return ComponentExpression.ofVector(this, axis);
    }

    public Expression componentAt(int row, int col) {
        return ComponentExpression.ofMatrix(this, row, col);
    }

    public abstract FieldType resultType();

    public abstract String prettyPrint();

    public abstract Expression expand();

    public abstract boolean isTimeDifferentiable();

    public abstract boolean isSpaceDifferentiable();

    /**
     * do we appear at the right of expression
     */
    public boolean appearAfter(Expression expression) {
        return this.appearanceOrder() > expression.appearanceOrder() ||
                this.appearanceOrder() == expression.appearanceOrder() && this.getDepth() > expression.getDepth();
    }

    /**
     * define a priority for binary operators, the highest numer will be placed last
     */
    public int appearanceOrder() {
        return 0;
    }

    public int getDepth() {
        return depth;
    }
}