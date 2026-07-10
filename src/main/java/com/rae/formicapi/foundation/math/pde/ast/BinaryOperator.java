package com.rae.formicapi.foundation.math.pde.ast;


import java.util.Optional;

public enum BinaryOperator {

    ADD('+', 0, false),
    SUBTRACT('-', 0, false),
    MULTIPLY('*', 1, false),
    DIVIDE('/', 1, false),
    POWER('^', 2, true);

    public final char representation;
    public final int priority;
    public final boolean rightAssociative;

    BinaryOperator(char representation, int priority, boolean rightAssociative) {
        this.representation = representation;
        this.priority = priority;
        this.rightAssociative = rightAssociative;
    }

    public static Optional<BinaryOperator> parse(char value) {
        for (BinaryOperator op : values()) {
            if (op.representation == value)
                return Optional.of(op);
        }
        return Optional.empty();
    }

    public static boolean isOperatorChar(char c) {
        return parse(c).isPresent();
    }
}