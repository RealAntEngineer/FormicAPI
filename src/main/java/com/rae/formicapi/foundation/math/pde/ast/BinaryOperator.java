package com.rae.formicapi.foundation.math.pde.ast;


import java.util.Optional;

public enum BinaryOperator {

    ADD("+", 0),
    SUBTRACT("-", 0),
    MULTIPLY("*", 1),
    DIVIDE("/", 1),
    POWER("^", 2);

    public final String representation;
    public final int priority;

    BinaryOperator(String representation, int priority) {
        this.representation = representation;
        this.priority = priority;
    }

    public static Optional<BinaryOperator> parse(String value) {

        for (BinaryOperator op : values()) {

            if(op.representation.equals(value))
                return Optional.of(op);
        }

        return Optional.empty();
    }
}