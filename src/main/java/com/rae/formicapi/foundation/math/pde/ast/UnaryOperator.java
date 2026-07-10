package com.rae.formicapi.foundation.math.pde.ast;

import java.util.Optional;

public enum UnaryOperator {

    NEGATE("-"),
    GRAD("grad"),
    DIV("div"),
    LAPLACIAN("lap");

    private final String representation;


    UnaryOperator(String representation) {
        this.representation = representation;
    }


    public String representation() {
        return representation;
    }


    public static Optional<UnaryOperator> parse(String value) {

        for (UnaryOperator op : values()) {

            if (op.representation.equals(value))
                return Optional.of(op);
        }

        return Optional.empty();
    }
}
