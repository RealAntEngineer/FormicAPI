package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Optional;

public enum UnaryOperators {

    NEGATE("-") {
        @Override public FieldType resultType(FieldType operand) { return operand; }
    },
    GRAD("grad") {
        @Override public FieldType resultType(FieldType operand) {
            return switch (operand) {
                case SCALAR -> FieldType.VECTOR;
                case VECTOR -> FieldType.TENSOR;
                case TENSOR -> throw new FieldType.TypeMismatchException("grad() is not defined for TENSOR fields");
            };
        }
    },
    DIV("div") {
        @Override public FieldType resultType(FieldType operand) {
            return switch (operand) {
                case VECTOR -> FieldType.SCALAR;
                case TENSOR -> FieldType.VECTOR;
                case SCALAR -> throw new FieldType.TypeMismatchException("div() requires a VECTOR or TENSOR field, got SCALAR");
            };
        }
    },
    LAPLACIAN("lap") {
        @Override public FieldType resultType(FieldType operand) { return operand; } // componentwise
    },
    DDT("ddt") {
        @Override public FieldType resultType(FieldType operand) { return operand; }
    },
    D2DT("d2dt") {
        @Override public FieldType resultType(FieldType operand) { return operand; }
    };

    private final String representation;

    UnaryOperators(String representation) {
        this.representation = representation;
    }

    public String representation() {
        return representation;
    }

    public abstract FieldType resultType(FieldType operand);

    public static Optional<UnaryOperators> parse(String value) {
        for (UnaryOperators op : values()) {
            if (op.representation.equals(value)) return Optional.of(op);
        }
        return Optional.empty();
    }
}