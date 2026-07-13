package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Optional;

public enum UnaryOperators {

    NEGATE("-") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    GRAD("grad") {
        @Override
        public FieldType resultType(FieldType operand) {
            return switch (operand) {
                case SCALAR -> FieldType.VECTOR;
                case VECTOR -> FieldType.TENSOR;
                case TENSOR -> throw new FieldType.TypeMismatchException("grad() is not defined for TENSOR fields");
            };
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    DIV("div") {
        @Override
        public FieldType resultType(FieldType operand) {
            return switch (operand) {
                case VECTOR -> FieldType.SCALAR;
                case TENSOR -> FieldType.VECTOR;
                case SCALAR ->
                        throw new FieldType.TypeMismatchException("div() requires a VECTOR or TENSOR field, got SCALAR");
            };
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    LAPLACIAN("lap") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    DDT("ddt") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DT("d2dt2") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    DDX("ddx") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DX2("d2dx2") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    DDY("ddy") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DY2("d2dy2") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    DDZ("ddz") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DZ2("d2dz2") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DXDY("d2dxdy") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DXDZ("d2dxdz") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    },
    D2DYDZ("d2dydz") {
        @Override
        public FieldType resultType(FieldType operand) {
            return operand;
        }

        @Override
        public boolean isLinear() {
            return true;
        }
    };

    private final String representation;

    UnaryOperators(String representation) {
        this.representation = representation;
    }

    public static Optional<UnaryOperators> parse(String value) {
        for (UnaryOperators op : values()) {
            if (op.representation.equals(value)) return Optional.of(op);
        }
        return Optional.empty();
    }

    public String representation() {
        return representation;
    }

    public abstract FieldType resultType(FieldType operand);

    /**
     * Whether this operator distributes over addition/subtraction: op(a+b) = op(a) + op(b).
     */
    public abstract boolean isLinear();
}