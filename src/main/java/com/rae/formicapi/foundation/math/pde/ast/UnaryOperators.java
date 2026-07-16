package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Optional;

public enum UnaryOperators {
    //TODO remove negate in favor of subtract with no left member. (right now subtract doesn't support it, it should be better to fail only on missing right member)
    NEGATE("-", false, false),
    GRAD("grad", false, true) {
        @Override
        public FieldType resultType(FieldType operand) {
            return switch (operand) {
                case SCALAR -> FieldType.VECTOR;
                case VECTOR -> FieldType.MATRIX;
                case MATRIX -> throw new FieldType.TypeMismatchException("grad() is not defined for TENSOR fields");
            };
        }

    },
    DIV("div", false, true) {
        @Override
        public FieldType resultType(FieldType operand) {
            return switch (operand) {
                case VECTOR -> FieldType.SCALAR;
                case MATRIX -> FieldType.VECTOR;
                case SCALAR ->
                        throw new FieldType.TypeMismatchException("div() requires a VECTOR or TENSOR field, got SCALAR");
            };
        }

    },
    LAPLACIAN("lap", false, true),
    DDT("ddt", true, false),
    D2DT2("d2dt2", true, false),
    DDX("ddx", false, true),
    D2DX2("d2dx2", false, true),
    DDY("ddy", false, true),
    D2DY2("d2dy2", false, true),
    DDZ("ddz", false, true),
    D2DZ2("d2dz2", false, true),
    D2DXDY("d2dxdy", false, true),
    D2DXDZ("d2dxdz", false, true),
    D2DYDZ("d2dydz", false, true),
    

    ;

    public final String  representation;
    public final boolean appliesTimeDifferentiation;
    public final boolean appliesSpaceDifferentiation;

    UnaryOperators(String representation, boolean appliesTimeDifferentiation, boolean appliesSpaceDifferentiation) {
        this.representation = representation;
        this.appliesTimeDifferentiation = appliesTimeDifferentiation;
        this.appliesSpaceDifferentiation = appliesSpaceDifferentiation;
    }

    public static Optional<UnaryOperators> parse(String value) {
        for (UnaryOperators op : values()) {
            if (op.representation.equals(value)) return Optional.of(op);
        }
        return Optional.empty();
    }

    public FieldType resultType(FieldType operand) {
        return operand;
    }

    /**
     * Whether this operator distributes over addition/subtraction: op(a+b) = op(a) + op(b).
     */
    @SuppressWarnings("SameReturnValue")
    public boolean isLinear() {
        return true;
    }
}