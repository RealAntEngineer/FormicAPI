package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Optional;

public enum UnaryOperators {

    NEGATE("-") {
    },
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
    D2DT("d2dt2", true, false),
    DDX("ddx", false, true),
    D2DX2("d2dx2", false, true),
    DDY("ddy", false, true),
    D2DY2("d2dy2", false, true),
    DDZ("ddz", false, true),
    D2DZ2("d2dz2", false, true),
    D2DXDY("d2dxdy", false, true),
    D2DXDZ("d2dxdz", false, true) ,
    D2DYDZ("d2dydz", false, true);

    private final String representation;
    private final boolean appliesTimeDifferentiation;
    private final boolean appliesSpaceDifferentiation;

    UnaryOperators(String representation, boolean appliesTimeDifferentiation, boolean appliesSpaceDifferentiation) {
        this.representation = representation;
        this.appliesTimeDifferentiation = appliesTimeDifferentiation;
        this.appliesSpaceDifferentiation = appliesSpaceDifferentiation;
    }

    UnaryOperators(String representation) {
        this(representation, false, false);
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

    public boolean appliesTimeDifferentiation() {
        return appliesTimeDifferentiation;
    }

    public boolean appliesSpaceDifferentiation() {
        return appliesSpaceDifferentiation;
    }


    public FieldType resultType(FieldType operand){
        return operand;
    }

    /**
     * Whether this operator distributes over addition/subtraction: op(a+b) = op(a) + op(b).
     */
    public boolean isLinear(){
        return true;
    }
}