package com.rae.formicapi.foundation.math.pde.ast;


import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.Optional;

public enum BinaryOperators {

    ADD('+', 0) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            requireSameType(this, left, right);
            return left;
        }
    },
    SUBTRACT('-', 0) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            requireSameType(this, left, right);
            return left;
        }
    },
    MULTIPLY('*', 1) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.SCALAR) return right;
            if (right == FieldType.SCALAR) return left;
            throw new FieldType.TypeMismatchException(
                    "Cannot multiply " + left + " by " + right + " with '*' (need a SCALAR operand; use dot/cross for " + left + "x" + right + ")");
        }
    },
    DOT_PRODUCT('.', 1) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.VECTOR && right == FieldType.VECTOR) return FieldType.SCALAR;
            throw new FieldType.TypeMismatchException(
                    "Cannot dot-multiply " + left + " by " + right + " with '.' (both operands must be VECTOR)");
        }
    },
    CROSS_PRODUCT('@', 1) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.VECTOR && right == FieldType.VECTOR) return FieldType.VECTOR;
            throw new FieldType.TypeMismatchException(
                    "Cannot cross-multiply " + left + " by " + right + " with '@' (both operands must be VECTOR)");
        }
    },
    OUTER_PRODUCT('#', 1) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.VECTOR && right == FieldType.VECTOR) return FieldType.TENSOR;
            throw new FieldType.TypeMismatchException(
                    "Cannot outer-multiply " + left + " by " + right + " with '#' (both operands must be VECTOR)");
        }
    },
    DIVIDE('/', 1) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            if (right != FieldType.SCALAR) {
                throw new FieldType.TypeMismatchException("Cannot divide by a " + right + " (division is only defined by SCALAR)");
            }
            return left;
        }
    },
    POWER('^', 2, true) {
        @Override
        public FieldType resultType(FieldType left, FieldType right) {
            requireScalar(this, left);
            requireScalar(this, right);
            return FieldType.SCALAR;
        }
    };

    public final char    representation;
    public final int     priority;
    public final boolean rightAssociative;

    BinaryOperators(char representation, int priority) {
        this(representation, priority, false);
    }

    BinaryOperators(char representation, int priority, boolean rightAssociative) {
        this.representation = representation;
        this.priority = priority;
        this.rightAssociative = rightAssociative;
    }

    protected static void requireSameType(BinaryOperators op, FieldType left, FieldType right) {
        if (left != right) {
            throw new FieldType.TypeMismatchException("Operator '" + op.representation + "' requires matching operand types, got " + left + " and " + right);
        }
    }

    protected static void requireScalar(BinaryOperators op, FieldType type) {
        if (type != FieldType.SCALAR) {
            throw new FieldType.TypeMismatchException("Operator '" + op.representation + "' requires a SCALAR operand, got " + type);
        }
    }

    public static boolean isOperatorChar(char c) {
        return parse(c).isPresent();
    }

    public static Optional<BinaryOperators> parse(char value) {
        for (BinaryOperators op : values()) {
            if (op.representation == value) return Optional.of(op);
        }
        return Optional.empty();
    }

    public abstract FieldType resultType(FieldType left, FieldType right);

    /**
     * Whether this operator distributes over `inner` from the given side: (this-left {inner} this-right).
     */
    public boolean distributesOver(BinaryOperators inner, boolean asLeftOperand) {
        if (this == MULTIPLY || this == DOT_PRODUCT || this == CROSS_PRODUCT || this == OUTER_PRODUCT) {
            return inner == ADD || inner == SUBTRACT; // bilinear: distributes symmetrically, both sides
        }
        if (this == DIVIDE) {
            return asLeftOperand && (inner == ADD || inner == SUBTRACT);
        }
        return false;
    }
}