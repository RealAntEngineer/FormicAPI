package com.rae.formicapi.foundation.math.pde.ast;


import com.rae.formicapi.foundation.math.pde.FieldType;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public enum BinaryOperators {

    ADD('+', 0) {
        @Override public FieldType resultType(FieldType left, FieldType right) { requireSameType(this, left, right); return left; }
        @Override protected Expression combine(Expression left, Expression right) { return componentwiseOrPlain(this, left, right); }
    },
    SUBTRACT('-', 0) {
        @Override public FieldType resultType(FieldType left, FieldType right) { requireSameType(this, left, right); return left; }
        @Override protected Expression combine(Expression left, Expression right) { return componentwiseOrPlain(this, left, right); }
    },
    MULTIPLY('*', 1) {
        @Override public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.SCALAR) return right;
            if (right == FieldType.SCALAR) return left;
            throw new FieldType.TypeMismatchException("Cannot multiply " + left + " by " + right + " with '*' (need a SCALAR operand; use dot/cross for " + left + "x" + right + ")");
        }
        @Override protected Expression combine(Expression left, Expression right) { return scalarProductOrPlain(this, left, right); }
    },
    DOT_PRODUCT('.', 1) {
        @Override public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.VECTOR && right == FieldType.VECTOR) return FieldType.SCALAR;
            throw new FieldType.TypeMismatchException("Cannot dot-multiply " + left + " by " + right + " with '.' (both operands must be VECTOR)");
        }
        @Override protected Expression combine(Expression left, Expression right) {
            int dimensions = left.dimensions();
            Expression sum = null;
            for (int i = 0; i < dimensions; i++) {
                Expression product = new BinaryExpression(MULTIPLY, left.componentAt(i), right.componentAt(i));
                sum = (sum == null) ? product : new BinaryExpression(ADD, sum, product);
            }
            return sum;
        }
    },
    CROSS_PRODUCT('@', 1) {
        @Override public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.VECTOR && right == FieldType.VECTOR) return FieldType.VECTOR;
            throw new FieldType.TypeMismatchException("Cannot cross-multiply " + left + " by " + right + " with '@' (both operands must be VECTOR)");
        }
        @Override
        protected Expression combine(Expression left, Expression right) {
            if (left.dimensions() != 3) {
                throw new UnsupportedOperationException("Cross product is only defined for 3 dimensions, got " + left.dimensions());
            }
            return new VectorExpression(List.of(
                    sub(mul(left.componentAt(1), right.componentAt(2)), mul(left.componentAt(2), right.componentAt(1))),
                    sub(mul(left.componentAt(2), right.componentAt(0)), mul(left.componentAt(0), right.componentAt(2))),
                    sub(mul(left.componentAt(0), right.componentAt(1)), mul(left.componentAt(1), right.componentAt(0)))
            ));
        }
    },
    OUTER_PRODUCT('#', 1) {
        @Override public FieldType resultType(FieldType left, FieldType right) {
            if (left == FieldType.VECTOR && right == FieldType.VECTOR) return FieldType.MATRIX;
            throw new FieldType.TypeMismatchException("Cannot outer-multiply " + left + " by " + right + " with '#' (both operands must be VECTOR)");
        }
        @Override protected Expression combine(Expression left, Expression right) {
            int dimensions = left.dimensions();
            List<List<Expression>> rows = new ArrayList<>();
            for (int i = 0; i < dimensions; i++) {
                List<Expression> row = new ArrayList<>();
                for (int j = 0; j < dimensions; j++) row.add(mul(left.componentAt(i), right.componentAt(j)));
                rows.add(row);
            }
            return new MatrixExpression(rows);
        }
    },
    DIVIDE('/', 1) {
        @Override public FieldType resultType(FieldType left, FieldType right) {
            if (right != FieldType.SCALAR) throw new FieldType.TypeMismatchException("Cannot divide by a " + right + " (division is only defined by SCALAR)");
            return left;
        }
        @Override protected Expression combine(Expression left, Expression right) { return scalarProductOrPlain(this, left, right); }
    },
    POWER('^', 2, true) {
        @Override public FieldType resultType(FieldType left, FieldType right) { requireScalar(this, left); requireScalar(this, right); return FieldType.SCALAR; }
        @Override protected Expression combine(Expression left, Expression right) { return new BinaryExpression(this, left, right); }
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

    private static Expression componentwiseOrPlain(BinaryOperators op, Expression left, Expression right) {
        FieldType type = left.resultType();
        if (type == FieldType.VECTOR) {
            List<Expression> components = new ArrayList<>();
            for (int i = 0; i < left.dimensions(); i++)
                components.add(new BinaryExpression(op, left.componentAt(i), right.componentAt(i)));
            return new VectorExpression(components);
        }
        if (type == FieldType.MATRIX) {
            List<List<Expression>> rows = new ArrayList<>();
            for (int i = 0; i < left.dimensions(); i++) {
                List<Expression> row = new ArrayList<>();
                for (int j = 0; j < left.dimensions(); j++)
                    row.add(new BinaryExpression(op, left.componentAt(i, j), right.componentAt(i, j)));
                rows.add(row);
            }
            return new MatrixExpression(rows);
        }
        return new BinaryExpression(op, left, right); // SCALAR
    }

    private static Expression scalarProductOrPlain(BinaryOperators op, Expression left, Expression right) {
        boolean leftIsScalar  = left.resultType() == FieldType.SCALAR;
        boolean rightIsScalar = right.resultType() == FieldType.SCALAR;
        if (leftIsScalar && rightIsScalar) return new BinaryExpression(op, left, right);

        Expression scalar     = leftIsScalar ? left : right;
        Expression vectorLike = leftIsScalar ? right : left;

        if (vectorLike.resultType() == FieldType.VECTOR) {
            List<Expression> components = new ArrayList<>();
            for (int i = 0; i < vectorLike.dimensions(); i++) {
                Expression c = vectorLike.componentAt(i);
                components.add(leftIsScalar ? new BinaryExpression(op, scalar, c) : new BinaryExpression(op, c, scalar));
            }
            return new VectorExpression(components);
        }
        List<List<Expression>> rows = new ArrayList<>();
        for (int i = 0; i < vectorLike.dimensions(); i++) {
            List<Expression> row = new ArrayList<>();
            for (int j = 0; j < vectorLike.dimensions(); j++) {
                Expression c = vectorLike.componentAt(i, j);
                row.add(leftIsScalar ? new BinaryExpression(op, scalar, c) : new BinaryExpression(op, c, scalar));
            }
            rows.add(row);
        }
        return new MatrixExpression(rows);
    }

    private static Expression mul(Expression a, Expression b) {
        return new BinaryExpression(MULTIPLY, a, b);
    }

    private static Expression sub(Expression a, Expression b) {
        return new BinaryExpression(SUBTRACT, a, b);
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

    /** Template method: check distribution first, otherwise delegate to this operator's own combine(). */
    public final Expression expand(Expression left, Expression right) {
        if (left instanceof BinaryExpression lb && distributesOver(lb.getOperator(), true)) {
            return lb.getOperator().expand(this.expand(lb.getLeft(), right), this.expand(lb.getRight(), right));
        }
        if (right instanceof BinaryExpression rb && distributesOver(rb.getOperator(), false)) {
            return rb.getOperator().expand(this.expand(left, rb.getLeft()), this.expand(left, rb.getRight()));
        }
        return combine(left, right);
    }

    /** Combines two already-fully-expanded, non-further-distributable operands into a result. */
    protected abstract Expression combine(Expression left, Expression right);
}