package com.rae.formicapi.foundation.math.operators.physics;

/**
 * Compact bitmask describing which {@link Variable}s a physics operator treats as
 * unknowns, i.e. which rows/columns of the global state vector it owns.
 *
 * <p>Examples:
 * <ul>
 *     <li>URANS: {@code VariableMask.of(U, V, W, PRESSURE)} — {@code 01111}</li>
 *     <li>Temperature: {@code VariableMask.of(TEMPERATURE)} — {@code 10000}</li>
 * </ul>
 *
 * <p>This mask is fixed for the lifetime of a {@link com.rae.formicapi.foundation.math
 * .operators.linear.BlockSparseMatrix} — it is the same for every block in the matrix.
 * It is what turns the "variable number of active variables per row" problem into a
 * fixed, known-at-construction {@code blockSize}, so no branching is needed inside the
 * multiply loop.
 *
 * <p>Immutable. Currently small enough (5 variables) to be represented as a single int.
 */
public final class VariableMask {

    private final int bits;

    private VariableMask(int bits) {
        this.bits = bits;
    }

    public static VariableMask of(Variable... variables) {
        int bits = 0;
        for (Variable v : variables) {
            bits |= v.bit();
        }
        return new VariableMask(bits);
    }

    public boolean contains(Variable variable) {
        return (bits & variable.bit()) != 0;
    }

    /** Number of active variables in this mask == block size for a matrix using it. */
    public int count() {
        return Integer.bitCount(bits);
    }

    /**
     * Active variables in ascending ordinal order.
     *
     * <p>This order defines the local row/column layout inside every block of a matrix
     * built with this mask — e.g. for URANS, local slot 0..3 = U, V, W, PRESSURE.
     */
    public Variable[] indices() {
        Variable[] out = new Variable[count()];
        int k = 0;
        for (Variable v : Variable.values()) {
            if (contains(v)) {
                out[k++] = v;
            }
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof VariableMask && ((VariableMask) o).bits == bits;
    }

    @Override
    public int hashCode() {
        return bits;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Variable v : Variable.values()) {
            sb.append(contains(v) ? '1' : '0');
        }
        return sb.reverse().toString();
    }
}