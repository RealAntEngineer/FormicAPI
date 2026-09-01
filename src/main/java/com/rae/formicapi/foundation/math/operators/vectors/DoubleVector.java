package com.rae.formicapi.foundation.math.operators.vectors;

public interface DoubleVector extends Vector {
    /**
     * Euclidean norm.
     */
    default double norm() {
        return Math.sqrt(dot(this));
    }

    /**
     * Computes the dot product:
     *
     * <pre>
     * this · other
     * </pre>
     */
    double dot(DoubleVector other);

    default double skippedDot(DoubleVector other, IntegerVector unknowIdx){
        return skippedDot(other, unknowIdx, true, false);
    }
    double skippedDot(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip);

    /**
     * Performs:
     *
     * <pre>
     * this = this + a*x
     * </pre>
     *
     * @param a scalar multiplier
     * @param x vector to add
     */
    void axpy(double a, DoubleVector x);

    default void skippedAxpy(double alpha, DoubleVector other, IntegerVector unknowIdx) {
        skippedAxpy(alpha, other, unknowIdx, true, false);
    }

    /**
     * Scattered axpy: this[unknowIdx.get(i)] += alpha * source.get(i), for i in [0, unknowIdx.size()).
     * The scatter counterpart of skippedDot — same index-remap contract, opposite direction
     * (skippedDot gathers into a reduction, this scatters into positions of `this`).
     */
    void skippedAxpy(double alpha, DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip);

    void scale(double a);

    void scale(DoubleVector x);

    default void skippedScale(DoubleVector other, IntegerVector unknowIdx) {
        skippedAdd(other, unknowIdx, true, false);
    }

    void skippedScale(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    void divide(DoubleVector x);

    default void skippedDivide(DoubleVector other, IntegerVector unknowIdx) {
        skippedAdd(other, unknowIdx, true, false);
    }

    void skippedDivide(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    void add(double a);

    void add(DoubleVector x);

    default void skippedAdd(DoubleVector other, IntegerVector unknowIdx) {
        skippedAdd(other, unknowIdx, true, false);
    }

    void skippedAdd(DoubleVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    void subtract(DoubleVector x);

    default void skippedSubtract(DoubleVector other, IntegerVector unknowIdx) {
        skippedAdd(other, unknowIdx, true, false);
    }

    void skippedSubtract(DoubleVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);


    //TODO add fluent chain version ?
}
