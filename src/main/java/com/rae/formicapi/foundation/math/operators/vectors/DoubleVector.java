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

    double skippedDot(DoubleVector other, IntegerVector unknowIdx);

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

    /**
     * Scattered axpy: this[unknowIdx.get(i)] += alpha * source.get(i), for i in [0, unknowIdx.size()).
     * The scatter counterpart of skippedDot — same index-remap contract, opposite direction
     * (skippedDot gathers into a reduction, this scatters into positions of `this`).
     */
    void scatterAxpy(double alpha, DoubleVector source, IntegerVector unknowIdx);

    void scale(double a);

    void add(double a);

    void add(Vector x);
}
