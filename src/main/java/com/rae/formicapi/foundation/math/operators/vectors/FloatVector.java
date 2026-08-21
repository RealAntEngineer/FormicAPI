package com.rae.formicapi.foundation.math.operators.vectors;

public interface FloatVector extends Vector {
    /**
     * Euclidean norm.
     */
    default float norm() {
        return (float) Math.sqrt(dot(this));
    }

    /**
     * Computes the dot product:
     *
     * <pre>
     * this · other
     * </pre>
     */
    float dot(FloatVector other);

    float skippedDot(FloatVector other, IntegerVector unknowIdx);

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
    void axpy(float a, FloatVector x);

    /**
     * Scattered axpy: this[idx.get(i)] += alpha * source.get(i), for i in [0, idx.size()).
     * The scatter counterpart of skippedDot — same index-remap contract, opposite direction
     * (skippedDot gathers into a reduction, this scatters into positions of `this`).
     */
    void scatterAxpy(float alpha, FloatVector source, IntegerVector idx);

    void scale(float a);

    void add(float a);

    void add(FloatVector x);
}
