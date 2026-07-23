package com.rae.formicapi.foundation.math.operators.vectors;

/**
 * Represents a mutable mathematical vector.
 *
 * <p>Vectors are used by iterative solvers and linear algebra operators.
 * Implementations may store data in host memory, device memory, or another
 * optimized representation.
 */
public interface Vector {

    /**
     * Returns the number of elements.
     */
    int size();

    /**
     * Changes the size of this vector.
     *
     * <p>If the vector grows, new elements are initialized to zero.
     * Existing values are preserved when possible.
     *
     * @param size new vector length
     */
    void resize(int size);

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
    double dot(Vector other);

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
    void axpy(double a, Vector x);

    void scale(double a);

    void add(double a);

    void add(Vector x);

    /**
     * this = x
     */
    void copy(Vector x);

    /**
     * Sets all elements to zero.
     */
    void clear();
}
