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
     * this = x
     */
    void copy(Vector x);

    //create a new vector object with the same values
    Vector copy();
    /**
     * Sets all elements to zero.
     */
    void clear();
}
