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
     * logical size, not the size of the underlying array (lazy allocation)
     * @return the number of elements
     */
    int size();

    /**
     * Changes the size of this vector.
     *
     * <p>If the vector grows, new elements are initialized to zero.
     * Existing values are preserved when possible.
     *
     * @param size new vector length
     * @return self
     */
    Vector resize(int size);

    /**
     * this = x
     * @return self
     */
    Vector copy(Vector x);

    /**
     * create a new vector object with the same values
     * @return self
     */
    Vector copy();

    /**
     * Sets all elements to zero.
     * @return self
     */
    Vector clear();

    /**
     * internal copy of section from [srcStart, srcStart + count] to [dstStart, dstStart + count]
     * @param srcStart the source index
     * @param dstStart the destination index
     * @param count the size of the section to copy
     * @return self
     */
    Vector internalCopy(int srcStart, int dstStart, int count);
}
