package com.rae.formicapi.foundation.math.operators.vectors;

import java.util.Arrays;

/**
 * Reusable temporary vector storage for iterative solvers.
 *
 * <p>Solvers request temporary vectors from this buffer instead of
 * allocating new objects during iterations. Implementations may reuse
 * memory between solves and may allocate vectors on CPU or GPU backends.
 */
public final class WorkingBuffer {

    private final Vector[] vectors;


    /**
     * Creates a working buffer.
     *
     * @param vectors reusable vectors
     */
    public WorkingBuffer(Vector[] vectors) {
        this.vectors = vectors;
    }

    public WorkingBuffer(int number, Vector defaultVec) {
        this.vectors = new Vector[number];
        Arrays.fill(vectors,defaultVec.copy());
    }


    /**
     * Returns a temporary vector by index.
     *
     * @param index buffer slot
     * @return reusable vector
     */
    public Vector get(int index) {
        return vectors[index];
    }


    /**
     * Ensures all vectors have the requested size.
     */
    public void resize(int size) {
        for (Vector vector : vectors) {
            vector.resize(size);
        }
    }


    /**
     * Clears all temporary vectors.
     */
    public void clear() {
        for (Vector vector : vectors) {
            vector.clear();
        }
    }
}
