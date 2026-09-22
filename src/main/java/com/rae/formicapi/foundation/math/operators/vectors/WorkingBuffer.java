package com.rae.formicapi.foundation.math.operators.vectors;


import java.util.Objects;
import java.util.function.Supplier;

/**
 * Reusable temporary vector storage for iterative solvers.
 *
 * <p>Solvers request temporary vectors from this buffer instead of
 * allocating new objects during iterations. Implementations may reuse
 * memory between solves and may allocate vectors on CPU or GPU backends.
 */
public final class WorkingBuffer<T extends Vector> {

    private final T[] vectors;

    /**
     * Creates a working buffer.
     *
     * @param vectors reusable vectors
     */
    public WorkingBuffer(T[] vectors) {
        this.vectors = vectors;
    }

    @SuppressWarnings("unchecked")//we know it's safe
    public WorkingBuffer(int number, Supplier<T> defaultVec) {
        this.vectors = (T[]) new Vector[number];
        for (int i = 0; i < number; i++) {
            this.vectors[i] = defaultVec.get();
        }
    }

    /**
     * Returns a temporary vector by index.
     *
     * @param index buffer slot
     * @return reusable vector
     */
    public T get(int index) {
        return Objects.requireNonNull(vectors[index]);
    }


    public int vectorNumber(){
        return vectors.length;
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