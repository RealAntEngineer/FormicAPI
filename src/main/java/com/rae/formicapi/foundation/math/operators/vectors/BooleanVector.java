package com.rae.formicapi.foundation.math.operators.vectors;


/**
 * A mutable vector of booleans, used primarily as a liveness/inclusion
 * mask (e.g. {@code ParticleSystem}'s alive flags) alongside
 * {@link RealVector} payload fields.
 *
 * <p><b>Fluent chaining.</b> Mutating methods return {@code this}, matching
 * {@link RealVector}'s convention.
 */
public interface BooleanVector extends Vector {

    /**
     * Sets element {@code idx} to {@code value}.
     *
     * @return {@code this}, for chaining
     */
    BooleanVector set(boolean value, int idx);

    boolean get(int idx);

    /**
     * Sets every element to {@code value} in one pass — the boolean
     * counterpart of a uniform scalar op, useful for bulk re-flagging
     * (e.g. marking a freshly compacted range alive again) without a
     * per-index Java loop.
     *
     * @return {@code this}, for chaining
     */
    BooleanVector fill(boolean value, int fromInclusive, int toExclusive);

    /**
     * Gather: {@code this[i] = source[indices[i]]}, for
     * {@code i in [0, indices.size())}. {@code this} must already have at
     * least {@code indices.size()} elements; entries beyond that are left
     * untouched.
     *
     * <p>Safe to call with {@code source == this} — implementations must
     * read every source value before it could be overwritten, since
     * {@code indices} may reorder or duplicate entries. Same contract as
     * {@link RealVector#gather}.
     *
     * @return {@code this}, for chaining
     */
    BooleanVector gather(BooleanVector source, IntegerVector indices);
}