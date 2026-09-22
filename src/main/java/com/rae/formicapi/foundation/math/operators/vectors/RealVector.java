package com.rae.formicapi.foundation.math.operators.vectors;

/**
 * A mutable, arbitrarily-backed (CPU, GPU, ...) vector of real (float or doubles): the
 * elementwise/linear-algebra primitives that iterative solvers (CG,
 * BiCGSTAB, Newton-Krylov, ...) and the Lagrangian particle layer are built
 * out of.
 *
 * <p><b>Fluent chaining.</b> Every mutating method returns {@code this}, so
 * calls can be composed without intermediate temporaries, e.g.
 * <pre>
 * r.copy(b).subtract(Ax).scale(scaling);
 * </pre>
 * The return value is always the receiver, never a new vector, so chaining
 * never allocates and never changes which object a variable refers to.
 *
 * <p><b>Aliasing.</b> Unless a method says otherwise, it is safe to pass
 * {@code this} as an argument to itself (e.g. {@code v.add(v)} doubles every
 * element). Implementations must honor this by reading each operand before
 * overwriting it, not by assuming distinct backing storage.
 *
 * <p><b>Skipped / scattered operations.</b> Methods prefixed {@code skipped}
 * operate on a reduced index space (e.g. free-variable-only systems, see
 * {@code unknownIdx} in {@code BiCGStab}) without allocating a smaller
 * vector for the reduced system. {@code unknowIdx} maps a compact index
 * {@code i in [0, unknowIdx.size())} to a position in the full-length
 * vector. {@code thisSkip} / {@code sourceSkip} independently control
 * whether {@code this} and the operand are addressed through that remapping
 * (skip = true) or directly by position (skip = false), so one operator can
 * mix full-space and reduced-space vectors depending on which side of the
 * boundary each one lives on.
 */
public interface RealVector extends Vector {

    /** Euclidean norm: {@code sqrt(this . this)}. */
    default double norm() {
        return Math.sqrt(dot(this));
    }

    /**
     * Dot product: {@code this . other}. Both vectors must be the same
     * size.
     */
    double dot(RealVector other);

    /**
     * Reduced-space dot product: equivalent to gathering both operands
     * through {@code unknowIdx} (per {@code thisSkip}/{@code sourceSkip})
     * and dotting the result, without materializing the gathered vectors.
     *
     * @see RealVector class-level docs on skipped/scattered operations
     */
    default double skippedDot(RealVector other, IntegerVector unknowIdx) {
        return skippedDot(other, unknowIdx, true, false);
    }

    double skippedDot(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip);

    /**
     * {@code this = this + a*x} (AXPY).
     *
     * @return {@code this}, for chaining
     */
    RealVector axpy(double a, RealVector x);

    /**
     * Reduced-space AXPY.
     *
     * @return {@code this}, for chaining
     * @see RealVector class-level docs on skipped/scattered operations
     */
    default RealVector skippedAxpy(double alpha, RealVector other, IntegerVector unknowIdx) {
        return skippedAxpy(alpha, other, unknowIdx, true, false);
    }

    /**
     * Scattered AXPY: {@code this[unknowIdx[i]] += alpha * source[i]}, for
     * {@code i in [0, unknowIdx.size())}. The scatter counterpart of
     * {@link #skippedDot}: that gathers into a reduction, this scatters into
     * positions of {@code this}.
     *
     * @return {@code this}, for chaining
     */
    RealVector skippedAxpy(double alpha, RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean sourceSkip);

    /**
     * {@code this = a * this} (uniform scalar scale).
     *
     * @return {@code this}, for chaining
     */
    RealVector scale(double a);

    /**
     * Elementwise scale: {@code this[i] *= x[i]}.
     *
     * @return {@code this}, for chaining
     */
    RealVector scale(RealVector x);

    /**
     * Reduced-space elementwise scale.
     *
     * @return {@code this}, for chaining
     * @see RealVector class-level docs on skipped/scattered operations
     */
    default RealVector skippedScale(RealVector other, IntegerVector unknowIdx) {
        return skippedScale(other, unknowIdx, true, false);
    }

    RealVector skippedScale(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    /**
     * Elementwise division: {@code this[i] /= x[i]}. Behavior on a zero
     * divisor is backend-defined (typically IEEE {@code Infinity}/
     * {@code NaN}); callers relying on masked/skipped regions should ensure
     * those entries never reach a zero divisor.
     *
     * @return {@code this}, for chaining
     */
    RealVector divide(RealVector x);

    /**
     * Reduced-space elementwise division.
     *
     * @return {@code this}, for chaining
     * @see RealVector class-level docs on skipped/scattered operations
     */
    default RealVector skippedDivide(RealVector other, IntegerVector unknowIdx) {
        return skippedDivide(other, unknowIdx, true, false);
    }

    RealVector skippedDivide(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    /**
     * {@code this[i] += a}, for every element (uniform scalar add).
     *
     * @return {@code this}, for chaining
     */
    RealVector add(double a);

    /**
     * Elementwise add: {@code this[i] += x[i]}.
     *
     * @return {@code this}, for chaining
     */
    RealVector add(RealVector x);

    /**
     * Reduced-space elementwise add.
     *
     * @return {@code this}, for chaining
     * @see RealVector class-level docs on skipped/scattered operations
     */
    default RealVector skippedAdd(RealVector other, IntegerVector unknowIdx) {
        return skippedAdd(other, unknowIdx, true, false);
    }

    RealVector skippedAdd(RealVector other, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    /**
     * Elementwise subtract: {@code this[i] -= x[i]}.
     *
     * @return {@code this}, for chaining
     */
    RealVector subtract(RealVector x);

    /**
     * Reduced-space elementwise subtract.
     *
     * @return {@code this}, for chaining
     * @see RealVector class-level docs on skipped/scattered operations
     */
    default RealVector skippedSubtract(RealVector other, IntegerVector unknowIdx) {
        return skippedSubtract(other, unknowIdx, true, false);
    }

    RealVector skippedSubtract(RealVector x, IntegerVector unknowIdx, boolean thisSkip, boolean otherSkip);

    /**
     * Gather: {@code this[i] = source[indices[i]]}, for
     * {@code i in [0, indices.size())}. {@code this} must already have at
     * least {@code indices.size()} elements; entries beyond that are left
     * untouched.
     *
     * <p>Safe to call with {@code source == this} — implementations must
     * read every source value before it could be overwritten (e.g. via an
     * internal temporary), since {@code indices} may reorder or duplicate
     * entries.
     *
     * @return {@code this}, for chaining
     */
    RealVector gather(RealVector source, IntegerVector indices);
}