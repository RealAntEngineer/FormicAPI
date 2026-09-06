package com.rae.formicapi.foundation.math.operators.vectors;

/**
 * Which norm {@link DoubleVector#norm()} computes.
 *
 * <p>{@code L2} is {@code sqrt(dot(x,x))} -- always an {@code O(n)} reduction,
 * no way around it. {@code L_INFINITY} is {@code max(|x_i|)}; on
 * {@link com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector}
 * this is {@code O(1)} whenever the vector's cached min/max are still valid
 * (i.e. the last mutation was a full-vector op like {@code add}/{@code subtract}/
 * {@code axpy}/{@code scale}/{@code divide}, not a single {@code set} or a
 * masked {@code skippedXxx} call) -- see {@code CpuDoubleVector}'s stats-cache
 * javadoc for exactly which ops keep it valid.
 *
 * <p>Note {@code ||x||_inf <= ||x||_2} always, so switching a convergence
 * check from {@code L2} to {@code L_INFINITY} at the <em>same</em> tolerance
 * value makes it easier to satisfy, not harder -- it changes what the
 * tolerance number means (a fixed bound on any single equation's residual,
 * independent of problem size) rather than making the check stricter.
 */
public enum NormMode {
    L2,
    L_INFINITY
}