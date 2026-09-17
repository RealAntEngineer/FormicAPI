package com.rae.formicapi.foundation.math.lagrangian;

import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;

/**
 * Midpoint (RK2) advection of a {@link ParticleSystem}'s position and
 * velocity, driven by two independent, optional velocity contributions per
 * sub-step:
 *
 * <ol>
 *   <li><b>Field velocity</b> -- an optional {@link VelocityResampler}
 *       overwrites {@code particles.velocity(*)} from wherever the
 *       particles currently are (typically G2P against a caller-built
 *       transfer matrix). Pass {@code null} to skip this entirely, e.g. for
 *       a standalone particle system with no grid coupling.</li>
 *   <li><b>Force integration</b> -- {@code particles.force(*)} and
 *       {@code particles.mass()}, both caller-filled and physics-agnostic
 *       (gravity, buoyancy, drag, springs, collision response, ...), are
 *       integrated on top via symplectic Euler:
 *       {@code velocity += h * force / mass}. This runs every sub-step
 *       regardless of whether a resampler is present.</li>
 * </ol>
 *
 * <p>Per sub-step, midpoint method, applied simultaneously across all
 * {@code dim()} axes:
 * <pre>
 * resample + apply forces at x_n            -> k1 = v(x_n)
 * x_mid = x_n + 0.5*h*k1
 * resample + apply forces at x_mid          -> k2 = v(x_mid)
 * x_{n+1} = x_n + h*k2
 * </pre>
 * Forces are re-read from {@code particles.force(*)} at both evaluations;
 * if the caller's forces are themselves position-dependent (e.g. a spring
 * to an anchor), update them inside the resampler callback (or pass a
 * resampler that only updates forces and leaves velocity to the field, if
 * there is no field). If forces are step-constant, this just integrates
 * them twice with the same value, which is what plain RK2 does anyway.
 *
 * <p>After the call, {@code particles.velocity(*)} holds the final
 * (post-force) k2 velocity.
 *
 * <p><b>Mass.</b> Particles with {@code mass == 0} are treated as massless
 * tracers by convention: callers that don't want force integration for a
 * given particle should leave its {@code force} at zero rather than rely
 * on divide-by-zero behavior, since that is backend-defined (see
 * {@link DoubleVector#divide}).
 */
public final class RK2Advector {

    private RK2Advector() {}

    /**
     * Advances {@code particles} by {@code dt}, split into
     * {@code subSteps} midpoint updates.
     *
     * @param resampler refreshes {@code particles.velocity(*)} from
     *                  {@code particles.position(*)} before force
     *                  integration each evaluation; {@code null} to skip
     *                  (pure force-driven integration)
     * @param scratch   working buffer providing at least
     *                  {@code 3 * particles.dim()} vectors: slots
     *                  {@code [0, dim)} = saved start position per axis,
     *                  {@code [dim, 2*dim)} = k1 per axis,
     *                  {@code [2*dim, 3*dim)} = acceleration temp per axis.
     *                  Each is resized internally to {@code particles.count()}
     */
    public static void step(ParticleSystem particles, VelocityResampler resampler,
                             double dt, int subSteps, WorkingBuffer<DoubleVector> scratch) {
        int dim = particles.dim();
        int n = particles.count();

        if (scratch.vectorNumber() < 3 * dim)
            throw new IllegalArgumentException(
                    "RK2Advector requires >= 3*dim (" + (3 * dim) + ") scratch vectors, got " + scratch.vectorNumber());

        double h = dt / subSteps;

        for (int s = 0; s < subSteps; s++) {
            // --- evaluation 1: k1 = field velocity(x_n) + force integration ---
            if (resampler != null) resampler.updateVelocity(particles);
            applyForces(particles, h, scratch, dim, n);

            for (int d = 0; d < dim; d++) {
                DoubleVector posStart = (DoubleVector) scratch.get(d).resize(n);
                DoubleVector k1 = (DoubleVector) scratch.get(dim + d).resize(n);

                posStart.copy(particles.position(d));
                k1.copy(particles.velocity(d));

                // move to midpoint: x_mid = x_n + 0.5*h*k1
                particles.position(d).axpy(0.5 * h, k1);
            }

            // --- evaluation 2: k2 = field velocity(x_mid) + force integration ---
            if (resampler != null) resampler.updateVelocity(particles);
            applyForces(particles, h, scratch, dim, n);

            for (int d = 0; d < dim; d++) {
                DoubleVector pos = particles.position(d);
                DoubleVector posStart = scratch.get(d);
                DoubleVector k2 = particles.velocity(d);

                // x_{n+1} = x_n + h*k2 : restore start, advance by full step
                pos.copy(posStart);
                pos.axpy(h, k2);
            }
        }
    }

    /** velocity(d) += h * force(d) / mass, for every axis, via a scratch accel temp. */
    private static void applyForces(ParticleSystem particles, double h,
                                     WorkingBuffer<DoubleVector> scratch, int dim, int n) {
        for (int d = 0; d < dim; d++) {
            DoubleVector accel = (DoubleVector) scratch.get(2 * dim + d).resize(n);
            accel.copy(particles.force(d));
            accel.divide(particles.mass());
            particles.velocity(d).axpy(h, accel);
        }
    }
}