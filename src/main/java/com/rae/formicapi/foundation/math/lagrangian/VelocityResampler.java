package com.rae.formicapi.foundation.math.lagrangian;

/**
 * Callback that refreshes a particle system's velocity from its current
 * position -- typically G2P: applying the caller's position-dependent
 * transfer matrix W (bilinear/B-spline/etc. weights against a grid) into
 * {@code particles.velocity(d)} for each axis {@code d}.
 *
 * <p>Reads {@code particles.position(*)}, writes {@code particles.velocity(*)}.
 * Grid layout, obstacle handling, and weighting scheme are entirely the
 * caller's concern -- the advector only needs "given where particles are
 * now, what is the field velocity there."
 */
public interface VelocityResampler {
    void updateVelocity(ParticleSystem particles);
}