package com.rae.formicapi.foundation.math.lagrangian;

import com.rae.formicapi.foundation.math.operators.vectors.*;
import java.util.function.Supplier;

/**
 * A resizable set of Lagrangian particles, storing position, velocity,
 * force accumulator, and mass as parallel {@link DoubleVector}s (one per
 * spatial axis for position/velocity/force), plus liveness as a
 * {@link BooleanVector}.
 *
 * <p>Carries no notion of physics, grid, or boundary conditions -- it is
 * pure storage plus lifecycle management (spawn/kill/compact). Callers
 * build whatever operators they need (P2G/G2P transfer matrices, force
 * laws, ...) against these fields directly, the same way solver-side code
 * builds a {@code Matrix} against a set of {@link DoubleVector}s.
 */
public final class ParticleSystem {

    private final int dim;

    private final DoubleVector[] position;   // dim vectors
    private final DoubleVector[] velocity;   // dim vectors
    private final DoubleVector[] force;      // dim vectors, accumulator
    private final DoubleVector   mass;

    private final BooleanVector alive;

    private int count;   // active slots occupy [0, count)

    public ParticleSystem(int dim, Supplier<DoubleVector> doubleFactory, Supplier<BooleanVector> boolFactory) {
        this.dim = dim;
        position = new DoubleVector[dim];
        velocity = new DoubleVector[dim];
        force    = new DoubleVector[dim];
        for (int d = 0; d < dim; d++) {
            position[d] = doubleFactory.get();
            velocity[d] = doubleFactory.get();
            force[d]    = doubleFactory.get();
        }
        mass = doubleFactory.get();
        alive = boolFactory.get();
    }

    public int dim()   { return dim; }
    public int count() { return count; }

    public DoubleVector position(int axis) { return position[axis]; }
    public DoubleVector velocity(int axis) { return velocity[axis]; }
    public DoubleVector force(int axis)    { return force[axis]; }
    public DoubleVector mass()             { return mass; }

    public void clearForces() { for (DoubleVector f : force) f.clear(); }

    public void ensureCapacity(int needed) {
        if (needed <= currentSize()) return;
        for (int d = 0; d < dim; d++) {
            position[d].resize(needed);
            velocity[d].resize(needed);
            force[d].resize(needed);
        }
        mass.resize(needed);
        alive.resize(needed);
    }

    private int currentSize() { return mass.size(); } // any field works; they're kept in lockstep

    /** Appends one particle, returns its index. Caller fills its fields after. */
    public int spawn() {
        ensureCapacity(count + 1);
        int idx = count++;
        alive.set(true, idx);
        return idx;
    }

    /**
     * Marks a particle for removal. Cheap (boolean-only); does not move any
     * payload data. Call {@link #compact} once, after all kill decisions
     * for the step are made, to actually reclaim the slots.
     */
    public void kill(int i) { alive.set(false, i); }

    public boolean isAlive(int i) { return alive.get(i); }

    /**
     * Removes all killed particles in one pass.
     *
     * <p>Two phases: first a swap-with-last pass over {@code alive} alone
     * (boolean-only, cheap) to compute a permutation of surviving original
     * indices; then one batched {@link DoubleVector#gather} per field
     * (position/velocity/force per axis, mass) to move the actual payload
     * -- the part worth doing as a single parallel/vectorized op instead of
     * a per-particle Java loop.
     *
     * <p>Must be called last in the step, after every {@link #kill} for
     * that step has already been issued.
     *
     * @param intScratch working buffer providing at least 1 vector of size
     *                   {@code >= count()}, used to hold the permutation
     */
    public void compact(WorkingBuffer<IntegerVector> intScratch) {
        IntegerVector perm = intScratch.get(0);
        perm.resize(count);
        for (int k = 0; k < count; k++) perm.set(k, k);

        int last = count - 1;
        int i = 0;
        while (i <= last) {
            int origIdx = perm.get(i);
            if (alive.get(origIdx)) {
                i++;
            } else {
                int tmp = perm.get(last);
                perm.set(last, perm.get(i));
                perm.set(i, tmp);
                last--;
                // don't advance i: re-check whatever just landed here
            }
        }
        int newCount = i; // == last + 1 at loop exit

        perm.resize(newCount);
        for (int d = 0; d < dim; d++) {
            position[d].gather(position[d], perm);
            velocity[d].gather(velocity[d], perm);
            force[d].gather(force[d], perm);
        }
        mass.gather(mass, perm);
        alive.gather(alive, perm);   // now uniform with the double fields, no special-case

        count = newCount;

        for (int d = 0; d < dim; d++) {
            position[d].resize(count);
            velocity[d].resize(count);
            force[d].resize(count);
        }
        mass.resize(count);
        alive.resize(count);
    }
}