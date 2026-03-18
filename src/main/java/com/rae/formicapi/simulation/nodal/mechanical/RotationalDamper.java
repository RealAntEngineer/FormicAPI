package com.rae.formicapi.simulation.nodal.mechanical;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.*;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * Viscous rotational damper between two mechanical nodes, with heat dissipation
 * into a thermal node.
 *
 * <p>This is a multi-physics component spanning the {@link PhysicsType#MECHANICAL}
 * and {@link PhysicsType#THERMAL} domains. Both are stamped in a single
 * {@link #stamp(Map)} call by the staggered solver.
 *
 * <p>The mechanical behavior follows the linear relation:
 *
 * <pre>
 *     τ = G · (ω_a - ω_b)
 * </pre>
 *
 * <p>The dissipated power is nonlinear and computed from the frozen mechanical
 * state of the current iteration:
 *
 * <pre>
 *     P = G · (ω_a - ω_b)²
 * </pre>
 *
 * <p>It is injected directly into the thermal RHS at the heat node. Convergence
 * of the outer iteration loop resolves the coupling between mechanical velocity
 * and thermal dissipation.
 *
 * <p>The mechanical stamp is symmetric:
 *
 * <pre>
 *     [ +G  -G ] [ω_a]
 *     [ -G  +G ] [ω_b]
 * </pre>
 *
 * @see PhysicsType
 * @see SimulationComponent
 */
public class RotationalDamper implements SimulationComponent {

    private final Node a;
    private final Node b;
    private final Node heat;
    private final double damping;

    /**
     * Creates a rotational damper between two mechanical nodes, dissipating
     * heat into a thermal node.
     *
     * @param a       first mechanical node (must be {@link PhysicsType#MECHANICAL})
     * @param b       second mechanical node (must be {@link PhysicsType#MECHANICAL})
     * @param heat    thermal sink node (must be {@link PhysicsType#THERMAL})
     * @param damping viscous damping coefficient G [N·m·s/rad], must be positive
     * @throws IllegalArgumentException if any node has the wrong domain, or damping is not positive
     */
    public RotationalDamper(Node a, Node b, Node heat, double damping) {
        if (a.getDomain() != PhysicsType.MECHANICAL)
            throw new IllegalArgumentException("Node 'a' must be MECHANICAL, got: " + a.getDomain());
        if (b.getDomain() != PhysicsType.MECHANICAL)
            throw new IllegalArgumentException("Node 'b' must be MECHANICAL, got: " + b.getDomain());
        if (heat.getDomain() != PhysicsType.THERMAL)
            throw new IllegalArgumentException("Node 'heat' must be THERMAL, got: " + heat.getDomain());
        if (damping <= 0)
            throw new IllegalArgumentException("Damping must be positive, got: " + damping);

        this.a       = a;
        this.b       = b;
        this.heat    = heat;
        this.damping = damping;
    }

    @Override
    public Set<PhysicsType> getDomains() {
        return EnumSet.of(PhysicsType.MECHANICAL, PhysicsType.THERMAL);
    }

    @Override
    public void stamp(Map<PhysicsType, SimulationContext> contexts) {

        SimulationContext mechCtx  = contexts.get(PhysicsType.MECHANICAL);
        SimulationContext thermCtx = contexts.get(PhysicsType.THERMAL);

        // mechanical stamp
        if (mechCtx != null) {
            boolean au = a.isUnknown();
            boolean bu = b.isUnknown();
            int i = a.getId();
            int j = b.getId();

            if (au) {
                mechCtx.matrix.add(i, i,  damping);
                mechCtx.matrix.add(i, j, -damping);
            }
            if (bu) {
                mechCtx.matrix.add(j, j,  damping);
                mechCtx.matrix.add(j, i, -damping);
            }
        }

        // thermal injection from dissipated power
        if (thermCtx != null && heat.isUnknown()) {
            double delta = a.getValue() - b.getValue();
            thermCtx.rhs[heat.getId()] += damping * delta * delta;
        }
    }
}