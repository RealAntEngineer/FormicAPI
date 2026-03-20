package com.rae.formicapi.fondation.simulation.nodal.linear.mechanical;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationContext;
import com.rae.formicapi.simulation.nodal.core.*;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Viscous rotational damper between two mechanical nodes, with heat dissipation
 * into a thermal node.
 *
 * <p>This is a multi-physics component spanning the {@link ModelType#MECHANICAL}
 * and {@link ModelType#THERMAL} domains. Both are stamped in a single
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
 * @see ModelType
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
     * @param a       first mechanical node (must be {@link ModelType#MECHANICAL})
     * @param b       second mechanical node (must be {@link ModelType#MECHANICAL})
     * @param heat    thermal sink node (must be {@link ModelType#THERMAL})
     * @param damping viscous damping coefficient G [N·m·s/rad], must be positive
     * @throws IllegalArgumentException if any node has the wrong domain, or damping is not positive
     */
    public RotationalDamper(Node a, Node b, Node heat, double damping) {
        if (!a.participatesIn(ModelType.MECHANICAL))
            throw new IllegalArgumentException("Node 'a' must be MECHANICAL, got: " + a.getDomains());
        if (!b.participatesIn(ModelType.MECHANICAL))
            throw new IllegalArgumentException("Node 'b' must be MECHANICAL, got: " + b.getDomains());
        if (!heat.participatesIn(ModelType.THERMAL))
            throw new IllegalArgumentException("Node 'heat' must be THERMAL, got: " + heat.getDomains());
        if (damping <= 0)
            throw new IllegalArgumentException("Damping must be positive, got: " + damping);

        this.a       = a;
        this.b       = b;
        this.heat    = heat;
        this.damping = damping;
    }

    @Override
    public Set<ModelType> getDomains() {
        return EnumSet.of(ModelType.MECHANICAL, ModelType.THERMAL);
    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of();
    }

    @Override
    public void stamp(Map<ModelType, SimulationContext> contexts) {

        SimulationContext mechCtx  = contexts.get(ModelType.MECHANICAL);
        SimulationContext thermCtx = contexts.get(ModelType.THERMAL);

        // mechanical stamp
        if (mechCtx != null) {
            boolean au = a.isUnknown(ModelType.MECHANICAL);
            boolean bu = b.isUnknown(ModelType.MECHANICAL);
            int i = a.getId(ModelType.MECHANICAL);
            int j = b.getId(ModelType.MECHANICAL);

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
        if (thermCtx != null && heat.isUnknown(ModelType.THERMAL)) {
            double delta = a.getValue(ModelType.MECHANICAL) - b.getValue(ModelType.MECHANICAL);
            thermCtx.rhs[heat.getId(ModelType.THERMAL)] += damping * delta * delta;
        }
    }
}