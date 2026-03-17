package com.rae.formicapi.simulation.nodal.hydrolic;

import com.rae.formicapi.simulation.nodal.core.LinearLink;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.PhysicsComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

/**
 * Represents advective (convective) transport between two nodes in a nodal network.
 *
 * <p>Models one-directional quantity transport driven by a known flow field,
 * of the form:
 *
 * <pre>
 *     Q = G * X_a
 * </pre>
 *
 * where the fluid carries the scalar quantity (e.g. temperature) from
 * the upstream node {@code a} to the downstream node {@code b} at a fixed
 * conductance {@code G} (e.g. ṁ·Cp for thermal advection).
 *
 * <p>Unlike diffusion ({@link LinearLink}), advection is asymmetric: only the
 * upstream node {@code a} drives the transport. The matrix contribution is:
 *
 * <pre>
 *     [ +G   0 ] [X_a]
 *     [ -G   0 ] [X_b]
 * </pre>
 *
 * <p>The implementation follows the same Dirichlet handling convention as
 * {@link LinearLink}: only unknown nodes receive matrix contributions.
 * Fixed nodes and right-hand-side corrections are handled externally
 * during boundary condition assembly.
 *
 * @see LinearLink
 * @see PhysicsComponent
 * @see SimulationContext
 */
public class Advection implements PhysicsComponent {

    private final Node a;  // upstream node
    private final Node b;  // downstream node
    private final double conductance;

    /**
     * Creates an advection link between two nodes.
     *
     * @param a           upstream node (source of the transported quantity)
     * @param b           downstream node (receiver of the transported quantity)
     * @param conductance advective transfer coefficient (e.g. ṁ·Cp, must be positive)
     */
    public Advection(Node a, Node b, double conductance) {
        this.a = a;
        this.b = b;
        this.conductance = conductance;
    }

    /**
     * Stamps the advection contribution into the global system matrix.
     *
     * <p>Only unknown nodes receive matrix contributions. Fixed nodes
     * and right-hand-side corrections are handled externally.
     *
     * @param ctx the simulation context containing the matrix and RHS
     */
    @Override
    public void stamp(SimulationContext ctx) {

        boolean au = a.isUnknown();
        boolean bu = b.isUnknown();
        int i = a.getId();
        int j = b.getId();

        if (au) {
            ctx.matrix.add(i, i, conductance);
        }

        if (bu) {
            ctx.matrix.add(j, i, -conductance);
        }

        // both fixed → nothing
    }
}
