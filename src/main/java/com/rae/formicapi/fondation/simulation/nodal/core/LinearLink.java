package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;

import java.util.List;

/**
 * Represents a generic linear flow link between two nodes in a nodal network.
 *
 * <p>This component models any physical quantity transfer that obeys a linear
 * relation of the form:
 *
 * <pre>
 *     Q = G * (X_a - X_b)
 * </pre>
 * <p>
 * where:
 * <ul>
 *     <li>{@code Q} is the flow of the quantity (e.g., heat flux, mass flux, current)</li>
 *     <li>{@code G} is the linear conductance/transfer coefficient</li>
 *     <li>{@code X_a}, {@code X_b} are the node values (e.g., temperature, pressure, voltage)</li>
 * </ul>
 *
 * <p>During matrix assembly, this component contributes to the global system:
 *
 * <pre>
 *     [ +G  -G ] [X_a]
 *     [ -G  +G ] [X_b]
 * </pre>
 *
 * <p>The implementation accounts for nodes that may already have fixed values
 * (Dirichlet boundary conditions):
 *
 * <ul>
 *     <li>If a node is unknown, it receives the usual contribution to the matrix.</li>
 *     <li>If a node is fixed, contributions are only added to the unknown nodes</li>
 *     <li>If both nodes are fixed, no matrix contribution is added.</li>
 * </ul>
 *
 * <p>This component is suitable for any linear nodal network simulation:
 * thermal, hydraulic, electrical, or other flow-based systems.
 *
 * @see SingleDomainComponent
 * @see SimulationContext
 */
public class LinearLink extends SingleDomainComponent {

    private final Node a;
    private final Node b;
    private final ModelType type;
    private final double conductance;

    /**
     * Creates a linear link between two nodes.
     *
     * @param a           first node
     * @param b           second node
     * @param conductance linear transfer coefficient (must be positive)
     */
    public LinearLink(Node a, Node b, ModelType type, double conductance) {
        this.a = a;
        this.b = b;
        this.type = type;
        this.conductance = conductance;
    }

    @Override
    public ModelType getDomain() {
        return type;
    }

    /**
     * Stamps the link contribution into the global system matrix.
     *
     * <p>This method handles unknown and fixed nodes appropriately:
     * only unknown nodes contribute to the matrix; fixed nodes affect the
     * right-hand side if necessary.
     *
     * @param ctx the simulation context containing the matrix and RHS
     */
    @Override
    public void stamp(SimulationContext ctx) {

        boolean au = a.isUnknown(type);
        boolean bu = b.isUnknown(type);
        int i = a.getId(type);
        int j = b.getId(type);

        if (au) {
            ctx.matrix.add(i, i, conductance);
            ctx.matrix.add(i, j, -conductance);
        }

        if (bu) {
            ctx.matrix.add(j, j, conductance);
            ctx.matrix.add(j, i, -conductance);
        }

        // both fixed → nothing
    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of();//no internal node
    }
}