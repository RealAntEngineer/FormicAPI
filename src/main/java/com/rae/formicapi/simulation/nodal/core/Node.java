package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

/**
 * Abstract base for all nodes in a nodal network simulation.
 *
 * <p>A node represents a point in the network at which a scalar physical
 * quantity (e.g. temperature, voltage, angular velocity) is either unknown
 * and solved for, or fixed as a boundary condition.
 *
 * <p>Every node belongs to a {@link PhysicsType}, which carries the
 * semantic meaning of its value and associated flow quantity.
 *
 * <p>Nodes must be registered with a {@link SimulationContext} before use,
 * which assigns them a matrix index via {@link #setId(int)}.
 *
 * @see UnknownNode
 * @see FixedValueNode
 * @see PhysicsType
 */
public abstract class Node {

    private boolean setup = false;
    private int id;
    private final PhysicsType domain;

    /**
     * Creates a node belonging to the given physical domain.
     *
     * @param domain the physical domain of this node
     */
    protected Node(PhysicsType domain) {
        this.domain = domain;
    }

    /**
     * Returns the physical domain of this node.
     *
     * @return the {@link PhysicsType} of this node
     */
    public PhysicsType getDomain() {
        return domain;
    }

    /**
     * Returns the matrix index assigned to this node.
     *
     * @return the matrix row/column index
     * @throws IllegalStateException if called before {@link #setId(int)}
     */
    public int getId() {
        if (!setup) throw new IllegalStateException("getId() called before association");
        return id;
    }

    /**
     * Assigns a matrix index to this node. May only be called once.
     *
     * @param id the matrix row/column index assigned by the simulation context
     * @throws IllegalStateException if called more than once
     */
    public void setId(int id) {
        if (setup) throw new IllegalStateException("setId() called after association");
        this.id    = id;
        this.setup = true;
    }

    /**
     * Returns whether this node is an unknown to be solved for.
     *
     * @return {@code true} if unknown, {@code false} if fixed (Dirichlet)
     */
    public abstract boolean isUnknown();

    /**
     * Returns the current value at this node.
     *
     * @return the scalar value (e.g. temperature, voltage, angular velocity)
     */
    public abstract double getValue();

    /**
     * Sets the value at this node.
     *
     * <p>For fixed nodes, implementations may silently ignore this call.
     *
     * @param value the new scalar value
     */
    public abstract void setValue(double value);
}