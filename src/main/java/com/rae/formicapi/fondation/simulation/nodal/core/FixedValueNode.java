package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;

/**
 * A node with a fixed, prescribed value acting as a Dirichlet boundary condition.
 *
 * <p>Fixed nodes do not contribute rows to the system matrix. Their known
 * values influence unknown nodes through the components that connect them.
 *
 * <p>Calls to {@link #setValue(ModelType, double)} (double)} are silently ignored.
 *
 * @see Node
 * @see ModelType
 */
public class FixedValueNode extends Node {

    private final double value;

    /**
     * Creates a fixed-value node in the given domain.
     *
     * @param domain the physical domain of this node
     * @param value  the prescribed scalar value (e.g. temperature, voltage)
     */
    public FixedValueNode(ModelType domain, double value) {
        super(domain);
        this.value = value;
    }

    @Override
    public boolean isUnknown(ModelType type) {
        return false;
    }

    @Override
    public double getValue(ModelType type) {
        return value;
    }

    @Override
    public void setValue(ModelType type, double value) {

    }
}
