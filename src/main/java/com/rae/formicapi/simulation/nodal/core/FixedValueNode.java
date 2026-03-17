package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsDomain;

/**
 * A node with a fixed, prescribed value acting as a Dirichlet boundary condition.
 *
 * <p>Fixed nodes do not contribute rows to the system matrix. Their known
 * values influence unknown nodes through the components that connect them.
 *
 * <p>Calls to {@link #setValue(double)} are silently ignored.
 *
 * @see Node
 * @see PhysicsDomain
 */
public class FixedValueNode extends Node {

    private final double value;

    /**
     * Creates a fixed-value node in the given domain.
     *
     * @param domain the physical domain of this node
     * @param value  the prescribed scalar value (e.g. temperature, voltage)
     */
    public FixedValueNode(PhysicsDomain domain, double value) {
        super(domain);
        this.value = value;
    }

    @Override public boolean isUnknown()             { return false; }
    @Override public double getValue()               { return value; }
    @Override public void setValue(double value)     { /* fixed — ignore */ }

    @Override
    public String toString() {
        return String.format("FixedNode[%s] id=%d  %s = %.4f",
                getDomain().name(), getId(), getDomain().valueName, value);
    }
}
