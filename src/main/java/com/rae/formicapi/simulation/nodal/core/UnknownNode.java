package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

/**
 * A node whose value is unknown and solved for by the linear system.
 *
 * <p>Carries a capacitance term used by the time-integration layer for
 * transient simulations (e.g. thermal capacity, moment of inertia).
 * In steady-state solves, the capacitance has no effect.
 *
 * @see Node
 * @see PhysicsType
 */
public class UnknownNode extends Node {

    private double value;
    private final double capacitance;

    /**
     * Creates an unknown node in the given domain.
     *
     * @param domain      the physical domain of this node
     * @param capacitance the storage term (e.g. Cp [J/K], J [kg·m²])
     */
    public UnknownNode(PhysicsType domain, double capacitance) {
        super(domain);
        this.capacitance = capacitance;
    }

    /**
     * Returns the storage term associated with this node.
     *
     * @return capacitance (e.g. Cp, J, C depending on domain)
     */
    public double getCapacitance() {
        return capacitance;
    }

    @Override public boolean isUnknown()                { return true;  }
    @Override public double getValue()                  { return value; }
    @Override public void setValue(double value)        { this.value = value; }

    @Override
    public String toString() {
        return String.format("UnknownNode[%s] id=%d  %s = %.4f",
                getDomain().name(), getId(), getDomain().valueName, value);
    }
}