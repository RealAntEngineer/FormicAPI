package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.ModelType;

// Represents a Lagrange multiplier — an unknown with no physical capacitance.
public class ConstraintNode extends Node {
    private double value;

    /**
     * Creates a node belonging to the given physical domain.
     *
     * @param domain the physical domain of this node
     */
    public ConstraintNode(ModelType domain) {
        super(domain);
    }

    @Override
    public boolean isUnknown(ModelType type) {
        return true;
    }

    @Override
    public double getValue(ModelType type) {
        return value;
    }

    @Override
    public void setValue(ModelType type, double value) {
        this.value = value;
    }
}