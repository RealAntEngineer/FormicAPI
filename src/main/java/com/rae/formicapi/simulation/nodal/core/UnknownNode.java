package com.rae.formicapi.simulation.nodal.core;

public class UnknownNode implements Node {

    private boolean setup = false;
    private int id;
    private double value;
    private final double capacitance;

    public UnknownNode(double capacitance) {
        this.capacitance = capacitance;
    }

    @Override
    public boolean isUnknown() {
        return true;
    }

    @Override
    public int getId() {
        if (!setup) throw new IllegalStateException("Node::getId called before association");
        return id;
    }

    @Override
    public void setId(int id) {
        if (setup) throw new IllegalStateException("Node::setId called after association");
        this.id = id;
        this.setup = true;
    }

    @Override
    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getCapacitance() {
        return capacitance;
    }
}