package com.rae.formicapi.simulation.nodal.core;

public class FixedValueNode implements Node {

    private final double value;
    private boolean setup = false;
    private int id;

    public FixedValueNode(double value) {
        this.value = value;
        //this.id = id;
    }

    @Override
    public boolean isUnknown() {
        return false;
    }

    @Override
    public int getId() {
        if (!setup) throw new IllegalStateException("getId() called before association");
        return id;
    }

    @Override
    public void setId(int id) {
        this.id = id;
        this.setup = true;
    }
    /*
    @Override
    public int getId() {
        throw new IllegalStateException("Fixed node has no matrix id");
    }*/

    @Override
    public double getValue() {
        return value;
    }

    @Override
    public void setValue(double value) {
        //ignore set value, we are fixed.
    }
}
