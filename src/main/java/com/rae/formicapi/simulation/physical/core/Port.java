package com.rae.formicapi.simulation.physical.core;

public abstract class Port {

    protected final String name;

    protected Port(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
