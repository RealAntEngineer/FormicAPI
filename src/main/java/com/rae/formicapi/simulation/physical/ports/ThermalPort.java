package com.rae.formicapi.simulation.physical.ports;

import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.physical.core.Port;

public class ThermalPort extends Port {

    private final Node node;

    public ThermalPort(String name, Node node) {
        super(name);
        this.node = node;
    }

    public Node getNode() {
        return node;
    }
}