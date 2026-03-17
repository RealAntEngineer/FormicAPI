package com.rae.formicapi.simulation.physical.ports;

import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.physical.core.Port;

public class MechanicalPort extends Port {

    private final Node torqueNode;
    private final Node speedNode;

    public MechanicalPort(String name, Node torqueNode, Node speedNode) {
        super(name);
        this.torqueNode = torqueNode;
        this.speedNode = speedNode;
    }
}
