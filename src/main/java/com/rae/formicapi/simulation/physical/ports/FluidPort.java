package com.rae.formicapi.simulation.physical.ports;

import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.physical.core.Port;

public class FluidPort extends Port {

    private final Node pressureNode;
    private final Node temperatureNode;

    public FluidPort(String name, Node pressureNode, Node temperatureNode) {
        super(name);
        this.pressureNode = pressureNode;
        this.temperatureNode = temperatureNode;
    }

    public Node getPressureNode() {
        return pressureNode;
    }

    public Node getTemperatureNode() {
        return temperatureNode;
    }
}
