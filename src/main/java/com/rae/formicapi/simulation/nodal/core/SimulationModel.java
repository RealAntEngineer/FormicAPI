package com.rae.formicapi.simulation.nodal.core;

import java.util.ArrayList;
import java.util.List;

public class SimulationModel {

    private final List<Node> nodes = new ArrayList<>();
    private final List<PhysicsComponent> components = new ArrayList<>();

    public Node addNode(Node node) {
        nodes.add(node);
        return node;
    }

    public void addComponent(PhysicsComponent component) {
        components.add(component);
    }

    public List<Node> getNodes() {
        return nodes;
    }

    public List<PhysicsComponent> getComponents() {
        return components;
    }
}
