package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

import java.util.ArrayList;
import java.util.List;

public class DomainModel {

    private final PhysicsType type;
    private final List<Node> nodes = new ArrayList<>();
    private final List<DomainComponent> components = new ArrayList<>();
    private SimulationContext context;

    public DomainModel(PhysicsType type) { this.type = type; }

    public PhysicsType getType() { return type; }
    public boolean isEmpty()     { return nodes.isEmpty(); }

    public Node addNode(Node node) {
        if (node.getDomain() != type)
            throw new IllegalArgumentException(
                    "Node domain " + node.getDomain() + " does not match DomainModel " + type);
        nodes.add(node);
        return node;
    }

    public void addComponent(DomainComponent component) { components.add(component); }

    public List<Node> getNodes()                  { return nodes; }
    public List<DomainComponent> getComponents() { return components; }

    /**
     * Rebuilds a fresh context from the current node state.
     * Must be called at the start of each iteration before stamping.
     */
    public SimulationContext rebuildContext() {
        int n = nodes.size();
        context = new SimulationContext(n, false);
        int id = 0;
        for (Node node : nodes) {
            node.setId(id);
            if (node instanceof FixedValueNode) {
                context.matrix.set(id, id, 1);
                context.rhs[id] = node.getValue();
            }
            id++;
        }
        return context;
    }

    /** Returns the current context — valid after rebuildContext(). */
    public SimulationContext getContext() {
        if (context == null)
            throw new IllegalStateException("rebuildContext() not called for domain " + type);
        return context;
    }
}