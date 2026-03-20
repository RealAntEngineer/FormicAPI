package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;

import java.util.ArrayList;
import java.util.List;

public class DomainModel {

    private final ModelType type;
    private final List<Node> nodes = new ArrayList<>();
    private SimulationContext context;

    public DomainModel(ModelType type) {
        this.type = type;
    }

    public ModelType getType() {
        return type;
    }

    public boolean isEmpty() {
        return nodes.isEmpty();
    }

    public void addNode(Node node) {
        if (!node.participatesIn(type))
            throw new IllegalArgumentException(
                    "Node domains " + node.getDomains() + " does not match DomainModel " + type);
        nodes.add(node);
    }

    public List<Node> getNodes() {
        return nodes;
    }

    /**
     * Rebuilds a fresh context from the current node state.
     * Must be called at the start of each iteration before stamping.
     *
     * @return
     */
    public SimulationContext rebuildContext() {
        int n = nodes.size();
        context = new SimulationContext(n, false);
        int id = 0;
        for (Node node : nodes) {
            node.setId(type, id);
            if (node instanceof FixedValueNode) {
                context.matrix.set(id, id, 1);
                context.rhs[id] = node.getValue(type);
            }
            id++;
        }
        return context;
    }

    /**
     * Returns the current context — valid after rebuildContext().
     */
    public SimulationContext getContext() {
        if (context == null)
            throw new IllegalStateException("rebuildContext() not called for domain " + type);
        return context;
    }
}