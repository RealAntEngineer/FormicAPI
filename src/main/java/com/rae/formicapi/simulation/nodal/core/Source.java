package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.ModelType;

import java.util.List;

public class Source extends SingleDomainComponent {

    private final Node node;
    private final double flux;
    private final ModelType type;

    public Source(Node node, ModelType type, double flux) {
        this.node = node;
        this.flux = flux;
        this.type = type;
    }

    @Override
    public ModelType getDomain() {
        return type;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        if (node.isUnknown(this.getDomain())) {
            ctx.rhs[node.getId(this.getDomain())] += flux;
        }

    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of();
    }
}