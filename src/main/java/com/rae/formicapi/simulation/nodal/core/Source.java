package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

public class Source extends SingleDomainComponent {

    private final Node node;
    private final double flux;

    public Source(Node node, double flux) {
        this.node = node;
        this.flux = flux;
    }

    @Override
    public PhysicsType getDomain() {
        return PhysicsType.THERMAL;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        if (node.isUnknown()) {
            ctx.rhs[node.getId()] += flux;
        }

    }
}