package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.PhysicsComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

public class HeatSource implements PhysicsComponent {

    private final Node node;
    private final double flux;

    public HeatSource(Node node, double flux) {
        this.node = node;
        this.flux = flux;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        if (node.isUnknown()) {
            ctx.rhs[node.getId()] += flux;
        }

    }
}