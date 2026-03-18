package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.DomainComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

public class HeatSource extends DomainComponent {

    private final Node node;
    private final double flux;

    public HeatSource(Node node, double flux) {
        this.node = node;
        this.flux = flux;
    }

    @Override
    public PhysicsType getDomain() {
        return PhysicsType.THERMAL;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        if (node.isUnknown()) {//todo, correct this shit
            ctx.rhs[node.getId()] += flux;
        }

    }
}