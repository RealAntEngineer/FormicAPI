package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.DomainComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

public class Convection extends DomainComponent {

    private final Node node;
    private final Node ambient;
    private final double h;

    public Convection(Node node, Node ambient, double h) {
        this.node = node;
        this.ambient = ambient;
        this.h = h;
    }

    @Override
    public PhysicsType getDomain() {
        return PhysicsType.THERMAL;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        boolean au = node.isUnknown();
        boolean bu = ambient.isUnknown();
        int i = node.getId();
        int j = ambient.getId();

        /*if (au && bu) {



            ctx.matrix.add(i, i, conductance);
            ctx.matrix.add(i, j, -conductance);
            ctx.matrix.add(j, j, conductance);
            ctx.matrix.add(j, i, -conductance);
        }*/

        if (au) {

            //double vb = b.getValue();

            ctx.matrix.add(i, i, h);
            ctx.matrix.add(i, j, -h);
        }

        if (bu) {

            //double va = a.getValue();

            ctx.matrix.add(j, j, h);
            ctx.matrix.add(j, i, -h);
        }

        // both fixed → nothing
    }
}
