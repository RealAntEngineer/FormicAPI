package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.core.*;

public class Conduction implements PhysicsComponent {

    private final Node a;
    private final Node b;
    private final double conductance;

    public Conduction(Node a, Node b, double conductance) {
        this.a = a;
        this.b = b;
        this.conductance = conductance;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        boolean au = a.isUnknown();
        boolean bu = b.isUnknown();
        int i = a.getId();
        int j = b.getId();

        /*if (au && bu) {



            ctx.matrix.add(i, i, conductance);
            ctx.matrix.add(i, j, -conductance);
            ctx.matrix.add(j, j, conductance);
            ctx.matrix.add(j, i, -conductance);
        }*/

        if (au) {

            //double vb = b.getValue();

            ctx.matrix.add(i, i, conductance);
            ctx.matrix.add(i, j, -conductance);
        }

        if (bu) {

            //double va = a.getValue();

            ctx.matrix.add(j, j, conductance);
            ctx.matrix.add(j, i, -conductance);
        }

        // both fixed → nothing
    }
}