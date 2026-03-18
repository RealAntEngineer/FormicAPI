package com.rae.formicapi.simulation.nodal.radiative;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.DomainComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

public class Radiation extends DomainComponent {

    private static final double SIGMA = 5.670374419e-8;

    private final Node a;
    private final Node b;

    private final double emissivity;
    private final double area;

    public Radiation(Node a, Node b, double emissivity, double area) {
        this.a = a;
        this.b = b;
        this.emissivity = emissivity;
        this.area = area;
    }

    @Override
    public PhysicsType getDomain() {
        return PhysicsType.THERMAL;//it's also radiative (if there is reflexions)
    }

    @Override
    public void stamp(SimulationContext ctx) {

        double Ta = a.getValue();
        double Tb = b.getValue();

        double Tm = 0.5 * (Ta + Tb);
        double g = 4 * SIGMA * emissivity * area * Math.pow(Tm, 3);

        boolean au = a.isUnknown();
        boolean bu = b.isUnknown();
        int i = a.getId();
        int j = b.getId();

        if (au) {
            ctx.matrix.add(i, i,  g);
            ctx.matrix.add(i, j, -g);
        }

        if (bu) {
            ctx.matrix.add(j, j,  g);
            ctx.matrix.add(j, i, -g);
        }
    }
}