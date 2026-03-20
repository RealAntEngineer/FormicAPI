package com.rae.formicapi.simulation.nodal.linear.thermal;

import com.rae.formicapi.simulation.nodal.ModelType;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.SingleDomainComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

import java.util.List;

public class Radiation extends SingleDomainComponent {

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
    public ModelType getDomain() {
        return ModelType.THERMAL;//it's also radiative (if there is reflexions)
    }

    @Override
    public void stamp(SimulationContext ctx) {

        double Ta = a.getValue(ModelType.THERMAL);
        double Tb = b.getValue(ModelType.THERMAL);

        double Tm = 0.5 * (Ta + Tb);
        double g = 4 * SIGMA * emissivity * area * Math.pow(Tm, 3);

        boolean au = a.isUnknown(ModelType.THERMAL);
        boolean bu = b.isUnknown(ModelType.THERMAL);
        int i = a.getId(ModelType.THERMAL);
        int j = b.getId(ModelType.THERMAL);

        if (au) {
            ctx.matrix.add(i, i,  g);
            ctx.matrix.add(i, j, -g);
        }

        if (bu) {
            ctx.matrix.add(j, j,  g);
            ctx.matrix.add(j, i, -g);
        }
    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of();
    }
}