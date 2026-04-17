package com.rae.formicapi.fondation.simulation.nodal.linear.mechanical;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.core.ConstraintNode;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationContext;
import com.rae.formicapi.fondation.simulation.nodal.core.SingleDomainComponent;

import java.util.List;

public class Gear extends SingleDomainComponent {

    private final Node   a;
    private final Node   b;
    private final Node   lambda;   // constraint node, added to domain on construction
    private final double ratio;

    public Gear(Node a, Node b, double ratio) {
        this.a = a;
        this.b = b;
        this.ratio = ratio;
        this.lambda = new ConstraintNode(ModelType.MECHANICAL);  // registers with MECHANICAL domain
    }

    @Override
    public ModelType getDomain() {
        return ModelType.MECHANICAL;
    }

    @Override
    public void stamp(SimulationContext ctx) {

        int    i = a.getId(ModelType.MECHANICAL);
        int    j = b.getId(ModelType.MECHANICAL);
        int    k = lambda.getId(ModelType.MECHANICAL);
        double r = ratio;

        // torque reactions on physical nodes from λ
        if (a.isUnknown(ModelType.MECHANICAL)) ctx.matrix.add(i, k, +r);
        if (b.isUnknown(ModelType.MECHANICAL)) ctx.matrix.add(j, k, -1);

        // constraint row: r·ω_a − ω_b = 0
        ctx.matrix.add(k, i, +r);
        ctx.matrix.add(k, j, -1);
        // rhs[k] = 0 already from rebuildContext
    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of(lambda);
    }
}