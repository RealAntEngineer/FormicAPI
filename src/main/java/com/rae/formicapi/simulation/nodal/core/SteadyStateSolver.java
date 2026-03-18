package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.math.solvers.LeastSquare;
import com.rae.formicapi.simulation.nodal.PhysicsType;

import java.util.List;

public class SteadyStateSolver {

    private static final int    MAX_ITER  = 50;
    private static final double TOLERANCE = 1e-3;

    public static void solve(SimulationModel model) {

        List<PhysicsType> order = model.getSolveOrder();

        for (int iter = 0; iter < MAX_ITER; iter++) {

            double maxDelta = 0.0;

            for (PhysicsType type : order) {

                DomainModel domain = model.domain(type);
                if (domain.isEmpty()) continue;

                double[] before = snapshot(domain);

                domain.rebuildContext();

                for (CoupledComponent c : model.getCouplings()) {
                    if (c.getSinkDomain() == type) {
                        SimulationContext srcCtx = model.domain(c.getSourceDomain()).getContext();
                        c.stampCoupling(srcCtx, domain.getContext());
                    }
                }

                for (PhysicsComponent c : domain.getComponents())
                    c.stamp(domain.getContext());

                double[] x_init = domain.getNodes().stream()
                        .mapToDouble(Node::getValue).toArray();

                double[] result = LeastSquare.solve(
                        domain.getContext().matrix, x_init, domain.getContext().rhs, 5000, 1e-3f);

                for (Node node : domain.getNodes())
                    node.setValue(result[node.getId()]);

                maxDelta = Math.max(maxDelta, maxChange(domain, before));
            }

            if (maxDelta < TOLERANCE) break;
        }
    }

    private static double[] snapshot(DomainModel domain) {
        return domain.getNodes().stream().mapToDouble(Node::getValue).toArray();
    }

    private static double maxChange(DomainModel domain, double[] before) {
        double max = 0;
        List<Node> nodes = domain.getNodes();
        for (int i = 0; i < nodes.size(); i++)
            max = Math.max(max, Math.abs(nodes.get(i).getValue() - before[i]));
        return max;
    }
}