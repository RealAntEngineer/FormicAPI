package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.math.solvers.LeastSquare;
import com.rae.formicapi.simulation.nodal.PhysicsType;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

public class SteadyStateSolver {

    private static final int    MAX_ITER  = 50;
    private static final double TOLERANCE = 1e-3;

    public static void solve(SimulationModel model) {

        List<PhysicsType> order = model.getSolveOrder();

        for (int iter = 0; iter < MAX_ITER; iter++) {

            double maxDelta = 0.0;

            // 1. rebuild and stamp all domain contexts
            Map<PhysicsType, SimulationContext> ctxMap = new EnumMap<>(PhysicsType.class);
            for (PhysicsType type : order) {
                DomainModel domain = model.domain(type);
                if (domain.isEmpty()) continue;

                domain.rebuildContext();

                ctxMap.put(type, domain.getContext());
            }

            // 2. stamp all couplings into the freshly built contexts
            for (SimulationComponent c : model.getComponents())
                c.stamp(ctxMap);

            // 3. solve each domain and measure convergence
            for (PhysicsType type : order) {
                DomainModel domain = model.domain(type);
                if (domain.isEmpty()) continue;

                double[] before = snapshot(domain);

                double[] x_init = domain.getNodes().stream()
                        .mapToDouble(Node::getValue)
                        .toArray();

                double[] result = LeastSquare.solve(
                        domain.getContext().matrix,
                        x_init,
                        domain.getContext().rhs,
                        5000, 1e-3f);

                for (Node node : domain.getNodes())
                    node.setValue(result[node.getId()]);

                maxDelta = Math.max(maxDelta, maxChange(domain, before));
            }

            if (maxDelta < TOLERANCE) break;
        }
    }

    private static double[] snapshot(DomainModel domain) {
        return domain.getNodes().stream()
                .mapToDouble(Node::getValue)
                .toArray();
    }

    private static double maxChange(DomainModel domain, double[] before) {
        double max = 0;
        List<Node> nodes = domain.getNodes();
        for (int i = 0; i < nodes.size(); i++)
            max = Math.max(max, Math.abs(nodes.get(i).getValue() - before[i]));
        return max;
    }
}