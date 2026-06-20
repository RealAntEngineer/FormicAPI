package com.rae.formicapi.fondation.simulation.nodal;

import com.rae.formicapi.fondation.simulation.nodal.core.*;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

public class SteadyStateSolver {

    private static final int    MAX_ITER  = 50;
    private static final double TOLERANCE = 1e-3;

    public static void solve(SimulationModel model) {

        List<ModelType> order = model.getSolveOrder();

        for (int iter = 0; iter < MAX_ITER; iter++) {

            // 1. Rebuild all contexts — zeroes every matrix and RHS
            Map<ModelType, SimulationContext> ctxMap = new EnumMap<>(ModelType.class);
            for (ModelType type : order) {
                DomainModel domain = model.domain(type);
                if (domain.isEmpty()) continue;
                ctxMap.put(type, domain.rebuildContext());
            }

            // 2. Stamp all components into all contexts.
            //    Cross-domain components (e.g. RotationalDamper) inject
            //    frozen mechanical losses into the thermal RHS here.
            for (SimulationComponent c : model.getComponents())
                c.stamp(ctxMap);

            // 3. Each domain solves itself — no solver logic here.
            double maxDelta = 0.0;
            for (ModelType type : order) {
                DomainModel domain = model.domain(type);
                if (domain.isEmpty()) continue;

                double[] before = snapshot(domain);
                type.solve(domain, model.getComponents());     // ← strategy dispatch
                maxDelta = Math.max(maxDelta, maxChange(domain, before));
            }

            if (maxDelta < TOLERANCE) break;
        }
    }

    private static double[] snapshot(DomainModel domain) {
        return domain.getNodes().stream()
                .mapToDouble((node) -> node.getValue(domain.getType()))
                .toArray();
    }

    private static double maxChange(DomainModel domain, double[] before) {
        double     max   = 0;
        List<Node> nodes = domain.getNodes();
        for (int i = 0; i < nodes.size(); i++)
            max = Math.max(max, Math.abs(nodes.get(i).getValue(domain.getType()) - before[i]));
        return max;
    }
}