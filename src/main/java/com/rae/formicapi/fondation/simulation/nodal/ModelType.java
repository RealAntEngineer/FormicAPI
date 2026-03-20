package com.rae.formicapi.fondation.simulation.nodal;

import com.rae.formicapi.fondation.math.solvers.LeastSquare;
import com.rae.formicapi.fondation.simulation.nodal.core.DomainModel;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationContext;

import java.util.List;

/**
 * Enumerates the supported physical domains in the nodal network.
 *
 * <p>Each domain defines the physical meaning of the node value and its
 * associated flow quantity, following the general relation:
 *
 * <pre>
 *     flow = conductance × (value_a - value_b)
 * </pre>
 *
 * <p>Useful for logging, debugging, and domain-aware printing.
 */
public enum ModelType {

    THERMAL("Temperature [K]", "Heat flux [W]") {
        @Override
        public void solve(DomainModel domain, List<SimulationComponent> components) {
            SimulationContext ctx = domain.getContext();
            double[] x0 = nodeValues(domain);
            double[] result = LeastSquare.solve(ctx.matrix, x0, ctx.rhs, 5000, 1e-3f);
            applyResult(domain, result);
        }
    },

    MECHANICAL("Angular velocity [rad/s]", "Torque [N·m]") {
        @Override
        public void solve(DomainModel domain, List<SimulationComponent> components) {
            SimulationContext ctx = domain.getContext();
            double[] x0 = nodeValues(domain);
            double[] result = LeastSquare.solve(ctx.matrix, x0, ctx.rhs, 5000, 1e-3f);
            applyResult(domain, result);        }
    },

    HYDRAULIC("Pressure [Pa]", "Mass flow [kg/s]") {
        @Override
        public void solve(DomainModel domain, List<SimulationComponent> components) {
            SimulationContext ctx = domain.getContext();
            double[] x0 = nodeValues(domain);
            double[] result = LeastSquare.solve(ctx.matrix, x0, ctx.rhs, 5000, 1e-3f);
            applyResult(domain, result);
        }
    };

    // ── metadata ───────────────────────────────────────────────────────────
    public final String valueName;
    public final String flowName;

    ModelType(String valueName, String flowName) {
        this.valueName = valueName;
        this.flowName  = flowName;
    }

    // ── strategy contract ──────────────────────────────────────────────────
    public abstract void solve(DomainModel domain, List<SimulationComponent> components);

    // ── shared helpers ─────────────────────────────────────────────────────
    protected static double[] nodeValues(DomainModel domain) {
        return domain.getNodes().stream().mapToDouble((node) -> node.getValue(domain.getType())).toArray();
    }

    protected static void applyResult(DomainModel domain, double[] result) {
        for (Node node : domain.getNodes())
            node.setValue(domain.getType(),result[node.getId(domain.getType())]);
    }
}