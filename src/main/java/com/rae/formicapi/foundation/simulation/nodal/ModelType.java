package com.rae.formicapi.foundation.simulation.nodal;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuVector;
import com.rae.formicapi.foundation.math.solvers.LeastSquare;
import com.rae.formicapi.foundation.simulation.nodal.core.DomainModel;
import com.rae.formicapi.foundation.simulation.nodal.core.Node;
import com.rae.formicapi.foundation.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.foundation.simulation.nodal.core.SimulationContext;

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
            SimulationContext ctx    = domain.getContext();
            double[]          x0     = nodeValues(domain);
            LeastSquare.solve(ctx.matrix, new CpuVector(ctx.rhs), new CpuVector(x0), 5000, 1e-3f);
            applyResult(domain, x0);
        }
    },

    MECHANICAL("Angular velocity [rad/s]", "Torque [N·m]") {
        @Override
        public void solve(DomainModel domain, List<SimulationComponent> components) {
            SimulationContext ctx    = domain.getContext();
            double[]          x0     = nodeValues(domain);
            LeastSquare.solve(ctx.matrix, new CpuVector(x0), new CpuVector(ctx.rhs), 5000, 1e-3f);
            applyResult(domain, x0);
        }
    },

    HYDRAULIC("Pressure [Pa]", "Mass flow [kg/s]") {
        @Override
        public void solve(DomainModel domain, List<SimulationComponent> components) {
            SimulationContext ctx    = domain.getContext();
            double[]          x0     = nodeValues(domain);
            LeastSquare.solve(ctx.matrix, new CpuVector(ctx.rhs), new CpuVector(x0), 5000, 1e-3f);
            applyResult(domain, x0);
        }
    };

    // ── metadata ───────────────────────────────────────────────────────────
    public final String valueName;
    public final String flowName;

    ModelType(String valueName, String flowName) {
        this.valueName = valueName;
        this.flowName = flowName;
    }

    // ── shared helpers ─────────────────────────────────────────────────────
    protected static double[] nodeValues(DomainModel domain) {
        return domain.getNodes().stream().mapToDouble((node) -> node.getValue(domain.getType())).toArray();
    }

    protected static void applyResult(DomainModel domain, double[] result) {
        for (Node node : domain.getNodes())
            node.setValue(domain.getType(), result[node.getId(domain.getType())]);
    }

    // ── strategy contract ──────────────────────────────────────────────────
    public abstract void solve(DomainModel domain, List<SimulationComponent> components);
}