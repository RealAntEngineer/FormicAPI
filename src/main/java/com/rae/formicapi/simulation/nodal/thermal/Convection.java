package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * Models convective heat transfer between two thermal nodes driven by a fluid flow.
 *
 * <p>This is a multi-physics component spanning the {@link PhysicsType#THERMAL} and
 * {@link PhysicsType#HYDRAULIC} domains. The mass flow rate ṁ is read from a hydraulic
 * node (frozen from the previous hydraulic solve) and used to compute both:
 *
 * <ul>
 *     <li><b>Film conductance</b> h·A = h(ṁ) via a user-supplied correlation</li>
 *     <li><b>Advective transport</b> Q_adv = ṁ·Cp · T_upstream</li>
 * </ul>
 *
 * <p>The total thermal stamp combines both mechanisms:
 *
 * <pre>
 *     diffusion:  Q = h(ṁ)·A · (Ta - Tb)      symmetric
 *     advection:  Q = ṁ·Cp  · T_upstream       asymmetric, direction from sign of ṁ
 * </pre>
 *
 * <p>The hydraulic node is <em>read-only</em> — no hydraulic matrix contribution is
 * stamped. The HYDRAULIC context may be absent (prescribed-flow mode), in which
 * case the mass flow node value is still used as-is from a previous solve or
 * initial condition.
 *
 * <p>Flow direction convention: {@code ṁ > 0} means flow from {@code thermalA}
 * to {@code thermalB}.
 */
public class Convection implements SimulationComponent {

    /**
     * Correlation mapping |ṁ| [kg/s] to convective film conductance h·A [W/K].
     * Implementations encode Nusselt/Reynolds correlations for the geometry.
     */
    @FunctionalInterface
    public interface HCorrelation {
        double compute(double absMassFlow);
    }

    private final Node thermalA;
    private final Node thermalB;
    private final Node massFlowNode;
    private final double cp;
    private final HCorrelation hCorr;

    /**
     * @param thermalA     first thermal node
     * @param thermalB     second thermal node
     * @param massFlowNode hydraulic node providing ṁ [kg/s]
     * @param cp           fluid specific heat [J/(kg·K)]
     * @param hCorr        correlation giving h·A [W/K] as a function of |ṁ| [kg/s]
     */
    public Convection(Node thermalA, Node thermalB, Node massFlowNode,
                      double cp, HCorrelation hCorr) {
        if (thermalA.getDomain() != PhysicsType.THERMAL)
            throw new IllegalArgumentException("thermalA must be THERMAL, got: " + thermalA.getDomain());
        if (thermalB.getDomain() != PhysicsType.THERMAL)
            throw new IllegalArgumentException("thermalB must be THERMAL, got: " + thermalB.getDomain());
        if (massFlowNode.getDomain() != PhysicsType.HYDRAULIC)
            throw new IllegalArgumentException("massFlowNode must be HYDRAULIC, got: " + massFlowNode.getDomain());
        this.thermalA     = thermalA;
        this.thermalB     = thermalB;
        this.massFlowNode = massFlowNode;
        this.cp           = cp;
        this.hCorr        = hCorr;
    }

    /**
     * Convenience constructor with a constant h·A independent of flow.
     *
     * @param hA fixed convective conductance h·A [W/K]
     */
    public Convection(Node thermalA, Node thermalB, Node massFlowNode,
                      double cp, double hA) {
        this(thermalA, thermalB, massFlowNode, cp, mDot -> hA);
    }

    @Override
    public Set<PhysicsType> getDomains() {
        return EnumSet.of(PhysicsType.THERMAL, PhysicsType.HYDRAULIC);
    }

    @Override
    public void stamp(Map<PhysicsType, SimulationContext> contexts) {

        SimulationContext thermCtx = contexts.get(PhysicsType.THERMAL);
        if (thermCtx == null) return;

        double mDot = massFlowNode.getValue();
        double hA   = hCorr.compute(Math.abs(mDot));
        double gAdv = mDot * cp;

        boolean au = thermalA.isUnknown();
        boolean bu = thermalB.isUnknown();
        int i = thermalA.getId();
        int j = thermalB.getId();

        // diffusive part — symmetric
        if (au) {
            thermCtx.matrix.add(i, i,  hA);
            thermCtx.matrix.add(i, j, -hA);
        }
        if (bu) {
            thermCtx.matrix.add(j, j,  hA);
            thermCtx.matrix.add(j, i, -hA);
        }

        // advective part — asymmetric, only upstream node drives transport
        if (gAdv > 0) {
            // flow a → b
            if (au) thermCtx.matrix.add(i, i,  gAdv);
            if (bu) thermCtx.matrix.add(j, i, -gAdv);
        } else if (gAdv < 0) {
            // flow b → a
            double gAbs = -gAdv;
            if (bu) thermCtx.matrix.add(j, j,  gAbs);
            if (au) thermCtx.matrix.add(i, j, -gAbs);
        }
    }
}
