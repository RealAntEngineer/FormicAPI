package com.rae.formicapi.fondation.simulation.nodal.linear.thermal;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationContext;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Models convective heat transfer between two thermal nodes driven by a fluid flow.
 *
 * <p>This is a multi-physics component spanning the {@link ModelType#THERMAL} and
 * {@link ModelType#HYDRAULIC} domains. The mass flow rate ṁ is read from a hydraulic
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

    private final Node wall;
    private final Node flowNode;
    private final double cp;
    private final HCorrelation hCorr;
    /**
     * Convenience constructor with a constant h·A independent of flow.
     *
     * @param hA fixed convective conductance h·A [W/K]
     */
    public Convection(Node wall, Node flowNode,
                      double cp, double hA) {
        this(wall, flowNode, cp, mDot -> hA);
    }

    /**
     * @param wall     thermal node
     * @param flowNode hydraulic node providing ṁ [kg/s]
     * @param cp       fluid specific heat [J/(kg·K)]
     * @param hCorr    correlation giving h·A [W/K] as a function of |ṁ| [kg/s]
     */
    public Convection(Node wall, Node flowNode,
                      double cp, HCorrelation hCorr) {
        if (!wall.participatesIn(ModelType.THERMAL))
            throw new IllegalArgumentException("wall must be THERMAL, got: " + wall.getDomains());
        if (!flowNode.participatesIn(ModelType.HYDRAULIC) || flowNode.participatesIn(ModelType.THERMAL))
            throw new IllegalArgumentException("massFlowNode must be HYDRAULIC and THERMAL, got: " + flowNode.getDomains());
        this.wall = wall;
        this.flowNode = flowNode;
        this.cp = cp;
        this.hCorr = hCorr;
    }

    @Override
    public Set<ModelType> getDomains() {
        return EnumSet.of(ModelType.THERMAL, ModelType.HYDRAULIC);
    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of();
    }

    @Override
    public void stamp(Map<ModelType, SimulationContext> contexts) {

        SimulationContext thermCtx = contexts.get(ModelType.THERMAL);
        if (thermCtx == null) return;

        double mDot = flowNode.getValue(ModelType.HYDRAULIC);
        double hA = hCorr.compute(Math.abs(mDot));
        double gAdv = mDot * cp;

        boolean au = wall.isUnknown(ModelType.THERMAL);
        boolean bu = flowNode.isUnknown(ModelType.THERMAL);
        int i = wall.getId(ModelType.THERMAL);
        int j = flowNode.getId(ModelType.THERMAL);

        // diffusive part — symmetric
        if (au) {
            thermCtx.matrix.add(i, i, hA);
            thermCtx.matrix.add(i, j, -hA);
        }
        if (bu) {
            thermCtx.matrix.add(j, j, hA);
            thermCtx.matrix.add(j, i, -hA);
        }

        // advective part — asymmetric, only upstream node drives transport
        if (gAdv > 0) {
            // flow a → b
            if (au) thermCtx.matrix.add(i, i, gAdv);
            if (bu) thermCtx.matrix.add(j, i, -gAdv);
        } else if (gAdv < 0) {
            // flow b → a
            double gAbs = -gAdv;
            if (bu) thermCtx.matrix.add(j, j, gAbs);
            if (au) thermCtx.matrix.add(i, j, -gAbs);
        }
    }

    /**
     * Correlation mapping |ṁ| [kg/s] to convective film conductance h·A [W/K].
     * Implementations encode Nusselt/Reynolds correlations for the geometry.
     */
    @FunctionalInterface
    public interface HCorrelation {
        double compute(double absMassFlow);
    }
}
