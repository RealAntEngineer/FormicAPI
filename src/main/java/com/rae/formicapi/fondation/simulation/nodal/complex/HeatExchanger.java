package com.rae.formicapi.fondation.simulation.nodal.complex;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationContext;
import com.rae.formicapi.fondation.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.fondation.simulation.nodal.linear.thermal.Convection;

import java.util.*;


/**
 * Discretised counter-/parallel-flow heat exchanger between two fluid streams.
 *
 * <p>The exchanger is split into {@code N = points} axial segments, each owning:
 * <ul>
 *   <li>one wall thermal node {@code Tw[i]}</li>
 *   <li>one fluid-1 temperature node {@code Tf1[i]}</li>
 *   <li>one fluid-2 temperature node {@code Tf2[i]}</li>
 * </ul>
 * plus a trailing node at each stream outlet ({@code Tf1[N]}, {@code Tf2[N]}).
 *
 * <p>Flow-direction convention (same as {@link Convection}):
 * <pre>
 *   ṁ₁ &gt; 0 → stream 1 flows Tf1[0] → Tf1[N]   (left to right)
 *   ṁ₂ &gt; 0 → stream 2 flows Tf2[0] → Tf2[N]   (same direction  → parallel)
 *   ṁ₂ &lt; 0 → stream 2 flows Tf2[N] → Tf2[0]   (opposite        → counter)
 * </pre>
 *
 * <p>Per segment, three pairs of terms are stamped:
 * <pre>
 *   advection stream 1 : ṁ₁·Cp₁ · T_upstream_1  (between adjacent Tf1 nodes)
 *   advection stream 2 : ṁ₂·Cp₂ · T_upstream_2  (between adjacent Tf2 nodes)
 *   diffusion  wall↔f1 : hA₁/N  · (Tf1[i] − Tw[i])   symmetric
 *   diffusion  wall↔f2 : hA₂/N  · (Tf2[i] − Tw[i])   symmetric
 * </pre>
 *
 * <p>The four hydraulic interface nodes are <em>read-only</em>: only their
 * mass-flow values are used, consistent with the frozen-coupling strategy.
 *
 *
 */
public class HeatExchanger implements SimulationComponent {
    // stream 1 ──► Tf1[0] ──► Tf1[1] ──► … ──► Tf1[N]
    // *                  ╲           ╲                 ╲
    // *                hA₁/N       hA₁/N             hA₁/N
    // *                    ╲           ╲                 ╲
    // *                   Tw[0]       Tw[1]  …          Tw[N-1]
    // *                    ╱           ╱                 ╱
    // *                hA₂/N       hA₂/N             hA₂/N
    // *                  ╱           ╱                 ╱
    // * stream 2 ◄── Tf2[0] ◄── Tf2[1] ◄── … ◄── Tf2[N]   (counter-flow: mDot2 < 0)
    //
    // ── interface nodes — THERMAL + HYDRAULIC ────────────────────────────────
    private final Node fluidInput1, fluidOutput1;
    private final Node fluidInput2, fluidOutput2;

    // ── internal nodes ────────────────────────────────────────────────────────
    /**
     * N−1 internal fluid nodes per stream, plus N wall nodes.
     */
    private final List<Node> fluid1InternalNodes;  // THERMAL + HYDRAULIC
    private final List<Node> fluid2InternalNodes;  // THERMAL + HYDRAULIC
    private final List<Node> wallNodes;            // THERMAL only

    /**
     * Full ordered chain including interface endpoints:
     * fluid1Chain = [fluidInput1, ...internal..., fluidOutput1]   length N+1
     * fluid2Chain = [fluidInput2, ...internal..., fluidOutput2]   length N+1
     */
    private final List<Node> fluid1Chain;
    private final List<Node> fluid2Chain;

    private final double cp1, cp2;
    private final double segHa1, segHa2;
    /**
     * Per-segment hydraulic conductance G = N / R_total [kg/(s·Pa)]
     */
    private final double segG1, segG2;
    private final int N;

    /**
     * @param fluidInput1  THERMAL+HYDRAULIC node at stream-1 inlet
     * @param fluidOutput1 THERMAL+HYDRAULIC node at stream-1 outlet
     * @param fluidInput2  THERMAL+HYDRAULIC node at stream-2 inlet
     * @param fluidOutput2 THERMAL+HYDRAULIC node at stream-2 outlet
     * @param cp1          stream-1 specific heat [J/(kg·K)]
     * @param cp2          stream-2 specific heat [J/(kg·K)]
     * @param totalHa1     total h·A for stream-1 side of wall [W/K]
     * @param totalHa2     total h·A for stream-2 side of wall [W/K]
     * @param totalR1      total hydraulic resistance of stream 1 [Pa·s/kg]
     * @param totalR2      total hydraulic resistance of stream 2 [Pa·s/kg]
     * @param points       number of axial segments (≥ 1)
     */
    public HeatExchanger(Node fluidInput1, Node fluidOutput1,
                         Node fluidInput2, Node fluidOutput2,
                         double cp1, double cp2,
                         double totalHa1, double totalHa2,
                         double totalR1, double totalR2,
                         int points) {

        if (points < 1)
            throw new IllegalArgumentException("points must be ≥ 1, got: " + points);

        validateFluidNode(fluidInput1, "fluidInput1");
        validateFluidNode(fluidOutput1, "fluidOutput1");
        validateFluidNode(fluidInput2, "fluidInput2");
        validateFluidNode(fluidOutput2, "fluidOutput2");

        this.fluidInput1 = fluidInput1;
        this.fluidOutput1 = fluidOutput1;
        this.fluidInput2 = fluidInput2;
        this.fluidOutput2 = fluidOutput2;
        this.cp1 = cp1;
        this.cp2 = cp2;
        this.N = points;
        this.segHa1 = totalHa1 / points;
        this.segHa2 = totalHa2 / points;
        // Segments are in series → R_seg = R_total / N → G_seg = N / R_total
        this.segG1 = N / totalR1;
        this.segG2 = N / totalR2;

        // N-1 internal fluid nodes per stream (endpoints are the interface nodes)
        fluid1InternalNodes = new ArrayList<>(N - 1);
        fluid2InternalNodes = new ArrayList<>(N - 1);
        for (int i = 0; i < N - 1; i++) {
            fluid1InternalNodes.add(new UnknownNode(ModelType.THERMAL, ModelType.HYDRAULIC));
            fluid2InternalNodes.add(new UnknownNode(ModelType.THERMAL, ModelType.HYDRAULIC));
        }

        wallNodes = new ArrayList<>(N);
        for (int i = 0; i < N; i++)
            wallNodes.add(new UnknownNode(ModelType.THERMAL));

        // Build full chains: inlet → [internals] → outlet
        fluid1Chain = buildChain(fluidInput1, fluid1InternalNodes, fluidOutput1);
        fluid2Chain = buildChain(fluidInput2, fluid2InternalNodes, fluidOutput2);
    }

    // ── SimulationComponent ───────────────────────────────────────────────────

    private static void validateFluidNode(Node n, String name) {
        if (!n.participatesIn(ModelType.THERMAL) || !n.participatesIn(ModelType.HYDRAULIC))
            throw new IllegalArgumentException(
                    name + " must be THERMAL+HYDRAULIC, got: " + n.getDomains());
    }

    private static List<Node> buildChain(Node inlet,
                                         List<Node> internals,
                                         Node outlet) {
        List<Node> chain = new ArrayList<>(internals.size() + 2);
        chain.add(inlet);
        chain.addAll(internals);
        chain.add(outlet);
        return Collections.unmodifiableList(chain);
    }

    @Override
    public Set<ModelType> getDomains() {
        return EnumSet.of(ModelType.THERMAL, ModelType.HYDRAULIC);
    }

    @Override
    public List<Node> getInternalNodes() {
        List<Node> all = new ArrayList<>(3 * N);
        all.addAll(wallNodes);
        all.addAll(fluid1InternalNodes);
        all.addAll(fluid2InternalNodes);
        return Collections.unmodifiableList(all);
    }

    // ── thermal stamping ──────────────────────────────────────────────────────

    @Override
    public List<Node> getInterfaceNodes() {
        return List.of(fluidInput1, fluidOutput1, fluidInput2, fluidOutput2);
    }

    // ── hydraulic stamping ────────────────────────────────────────────────────

    @Override
    public void stamp(Map<ModelType, SimulationContext> contexts) {
        stampHydraulic(contexts.get(ModelType.HYDRAULIC));
        stampThermal(contexts.get(ModelType.THERMAL));
    }

    private void stampHydraulic(SimulationContext ctx) {
        if (ctx == null) return;
        stampHydraulicChain(ctx, fluid1Chain, segG1);
        stampHydraulicChain(ctx, fluid2Chain, segG2);
    }

    // ── private helpers ───────────────────────────────────────────────────────

    private void stampThermal(SimulationContext ctx) {
        if (ctx == null) return;

        stampAdvectionChain(ctx, fluid1Chain, segG1, cp1);
        stampAdvectionChain(ctx, fluid2Chain, segG2, cp2);

        for (int i = 0; i < N; i++) {
            stampDiffusion(ctx, wallNodes.get(i), fluid1Chain.get(i), segHa1);
            stampDiffusion(ctx, wallNodes.get(i), fluid2Chain.get(i), segHa2);
        }
    }

    /**
     * Stamps a pressure-drop resistor between each adjacent pair of nodes.
     *
     * <pre>
     *   ṁ = G · (Pa − Pb)
     *
     *   row a :  +G·Pa  −G·Pb
     *   row b :  −G·Pa  +G·Pb
     * </pre>
     * <p>
     * Identical in form to thermal diffusion, with pressure playing the role
     * of temperature and hydraulic conductance playing the role of h·A.
     */
    private void stampHydraulicChain(SimulationContext ctx,
                                     List<Node> chain, double segG) {
        for (int k = 0; k < chain.size() - 1; k++) {
            Node    a  = chain.get(k);
            Node    b  = chain.get(k + 1);
            boolean au = a.isUnknown(ModelType.HYDRAULIC);
            boolean bu = b.isUnknown(ModelType.HYDRAULIC);
            int     ai = a.getId(ModelType.HYDRAULIC);
            int     bi = b.getId(ModelType.HYDRAULIC);

            if (au) {
                ctx.matrix.add(ai, ai, segG);
                ctx.matrix.add(ai, bi, -segG);
            }
            if (bu) {
                ctx.matrix.add(bi, bi, segG);
                ctx.matrix.add(bi, ai, -segG);
            }
        }
    }

    /**
     * Stamps advection along a chain, deriving local ṁ from the pressure
     * difference across each segment rather than a stored node value.
     *
     * <pre>
     *   ṁ_k = segG · (P_k − P_{k+1})     (positive → flow k → k+1)
     *   gAdv_k = ṁ_k · Cp
     * </pre>
     */
    private void stampAdvectionChain(SimulationContext ctx,
                                     List<Node> chain, double segG, double cp) {
        for (int k = 0; k < chain.size() - 1; k++) {
            Node a = chain.get(k);
            Node b = chain.get(k + 1);

            double Pa   = a.getValue(ModelType.HYDRAULIC);
            double Pb   = b.getValue(ModelType.HYDRAULIC);
            double gAdv = segG * (Pa - Pb) * cp;  // ṁ·Cp for this segment

            if (Math.abs(gAdv) < 1e-12) continue;

            boolean au = a.isUnknown(ModelType.THERMAL);
            boolean bu = b.isUnknown(ModelType.THERMAL);
            int     ai = a.getId(ModelType.THERMAL);
            int     bi = b.getId(ModelType.THERMAL);

            if (gAdv > 0) {
                // flow a → b, upstream temperature is Ta
                if (au) ctx.matrix.add(ai, ai, gAdv);
                if (bu) ctx.matrix.add(bi, ai, -gAdv);
            } else {
                // flow b → a, upstream temperature is Tb
                double gAbs = -gAdv;
                if (bu) ctx.matrix.add(bi, bi, gAbs);
                if (au) ctx.matrix.add(ai, bi, -gAbs);
            }
        }
    }

    private void stampDiffusion(SimulationContext ctx, Node a, Node b, double hA) {
        boolean au = a.isUnknown(ModelType.THERMAL);
        boolean bu = b.isUnknown(ModelType.THERMAL);
        int     i  = a.getId(ModelType.THERMAL);
        int     j  = b.getId(ModelType.THERMAL);
        if (au) {
            ctx.matrix.add(i, i, hA);
            ctx.matrix.add(i, j, -hA);
        }
        if (bu) {
            ctx.matrix.add(j, j, hA);
            ctx.matrix.add(j, i, -hA);
        }
    }
}