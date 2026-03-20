package com.rae.formicapi.mechanical_nodes;

import com.rae.formicapi.simulation.nodal.ModelType;
import com.rae.formicapi.simulation.nodal.SteadyStateSolver;
import com.rae.formicapi.simulation.nodal.core.*;
import com.rae.formicapi.simulation.nodal.linear.mechanical.Gear;
import com.rae.formicapi.simulation.nodal.linear.mechanical.RotationalDamper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for the mechanical nodal network.
 *
 * Convention used throughout:
 *   - "ground" node  = FixedValueNode(MECHANICAL, 0)  — the inertial reference
 *   - ω values in rad/s, τ in N·m, G in N·m·s/rad
 *   - Gear ratio r = ω_out / ω_in  (r > 1 = speed up, r < 1 = slow down)
 *
 * Each test builds a SimulationModel directly and solves it, so they are
 * independent of each other and of registration order.
 */
class MechanicalNetworkTest {

    static final double TIGHT = 1e-6;   // for exact/linear results
    static final double LOOSE = 1e-3;   // for converged NR results

    // ──────────────────────────────────────────────────────────────────────────
    // 1. SINGLE DAMPER — steady-state speed under applied torque
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * One shaft, one damper to ground, one torque source.
     *
     *   [ground ω=0] --G=2-- [shaft ?] <-- τ=6
     *
     * Torque balance at shaft:  G·ω = τ  →  ω = τ/G = 3.0
     */
    @Test
    void singleDamper_steadyStateSpeed() {
        SimulationModel model = new SimulationModel();

        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 0.0));
        Node shaft  = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new LinearLink(ground, shaft, ModelType.MECHANICAL, 2.0));
        model.addComponent(new Source(shaft,ModelType.MECHANICAL, 6.0));

        SteadyStateSolver.solve(model);

        assertEquals(3.0, shaft.getValue(ModelType.MECHANICAL), TIGHT,
                "ω_shaft should equal τ/G = 6/2 = 3.0");
    }

    /**
     * Same as above but the torque source drives the shaft backwards (negative τ).
     * Verifies the solver handles negative RHS correctly.
     *
     *   ω = −4 / 2 = −2.0
     */
    @Test
    void singleDamper_negativeAppliedTorque() {
        SimulationModel model = new SimulationModel();

        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 0.0));
        Node shaft  = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new LinearLink(ground, shaft, ModelType.MECHANICAL, 2.0));
        model.addComponent(new Source(shaft,ModelType.MECHANICAL, -4.0));

        SteadyStateSolver.solve(model);

        assertEquals(-2.0, shaft.getValue(ModelType.MECHANICAL), TIGHT);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2. TWO DAMPERS — speed divider
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Fixed-speed driver, two dampers in series to ground — mechanical
     * equivalent of a resistor voltage divider.
     *
     *   [input ω=10] --G1=3-- [mid ?] --G2=1-- [ground ω=0]
     *
     * KCL at mid:  G1·(ω_input − ω_mid) = G2·ω_mid
     *              3·(10 − ω_mid)        = ω_mid
     *              ω_mid = 30 / 4 = 7.5
     */
    @Test
    void twoDampers_speedDivider() {
        SimulationModel model = new SimulationModel();

        Node input  = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node mid    = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL,  0.0));

        model.addComponent(new LinearLink(input,  mid,    ModelType.MECHANICAL, 3.0));
        model.addComponent(new LinearLink(mid,    ground, ModelType.MECHANICAL, 1.0));

        SteadyStateSolver.solve(model);

        assertEquals(7.5, mid.getValue(ModelType.MECHANICAL), TIGHT);
    }

    /**
     * Three unknown shafts between a driver and ground.
     * Verifies matrix assembly is correct for larger systems.
     *
     *   [ω=12] --G=4-- [a?] --G=3-- [b?] --G=2-- [c?] --G=6-- [ω=0]
     *
     * Conductance network: equivalent single G = 1/(1/4+1/3+1/2+1/6) = 1.0
     * Torque through chain: τ = G_eq · 12 = 12
     * ω_a = 12 − τ/4 = 9,  ω_b = 9 − τ/3 = 5,  ω_c = 5 − τ/2 = −1
     *
     * Cross-check: τ = 6·(−1 − 0) = −6... let's just verify via the
     * torque-divider result directly.
     *
     * Correct calculation (series conductances, τ constant through chain):
     *   τ = Δω_total / (1/G1 + 1/G2 + 1/G3 + 1/G4) = 12 / (0.25+0.33+0.5+0.17) = 9.6
     *   ω_a = 12 − 9.6/4 = 9.6
     *   ω_b = 9.6 − 9.6/3 = 6.4
     *   ω_c = 6.4 − 9.6/2 = 1.6
     */
    @Test
    void threeDampers_chainSpeedDrop() {
        SimulationModel model = new SimulationModel();

        Node driver = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 12.0));
        Node a      = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node b      = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node c      = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL,  0.0));

        model.addComponent(new LinearLink(driver, a,      ModelType.MECHANICAL, 4.0));
        model.addComponent(new LinearLink(a,      b,      ModelType.MECHANICAL, 3.0));
        model.addComponent(new LinearLink(b,      c,      ModelType.MECHANICAL, 2.0));
        model.addComponent(new LinearLink(c,      ground, ModelType.MECHANICAL, 6.0));

        SteadyStateSolver.solve(model);

        // torque through chain: τ = 12 / (1/4 + 1/3 + 1/2 + 1/6) = 12/1.25 = 9.6
        double tau = 9.6;
        assertEquals(12.0 - tau / 4.0, a.getValue(ModelType.MECHANICAL), LOOSE);
        assertEquals(12.0 - tau / 4.0 - tau / 3.0, b.getValue(ModelType.MECHANICAL), LOOSE);
        assertEquals(12.0 - tau / 4.0 - tau / 3.0 - tau / 2.0, c.getValue(ModelType.MECHANICAL), LOOSE);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 3. GEAR — kinematic ratio enforcement
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Single gear pair, input shaft fixed, output shaft unknown.
     * No external load — only the gear constraint.
     *
     *   [ω_a = 10] --Gear r=3-- [ω_b ?]
     *
     * Constraint:  ω_b = r · ω_a = 30
     * Reaction torque λ = 0 (no load on output)
     */
    @Test
    void gear_ratioEnforced_noLoad() {
        SimulationModel model = new SimulationModel();

        Node shaftA = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node shaftB = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new Gear(shaftA, shaftB, 3.0));

        SteadyStateSolver.solve(model);

        assertEquals(30.0, shaftB.getValue(ModelType.MECHANICAL), LOOSE,
                "Output shaft must spin at r·ω_in = 3 × 10 = 30");
    }

    /**
     * Gear with a damper load on the output shaft.
     * The output speed must still satisfy the kinematic constraint,
     * regardless of the load torque.
     *
     *   [ω_a = 10] --Gear r=2-- [ω_b ?] --G=5-- [ground ω=0]
     *
     * Constraint still forces:  ω_b = 2 · 10 = 20
     * The damper torque (= 5·20 = 100) is absorbed by the gear reaction.
     */
    @Test
    void gear_ratioEnforced_withDamperLoad() {
        SimulationModel model = new SimulationModel();

        Node shaftA = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node shaftB = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL,   0.0));

        model.addComponent(new Gear(shaftA, shaftB, 2.0));
        model.addComponent(new LinearLink(shaftB, ground, ModelType.MECHANICAL, 5.0));

        SteadyStateSolver.solve(model);

        assertEquals(20.0, shaftB.getValue(ModelType.MECHANICAL), LOOSE,
                "Gear constraint must hold even under load: ω_b = 2 × 10 = 20");
    }

    /**
     * Speed-reduction gear (r < 1).
     *
     *   [ω_a = 100] --Gear r=0.1-- [ω_b ?]
     *
     * ω_b = 0.1 × 100 = 10
     */
    @Test
    void gear_speedReduction() {
        SimulationModel model = new SimulationModel();

        Node shaftA = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 100.0));
        Node shaftB = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new Gear(shaftA, shaftB, 0.1));

        SteadyStateSolver.solve(model);

        assertEquals(10.0, shaftB.getValue(ModelType.MECHANICAL), LOOSE);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 4. GEAR CHAIN — compound ratio
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Three-stage gear chain.
     *
     *   [ω_A = 1] --r=2-- [ω_B ?] --r=3-- [ω_C ?] --r=5-- [ω_D ?]
     *
     * ω_D = r1 · r2 · r3 · ω_A = 2 · 3 · 5 · 1 = 30
     *
     * This is the key test that would fail with compliant (penalty) gears
     * due to accumulated ratio error across stages.
     */
    @Test
    void gearChain_compoundRatio() {
        SimulationModel model = new SimulationModel();

        Node shaftA = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 1.0));
        Node shaftB = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node shaftC = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node shaftD = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new Gear(shaftA, shaftB, 2.0));
        model.addComponent(new Gear(shaftB, shaftC, 3.0));
        model.addComponent(new Gear(shaftC, shaftD, 5.0));

        SteadyStateSolver.solve(model);

        assertEquals( 2.0, shaftB.getValue(ModelType.MECHANICAL), LOOSE);
        assertEquals( 6.0, shaftC.getValue(ModelType.MECHANICAL), LOOSE);
        assertEquals(30.0, shaftD.getValue(ModelType.MECHANICAL), LOOSE,
                "Compound gear ratio must be exact — 2 × 3 × 5 × 1 = 30");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 5. THERMAL COUPLING — heat injected by damper dissipation
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Damper between two fixed-speed shafts, heat node connected to a
     * thermal ground.  Verifies the P = G·Δω² coupling between domains.
     *
     *   Mechanical:  [ω=10] --G=2-- [ω=4]   →   Δω = 6,  P = 2·36 = 72 W
     *   Thermal:     [T ?] --G_th=1-- [T=0]  →   T = P / G_th = 72 K
     *
     * The thermal domain is linear (AX=B), so it converges in one outer
     * iteration once P is injected from the mechanical side.
     */
    @Test
    void rotationalDamper_injectsThermalPower() {
        SimulationModel model = new SimulationModel();

        // mechanical — both shafts fixed so Δω is known exactly
        Node shaftA   = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node shaftB   = model.addNode(new FixedValueNode(ModelType.MECHANICAL,  4.0));

        // thermal
        Node heatNode = model.addNode(new UnknownNode  (ModelType.THERMAL));
        Node tGround  = model.addNode(new FixedValueNode(ModelType.THERMAL,  0.0));

        model.addComponent(new RotationalDamper(shaftA, shaftB, heatNode, 2.0));
        model.addComponent(new LinearLink(heatNode, tGround, ModelType.THERMAL, 1.0));

        SteadyStateSolver.solve(model);

        // P = G·Δω² = 2·(10−4)² = 72 W
        // T_heat = P / G_thermal = 72 / 1 = 72 K
        assertEquals(72.0, heatNode.getValue(ModelType.THERMAL), LOOSE,
                "Heat node temperature must equal P/G_th = 72 K");
    }

    /**
     * Same topology but verifies power scales as Δω² not Δω
     * by halving the speed difference and confirming power drops by 4×.
     *
     *   Δω = 3  →  P = 2·9 = 18  →  T = 18 K  (was 72 when Δω = 6)
     */
    @Test
    void rotationalDamper_powerIsQuadraticInSpeedDifference() {
        SimulationModel model = new SimulationModel();

        Node shaftA   = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 7.0));
        Node shaftB   = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 4.0));
        Node heatNode = model.addNode(new UnknownNode  (ModelType.THERMAL));
        Node tGround  = model.addNode(new FixedValueNode(ModelType.THERMAL,  0.0));

        model.addComponent(new RotationalDamper(shaftA, shaftB, heatNode, 2.0));
        model.addComponent(new LinearLink(heatNode, tGround, ModelType.THERMAL, 1.0));

        SteadyStateSolver.solve(model);

        assertEquals(18.0, heatNode.getValue(ModelType.THERMAL), LOOSE,
                "P = G·Δω² = 2·9 = 18 W, T = 18/1 = 18 K");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 6. GEAR + DAMPER + THERMAL — full multi-physics chain
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Full chain: gear drives a damped output shaft, damper dissipation
     * heats a thermal node.
     *
     *   [ω_in=10] --Gear r=3-- [ω_out ?] --G=4-- [ground ω=0]
     *                                       │
     *                                  RotationalDamper(G=4) → [T_heat ?] --G_th=2-- [T=0]
     *
     * Step 1 (mechanical): ω_out = 3·10 = 30  (gear enforces ratio)
     * Step 2 (thermal):    P = 4·(30−0)² = 3600 W
     *                      T_heat = P / G_th = 3600 / 2 = 1800 K
     */
    @Test
    void gearAndDamper_fullMultiPhysics() {
        SimulationModel model = new SimulationModel();

        Node shaftIn  = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node shaftOut = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node ground   = model.addNode(new FixedValueNode(ModelType.MECHANICAL,  0.0));
        Node heatNode = model.addNode(new UnknownNode  (ModelType.THERMAL));
        Node tGround  = model.addNode(new FixedValueNode(ModelType.THERMAL,     0.0));

        model.addComponent(new Gear(shaftIn, shaftOut, 3.0));
        model.addComponent(new RotationalDamper(shaftOut, ground, heatNode, 4.0));
        model.addComponent(new LinearLink(heatNode, tGround, ModelType.THERMAL, 2.0));

        SteadyStateSolver.solve(model);

        assertEquals(30.0,   shaftOut.getValue(ModelType.MECHANICAL), LOOSE, "ω_out = r·ω_in = 30");
        assertEquals(1800.0, heatNode.getValue(ModelType.MECHANICAL), LOOSE, "T = G·Δω²/G_th = 4·900/2 = 1800");
    }
}
