package com.rae.formicapi.mechanical_nodes;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.SteadyStateSolver;
import com.rae.formicapi.fondation.simulation.nodal.core.*;
import com.rae.formicapi.fondation.simulation.nodal.linear.mechanical.Gear;
import com.rae.formicapi.fondation.simulation.nodal.linear.mechanical.RotationalDamper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;


public class MechanicalNetworkTests {

    static final double TIGHT = 1e-6;   // for exact/linear results
    static final double LOOSE = 1e-3;   // for converged NR results

    @Test
    public void singleDamperSteadyStateSpeed() {
        SimulationModel model = new SimulationModel();

        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 0.0));
        Node shaft  = model.addNode(new UnknownNode(ModelType.MECHANICAL));

        model.addComponent(new LinearLink(ground, shaft, ModelType.MECHANICAL, 2.0));
        model.addComponent(new Source(shaft,ModelType.MECHANICAL, 6.0));

        SteadyStateSolver.solve(model);

        assertEquals(3.0, shaft.getValue(ModelType.MECHANICAL), TIGHT,
                "ω_shaft should equal τ/G = 6/2 = 3.0");
    }


    @Test
    public void singleDamperNegativeAppliedTorque() {
        SimulationModel model = new SimulationModel();

        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 0.0));
        Node shaft  = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new LinearLink(ground, shaft, ModelType.MECHANICAL, 2.0));
        model.addComponent(new Source(shaft,ModelType.MECHANICAL, -4.0));

        SteadyStateSolver.solve(model);

        assertEquals(-2.0, shaft.getValue(ModelType.MECHANICAL), TIGHT);
    }

    @Test
    public void twoDampers_speedDivider() {
        SimulationModel model = new SimulationModel();

        Node input  = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node mid    = model.addNode(new UnknownNode  (ModelType.MECHANICAL));
        Node ground = model.addNode(new FixedValueNode(ModelType.MECHANICAL,  0.0));

        model.addComponent(new LinearLink(input,  mid,    ModelType.MECHANICAL, 3.0));
        model.addComponent(new LinearLink(mid,    ground, ModelType.MECHANICAL, 1.0));

        SteadyStateSolver.solve(model);

        assertEquals(7.5, mid.getValue(ModelType.MECHANICAL), TIGHT);
    }

    @Test
    public void threeDampers_chainSpeedDrop() {
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

    @Test
    public void gearRatioEnforced_noLoad() {
        SimulationModel model = new SimulationModel();

        Node shaftA = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 10.0));
        Node shaftB = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new Gear(shaftA, shaftB, 3.0));

        SteadyStateSolver.solve(model);

        assertEquals(30.0, shaftB.getValue(ModelType.MECHANICAL), LOOSE,
                "Output shaft must spin at r·ω_in = 3 × 10 = 30");
    }

    @Test
    public void gearRatioEnforced_withDamperLoad() {
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

    @Test
    public void gearSpeedReduction() {
        SimulationModel model = new SimulationModel();

        Node shaftA = model.addNode(new FixedValueNode(ModelType.MECHANICAL, 100.0));
        Node shaftB = model.addNode(new UnknownNode  (ModelType.MECHANICAL));

        model.addComponent(new Gear(shaftA, shaftB, 0.1));

        SteadyStateSolver.solve(model);

        assertEquals(10.0, shaftB.getValue(ModelType.MECHANICAL), LOOSE);
    }

    @Test
    public void gearChainCompoundRatio() {
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

    @Test
    public void rotationalDamperInjectsThermalPower() {
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

    @Test
    public void rotationalDamperPowerIsQuadraticInSpeedDifference() {
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

    @Test
    public void gearAndDamperFullMultiPhysics() {
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
        assertEquals(1800.0, heatNode.getValue(ModelType.THERMAL), LOOSE, "T = G·Δω²/G_th = 4·900/2 = 1800");
    }
}
