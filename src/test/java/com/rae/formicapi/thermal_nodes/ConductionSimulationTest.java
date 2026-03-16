package com.rae.formicapi.thermal_nodes;

import static org.junit.jupiter.api.Assertions.*;

import com.rae.formicapi.simulation.nodal.core.*;
import com.rae.formicapi.simulation.nodal.thermal.*;
import org.junit.jupiter.api.Test;

public class ConductionSimulationTest {

    @Test
    public void twoNodeConduction() {

        SimulationModel model = new SimulationModel();
        UnknownNode a = new UnknownNode(0);
        FixedValueNode b = new FixedValueNode(0);
        model.addNode(a);
        model.addNode(b);

        model.addComponent(new Conduction(a, b, 10));
        model.addComponent(new HeatSource(a, 100));

        SteadyStateSolver.solve(model);

        double delta = a.getValue() - b.getValue();

        assertEquals(10, delta, 1e-6);
    }

    /*@Test
    public void thermalDivider() {

        SimulationModel model = new SimulationModel();

        UnknowNode a = model.addNode(0);
        UnknowNode b = model.addNode(0);
        UnknowNode c = model.addNode(0);

        model.addComponent(new Conduction(a, b, 10));
        model.addComponent(new Conduction(b, c, 10));

        model.addComponent(new HeatSource(a, 100));

        // Fix node C to absolute temperature
        model.addComponent(new FixedTemperature(c, 0));

        SteadyStateSolver.solve(model);

        double d1 = a.getValue() - b.getValue();
        double d2 = b.getValue() - c.getValue();

        assertEquals(d1, d2, 1e-6);
    }

    @Test
    public void fixedTemperatureBoundary() {

        SimulationModel model = new SimulationModel();

        UnknowNode heater = model.addNode(0);
        UnknowNode space = model.addNode(0);

        model.addComponent(new FixedTemperature(space, 0));
        model.addComponent(new Conduction(heater, space, 10));
        model.addComponent(new HeatSource(heater, 100));

        SteadyStateSolver.solve(model);

        assertEquals(10, heater.getValue(), 1e-6);
    }*/
}
