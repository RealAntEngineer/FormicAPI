package com.rae.formicapi.simulation.physical.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.simulation.nodal.core.LinearLink;
import com.rae.formicapi.simulation.physical.ports.ThermalPort;

public class PortConnector {

    public static void connectThermal(
            SimulationModel model,
            ThermalPort a,
            ThermalPort b,
            double conductance
    ) {
        model.addComponent(new LinearLink(
                a.getNode(),
                b.getNode(),
                PhysicsType.THERMAL,
                conductance
        ));
    }

    /*public static void connectMechanical(
            SimulationModel model,
            MechanicalPort a,
            MechanicalPort b,
            double stiffness // or conductance equivalent
    ) {
        model.addComponent(new RotationalDamping(
                a.getNode(),
                b.getNode(),
                stiffness
        ));
    }*/
}
