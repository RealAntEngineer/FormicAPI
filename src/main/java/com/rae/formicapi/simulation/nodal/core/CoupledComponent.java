package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

public interface CoupledComponent {
    PhysicsType getSourceDomain();
    PhysicsType getSinkDomain();
    void stampCoupling(SimulationContext source, SimulationContext sink);
}
