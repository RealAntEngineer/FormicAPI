package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

import java.util.Map;
import java.util.Set;

public interface SimulationComponent {
    Set<PhysicsType> getDomains();
    void stamp(Map<PhysicsType, SimulationContext> contexts);
}
