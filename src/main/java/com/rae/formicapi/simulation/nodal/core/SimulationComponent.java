package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.ModelType;

import java.util.List;
import java.util.Map;
import java.util.Set;

public interface SimulationComponent {
    Set<ModelType> getDomains();
    List<Node> getInternalNodes();
    void stamp(Map<ModelType, SimulationContext> contexts);
}
