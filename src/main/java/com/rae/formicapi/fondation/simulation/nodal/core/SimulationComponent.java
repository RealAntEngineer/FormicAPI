package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;

import java.util.List;
import java.util.Map;
import java.util.Set;

public interface SimulationComponent {
    Set<ModelType> getDomains();

    List<Node> getInternalNodes();

    void stamp(Map<ModelType, SimulationContext> contexts);
}
