package com.rae.formicapi.simulation.nodal.complex;

import com.rae.formicapi.simulation.nodal.ModelType;
import com.rae.formicapi.simulation.nodal.core.Node;
import com.rae.formicapi.simulation.nodal.core.SimulationComponent;
import com.rae.formicapi.simulation.nodal.core.SimulationContext;

import java.util.List;
import java.util.Map;
import java.util.Set;




public class HeatExchanger implements SimulationComponent {



    @Override
    public Set<ModelType> getDomains() {
        return Set.of(ModelType.HYDRAULIC, ModelType.THERMAL);
    }

    @Override
    public List<Node> getInternalNodes() {
        return List.of();
    }

    @Override
    public void stamp(Map<ModelType, SimulationContext> contexts) {

    }
}
