package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.ModelType;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

public class SimulationModel {
    private static final List<ModelType> SOLVE_ORDER = List.of(ModelType.MECHANICAL, ModelType.HYDRAULIC, ModelType.THERMAL);

    private final Map<ModelType, DomainModel> domains = new EnumMap<>(ModelType.class);
    private final List<SimulationComponent> components = new ArrayList<>();

    public DomainModel domain(ModelType type) {
        return domains.computeIfAbsent(type, DomainModel::new);
    }

    public Node addNode(Node node) {
        node.getDomains().forEach(type -> domain(type).addNode(node));
        return node;
    }

    public void addComponent(SimulationComponent coupling) {
        components.add(coupling);
        coupling.getInternalNodes().forEach(this::addNode);
    }

    public Map<ModelType, DomainModel> getDomains() { return domains; }
    public List<SimulationComponent> getComponents()      { return components; }

    public List<ModelType> getSolveOrder() {
        return SOLVE_ORDER;
    }
}