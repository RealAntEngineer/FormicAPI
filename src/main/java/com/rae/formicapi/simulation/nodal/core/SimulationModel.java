package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

public class SimulationModel {
    private static final List<PhysicsType> SOLVE_ORDER = List.of(PhysicsType.MECHANICAL, PhysicsType.HYDRAULIC, PhysicsType.THERMAL);

    private final Map<PhysicsType, DomainModel> domains = new EnumMap<>(PhysicsType.class);
    private final List<CoupledComponent> couplings = new ArrayList<>();

    public DomainModel domain(PhysicsType type) {
        return domains.computeIfAbsent(type, DomainModel::new);
    }

    public Node addNode(Node node) {
        return domain(node.getDomain()).addNode(node);
    }

    public void addComponent(PhysicsType type, PhysicsComponent component) {
        domain(type).addComponent(component);
    }

    public void addCoupling(CoupledComponent coupling) {
        couplings.add(coupling);
    }

    public Map<PhysicsType, DomainModel> getDomains() { return domains; }
    public List<CoupledComponent> getCouplings()      { return couplings; }

    public List<PhysicsType> getSolveOrder() {
        return SOLVE_ORDER;
    }
}