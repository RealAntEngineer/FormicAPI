package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.PhysicsType;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

public abstract class DomainComponent implements SimulationComponent {
    public abstract PhysicsType getDomain();
    public abstract void stamp(SimulationContext ctx);

    @Override
    public final Set<PhysicsType> getDomains() {
        return EnumSet.of(getDomain());
    }

    @Override
    public final void stamp(Map<PhysicsType, SimulationContext> contexts) {
        SimulationContext ctx = contexts.get(getDomain());
        if (ctx != null) stamp(ctx);
    }
}