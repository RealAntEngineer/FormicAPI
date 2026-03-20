package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

public abstract class SingleDomainComponent implements SimulationComponent {
    public abstract ModelType getDomain();
    public abstract void stamp(SimulationContext ctx);

    @Override
    public final Set<ModelType> getDomains() {
        return EnumSet.of(getDomain());
    }

    @Override
    public final void stamp(Map<ModelType, SimulationContext> contexts) {
        SimulationContext ctx = contexts.get(getDomain());
        if (ctx != null) stamp(ctx);
    }
}