package com.rae.formicapi.simulation.physical.core;

import com.rae.formicapi.simulation.nodal.core.SimulationModel;

import java.util.List;

public abstract class Part {

    /**
     * Build this part into the simulation model
     */
    public abstract void build(SimulationModel model);

    /**
     * External interaction points
     */
    public abstract List<Port> getPorts();
}
