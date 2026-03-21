package com.rae.formicapi.content.gui.elements.simulation;

import com.rae.formicapi.content.gui.elements.CompoundWidget;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent;


public class SimulationComponentWidget extends CompoundWidget {
    protected SimulationComponentWidget(SimulationComponent simulationComponent, int x, int y) {
        super(x, y);


        simulationComponent.getInternalNodes();
    }



}
