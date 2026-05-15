package com.rae.formicapi.content.gui.elements.simulation;

import com.rae.formicapi.content.gui.elements.CompoundWidget;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent;
import net.minecraft.client.gui.GuiGraphics;


public class SimulationComponentWidget extends CompoundWidget {


    public SimulationComponentWidget(SimulationComponent simulationComponent, int x, int y) {
        super(x, y);

        //this is a bit stupid, but we will make something smarter later
        int i = 0;
        for (Node node : simulationComponent.getInterfaceNodes()) {

            addWidget(new NodeWidget(node, x + i * 10, y));
            i++;
        }
    }

    //we need to have clickable widgets to create link bwn components.


    @Override
    protected void renderWidget(GuiGraphics graphics, int mouseX, int mouseY, float partialTicks) {
        super.renderWidget(graphics, mouseX, mouseY, partialTicks);
    }
}