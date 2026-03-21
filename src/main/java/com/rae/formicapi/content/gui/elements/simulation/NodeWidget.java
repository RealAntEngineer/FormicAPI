package com.rae.formicapi.content.gui.elements.simulation;

import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import net.createmod.catnip.gui.widget.AbstractSimiWidget;

public class NodeWidget extends AbstractSimiWidget {

    Node node;
    protected NodeWidget(Node node, int x, int y) {
        super(x, y);
        this.node = node;
    }
}
