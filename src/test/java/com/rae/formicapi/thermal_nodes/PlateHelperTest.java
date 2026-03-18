package com.rae.formicapi.thermal_nodes;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.FixedValueNode;
import com.rae.formicapi.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.simulation.nodal.core.LinearLink;
import com.rae.formicapi.simulation.nodal.thermal.Convection;
import com.rae.formicapi.simulation.nodal.thermal.PlateNodeHelper;
import com.rae.formicapi.simulation.nodal.core.SteadyStateSolver;
import com.rae.formicapi.simulation.physical.material.Material;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static com.rae.formicapi.thermal_nodes.Helper.savePlateHeatmap;

public class PlateHelperTest {
    @Test
    public void multiLayerPlateTest() {

        SimulationModel model = new SimulationModel();
        PlateNodeHelper helper = new PlateNodeHelper();

        helper.addLayer(new PlateNodeHelper.Layer(Material.STEEL, 0.5,0.05, 100, 10));
        helper.addLayer(new PlateNodeHelper.Layer(Material.COPPER, 0.5,0.1, 100, 3));

        UnknownNode[][] nodes = helper.createPlateNodes(model);

        int Nx = nodes.length;
        int Ny = nodes[0].length;

        FixedValueNode ambient = new FixedValueNode(PhysicsType.THERMAL,25);
        model.addNode(ambient);

        FixedValueNode hotPlate = new FixedValueNode(PhysicsType.THERMAL,120);
        model.addNode(hotPlate);

        double h = 10;

        // Heated bottom
        for (UnknownNode[] node : nodes) {
            model.addComponent(PhysicsType.THERMAL,new LinearLink(hotPlate, node[0], 20));
        }

        // Convection elsewhere
        for (UnknownNode[] node : nodes) {
            model.addComponent(PhysicsType.THERMAL,new Convection(node[Ny - 1], ambient, h));
        }

        for (int j = 0; j < Ny; j++) {
            model.addComponent(PhysicsType.THERMAL,new Convection(nodes[0][j], ambient, h));
            model.addComponent(PhysicsType.THERMAL,new Convection(nodes[Nx - 1][j], ambient, h));
        }

        SteadyStateSolver.solve(model);

        savePlateHeatmap(nodes, new ArrayList<>(helper.getLayers()),"multi_layer_plate.png");
    }
}