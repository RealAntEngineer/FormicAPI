package com.rae.formicapi.thermal_nodes;

import com.rae.formicapi.simulation.nodal.core.FixedValueNode;
import com.rae.formicapi.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.simulation.nodal.thermal.Conduction;
import com.rae.formicapi.simulation.nodal.thermal.Convection;
import com.rae.formicapi.simulation.nodal.thermal.PlateNodeHelper;
import com.rae.formicapi.simulation.nodal.thermal.SteadyStateSolver;
import org.junit.jupiter.api.Test;

import static com.rae.formicapi.Helper.savePlateHeatmap;

public class PlateHelperTest {
    @Test
    public void multiLayerPlateTest() {

        SimulationModel model = new SimulationModel();
        PlateNodeHelper helper = new PlateNodeHelper();

        helper.addLayer(new PlateNodeHelper.Layer(PlateNodeHelper.Material.ALUMINUM, 1,0.005, 10, 3));
        helper.addLayer(new PlateNodeHelper.Layer(PlateNodeHelper.Material.STEEL, 1,0.01, 10, 4));

        UnknownNode[][] nodes = helper.createPlateNodes(model);

        int Nx = nodes.length;
        int Ny = nodes[0].length;

        FixedValueNode ambient = new FixedValueNode(25);
        model.addNode(ambient);

        FixedValueNode hotPlate = new FixedValueNode(120);
        model.addNode(hotPlate);

        double h = 10;

        // Heated bottom
        for (UnknownNode[] node : nodes) {
            model.addComponent(new Conduction(hotPlate, node[0], 20));
        }

        // Convection elsewhere
        for (UnknownNode[] node : nodes) {
            model.addComponent(new Convection(node[Ny - 1], ambient, h));
        }

        for (int j = 0; j < Ny; j++) {
            model.addComponent(new Convection(nodes[0][j], ambient, h));
            model.addComponent(new Convection(nodes[Nx - 1][j], ambient, h));
        }

        SteadyStateSolver.solve(model);

        savePlateHeatmap(nodes, "multi_layer_plate.png");
    }
}