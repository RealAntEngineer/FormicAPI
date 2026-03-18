package com.rae.formicapi.thermal_nodes;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.FixedValueNode;
import com.rae.formicapi.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.simulation.nodal.core.LinearLink;
import com.rae.formicapi.simulation.nodal.thermal.PlateNodeHelper;
import com.rae.formicapi.simulation.nodal.core.SteadyStateSolver;
import com.rae.formicapi.simulation.nodal.material.Material;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static com.rae.formicapi.thermal_nodes.Helper.savePlateHeatmap;

public class PlateHelperTest {
    @Test
    public void multiLayerPlateTestFine() {

        SimulationModel model = new SimulationModel();
        PlateNodeHelper helper = new PlateNodeHelper();

        helper.addLayer(new PlateNodeHelper.Layer(Material.STEEL, 0.5,0.05, 100, 30));
        helper.addLayer(new PlateNodeHelper.Layer(Material.COPPER, 0.5,0.1, 100, 20));

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
            model.addComponent(new LinearLink(hotPlate, node[0],PhysicsType.THERMAL, 20));
        }

        // Convection elsewhere
        for (UnknownNode[] node : nodes) {
            model.addComponent(new LinearLink(node[Ny - 1], ambient, PhysicsType.THERMAL, h));
        }

        for (int j = 0; j < Ny; j++) {
            model.addComponent(new LinearLink(nodes[0][j], ambient, PhysicsType.THERMAL, h));
            model.addComponent(new LinearLink(nodes[Nx - 1][j], ambient, PhysicsType.THERMAL, h));
        }

        SteadyStateSolver.solve(model);

        savePlateHeatmap(nodes, new ArrayList<>(helper.getLayers()),"multi_layer_plate_fine.png");
    }


    @Test
    public void multiLayerPlateTest() {

        SimulationModel model = new SimulationModel();
        PlateNodeHelper helper = new PlateNodeHelper();

        helper.addLayer(new PlateNodeHelper.Layer(Material.STEEL, 0.5,0.05, 100, 10));
        helper.addLayer(new PlateNodeHelper.Layer(Material.COPPER, 0.5,0.1, 100, 5));

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
            model.addComponent(new LinearLink(hotPlate, node[0],PhysicsType.THERMAL, 20));
        }

        // Convection elsewhere
        for (UnknownNode[] node : nodes) {
            model.addComponent(new LinearLink(node[Ny - 1], ambient, PhysicsType.THERMAL, h));
        }

        for (int j = 0; j < Ny; j++) {
            model.addComponent(new LinearLink(nodes[0][j], ambient, PhysicsType.THERMAL, h));
            model.addComponent(new LinearLink(nodes[Nx - 1][j], ambient, PhysicsType.THERMAL, h));
        }

        SteadyStateSolver.solve(model);

        savePlateHeatmap(nodes, new ArrayList<>(helper.getLayers()),"multi_layer_plate.png");
    }


    @Test
    public void multiLayerPlateTestRough() {

        SimulationModel model = new SimulationModel();
        PlateNodeHelper helper = new PlateNodeHelper();

        helper.addLayer(new PlateNodeHelper.Layer(Material.STEEL, 0.5,0.05, 10, 2));
        helper.addLayer(new PlateNodeHelper.Layer(Material.COPPER, 0.5,0.1, 10, 2));

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
            model.addComponent(new LinearLink(hotPlate, node[0],PhysicsType.THERMAL, 20));
        }

        // Convection elsewhere
        for (UnknownNode[] node : nodes) {
            model.addComponent(new LinearLink(node[Ny - 1], ambient, PhysicsType.THERMAL, h));
        }

        for (int j = 0; j < Ny; j++) {
            model.addComponent(new LinearLink(nodes[0][j], ambient, PhysicsType.THERMAL, h));
            model.addComponent(new LinearLink(nodes[Nx - 1][j], ambient, PhysicsType.THERMAL, h));
        }

        SteadyStateSolver.solve(model);

        savePlateHeatmap(nodes, new ArrayList<>(helper.getLayers()),"multi_layer_plate_rough.png");
    }
}