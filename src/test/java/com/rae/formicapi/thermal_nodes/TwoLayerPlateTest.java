package com.rae.formicapi.thermal_nodes;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.core.FixedValueNode;
import com.rae.formicapi.fondation.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.fondation.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.fondation.simulation.nodal.core.LinearLink;
import com.rae.formicapi.fondation.simulation.nodal.SteadyStateSolver;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static com.rae.formicapi.thermal_nodes.Helper.savePlateHeatmap;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TwoLayerPlateTest {

    @Test()
    public void twoLayerPlateWithConvection() {

        SimulationModel model = new SimulationModel();



        // Unknown node
        UnknownNode layer1 = new UnknownNode(ModelType.THERMAL);

        // Boundary nodes
        FixedValueNode hot = new FixedValueNode(ModelType.THERMAL,100);
        FixedValueNode ambient = new FixedValueNode(ModelType.THERMAL,25);

        // Register nodes in the model
        model.addNode(layer1);
        model.addNode(hot);
        model.addNode(ambient);

        // Parameters
        double G = 5;
        double h = 10;

        // Physics
        model.addComponent(new LinearLink(hot, layer1, ModelType.THERMAL,G));
        model.addComponent(new LinearLink(layer1, ambient, ModelType.THERMAL,h));

        // Solve
        SteadyStateSolver.solve(model);

        // Analytical solution
        double T1_expected = (100 * G + 25 * h) / (G + h);

        assertEquals(T1_expected, layer1.getValue(ModelType.THERMAL), 1e-6);
    }

    @Test
    public void twoDPlateWithNonUniformLimits() {

        SimulationModel model = new SimulationModel();

        int Nx = 3; // columns
        int Ny = 2; // rows
        UnknownNode[][] nodes = new UnknownNode[Nx][Ny];

        // Create unknown nodes
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                nodes[i][j] = new UnknownNode(ModelType.THERMAL); // unique ID
                model.addNode(nodes[i][j]);
            }
        }

        // Boundary nodes
        FixedValueNode hotLeft = new FixedValueNode(ModelType.THERMAL,100);
        FixedValueNode coldRight = new FixedValueNode(ModelType.THERMAL,25);
        model.addNode(hotLeft);
        model.addNode(coldRight);

        // Add conduction between neighbors (non-uniform G)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                UnknownNode node = nodes[i][j];

                // Right neighbor
                if (i < Nx - 1) {
                    double Gx = 5 + i; // non-uniform across X
                    model.addComponent(new LinearLink(node, nodes[i + 1][j], ModelType.THERMAL,Gx));
                }

                // Top neighbor
                if (j < Ny - 1) {
                    double Gy = 10 + j; // non-uniform across Y
                    model.addComponent(new LinearLink(node, nodes[i][j + 1], ModelType.THERMAL,Gy));
                }

                // Connect leftmost nodes to hot boundary
                if (i == 0) {
                    double G_left = 8; // conduction to hot side
                    model.addComponent(new LinearLink(hotLeft, node, ModelType.THERMAL,G_left));
                }

                // Connect rightmost nodes to cold boundary via convection
                if (i == Nx - 1) {
                    double h_local = 5 + j; // non-uniform convection
                    model.addComponent(new LinearLink(node, coldRight, ModelType.THERMAL,h_local));
                }
            }
        }

        // Solve the system
        SteadyStateSolver.solve(model);

        // Simple assertion: temperatures should be between hot and cold
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double T = nodes[i][j].getValue(ModelType.THERMAL);
                assertTrue(T >= 25 && T <= 100, "Temperature out of bounds");
            }
        }
    }

    @Test
    public void twoDPlateWithHotSpot() {

        SimulationModel model = new SimulationModel();

        int Nx = 51; // columns
        int Ny = 31; // rows
        UnknownNode[][] nodes = new UnknownNode[Nx][Ny];

        // Create unknown nodes
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                nodes[i][j] = new UnknownNode(ModelType.THERMAL); // start at ambient
                model.addNode(nodes[i][j]);
            }
        }

        // Convection boundary nodes (ambient at all edges)
        FixedValueNode ambient = new FixedValueNode(ModelType.THERMAL,25);
        model.addNode(ambient);
        double h = 10; // uniform convection coefficient

        // Connect all edge nodes to ambient via convection
        for (int i = 0; i < Nx; i++) {
            model.addComponent(new LinearLink(nodes[i][0], ambient, ModelType.THERMAL, h));       // bottom row
            model.addComponent(new LinearLink(nodes[i][Ny - 1], ambient, ModelType.THERMAL, h));  // top row
        }
        for (int j = 0; j < Ny; j++) {
            model.addComponent(new LinearLink(nodes[0][j], ambient, ModelType.THERMAL, h));       // left column
            model.addComponent(new LinearLink(nodes[Nx - 1][j], ambient, ModelType.THERMAL, h));  // right column
        }

        // Add conduction between neighbors (uniform G)
        double G = 50;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                UnknownNode node = nodes[i][j];

                if (i < Nx - 1) model.addComponent(new LinearLink(node, nodes[i + 1][j], ModelType.THERMAL,G));
                if (j < Ny - 1) model.addComponent(new LinearLink(node, nodes[i][j + 1], ModelType.THERMAL,G));
            }
        }

        // Add hot spot in the center
        int cx = Nx / 2;
        int cy = Ny / 2;
        FixedValueNode hotSpot = new FixedValueNode( ModelType.THERMAL,100);
        model.addNode(hotSpot);
        model.addComponent(new LinearLink(hotSpot, nodes[cx][cy], ModelType.THERMAL,G));

        // Solve
        SteadyStateSolver.solve(model);

        // Print results
        savePlateHeatmap(nodes, new ArrayList<>(),"2d_plate_heatmap.png");

        // Basic sanity check: temperatures should be between ambient and hot spot
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double T = nodes[i][j].getValue(ModelType.THERMAL);
                assertTrue(T >= 25 && T <= 100, "Temperature out of bounds");
            }
        }
    }

    @Test
    public void radiatorBladeSimulation() {

        SimulationModel model = new SimulationModel();

        // Blade dimensions (Nx = along length, Ny = along thickness)
        int Nx = 50; // lengthwise nodes
        int Ny = 5;  // thickness nodes
        UnknownNode[][] nodes = new UnknownNode[Nx][Ny];

        // Create unknown nodes
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                nodes[i][j] = new UnknownNode(ModelType.THERMAL); // start at ambient
                model.addNode(nodes[i][j]);
            }
        }

        // Conduction along neighbors (uniform)
        double Gx = 50; // along length
        double Gy = 2; // along thickness
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                UnknownNode node = nodes[i][j];
                if (i < Nx - 1) model.addComponent(new LinearLink(node, nodes[i + 1][j], ModelType.THERMAL,Gx));
                if (j < Ny - 1) model.addComponent(new LinearLink(node, nodes[i][j + 1], ModelType.THERMAL,Gy));
            }
        }

        // Hot base (left edge connected to engine or heat source)
        FixedValueNode hotBase = new FixedValueNode(ModelType.THERMAL,100);
        model.addNode(hotBase);
        for (int j = 0; j < Ny; j++) {
            model.addComponent(new LinearLink(hotBase, nodes[0][j], ModelType.THERMAL,Gx));
        }

        // Convection to ambient on exposed surfaces (right edge + top/bottom surfaces)
        FixedValueNode ambient = new FixedValueNode(ModelType.THERMAL,25);
        model.addNode(ambient);
        double h = 5; // uniform convection coefficient

        // Right edge convection
        for (int j = 0; j < Ny; j++) {
            model.addComponent(new LinearLink(nodes[Nx - 1][j], ambient, ModelType.THERMAL,h));
        }

        // Top and bottom surfaces convection
        for (int i = 0; i < Nx; i++) {
            model.addComponent(new LinearLink(nodes[i][0], ambient, ModelType.THERMAL,h));       // bottom
            model.addComponent(new LinearLink(nodes[i][Ny - 1], ambient, ModelType.THERMAL,h));  // top
        }

        // Solve
        SteadyStateSolver.solve(model);


        // Save heatmap image
        savePlateHeatmap(nodes, new ArrayList<>(), "radiator_blade.png");

        // Basic sanity check
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double T = nodes[i][j].getValue(ModelType.THERMAL);
                assertTrue(T >= 25 && T <= 100, "Temperature out of bounds");
            }
        }
    }



}