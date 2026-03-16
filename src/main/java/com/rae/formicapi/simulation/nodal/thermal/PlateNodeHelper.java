package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.simulation.nodal.core.UnknownNode;

import java.util.ArrayList;
import java.util.List;

public class PlateNodeHelper {

    public static class Layer {
        public Material material;
        public double thickness; // meters, Y-direction
        public double length;    // meters, X-direction
        public int nx;           // nodes along length
        public int ny;           // nodes along thickness

        public Layer(Material material, double length, double thickness, int nx, int ny) {
            this.material = material;
            this.length = length;
            this.thickness = thickness;
            this.nx = nx;
            this.ny = ny;
        }
    }

    private List<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    /**
     * Create nodes for all layers, register in the model,
     * and add conduction between neighbors based on geometry and material.
     */
    public UnknownNode[][] createPlateNodes(SimulationModel model) {
        // Compute total dimensions
        int totalNx = layers.get(0).nx;
        int totalNy = 0;
        for (Layer layer : layers) totalNy += layer.ny;

        UnknownNode[][] nodes = new UnknownNode[totalNx][totalNy];
        int currentY = 0;

        for (Layer layer : layers) {
            for (int j = 0; j < layer.ny; j++) {
                for (int i = 0; i < layer.nx; i++) {
                    nodes[i][currentY + j] = new UnknownNode( 25); // start at ambient
                    model.addNode(nodes[i][currentY + j]);
                }
            }
            currentY += layer.ny;
        }

        // Add conduction between neighbors
        currentY = 0;
        for (Layer layer : layers) {
            double dx = layer.length / layer.nx;
            double dy = layer.thickness / layer.ny;

            for (int j = 0; j < layer.ny; j++) {
                for (int i = 0; i < layer.nx; i++) {
                    UnknownNode node = nodes[i][currentY + j];

                    // Right neighbor
                    if (i < layer.nx - 1) {
                        double Gx = layer.material.getConductivity() * dy / dx;
                        model.addComponent(new Conduction(node, nodes[i + 1][currentY + j], Gx));
                    }

                    // Top neighbor
                    if (j < layer.ny - 1) {
                        double Gy = layer.material.getConductivity() * dx / dy;
                        model.addComponent(new Conduction(node, nodes[i][currentY + j + 1], Gy));
                    }
                }
            }
            currentY += layer.ny;
        }

        return nodes;
    }

    public enum Material {
        ALUMINUM(205),
        COPPER(385),
        STEEL(50),
        WATER(0.6),
        AIR(0.025);

        private final double conductivity; // W/m·K

        Material(double conductivity) {
            this.conductivity = conductivity;
        }

        public double getConductivity() {
            return conductivity;
        }
    }
}
