package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.simulation.nodal.PhysicsType;
import com.rae.formicapi.simulation.nodal.core.LinearLink;
import com.rae.formicapi.simulation.nodal.core.SimulationModel;
import com.rae.formicapi.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.simulation.physical.material.Material;

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

    public List<Layer> getLayers() {
        return layers;
    }

    private final List<Layer> layers = new ArrayList<>();

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
                    nodes[i][currentY + j] = new UnknownNode(PhysicsType.THERMAL, 25); // start at ambient
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
                        model.addComponent(PhysicsType.THERMAL,new LinearLink(node, nodes[i + 1][currentY + j], Gx));
                    }

                    // Top neighbor
                    if (j < layer.ny - 1) {
                        double Gy = layer.material.getConductivity() * dx / dy;
                        model.addComponent(PhysicsType.THERMAL,new LinearLink(node, nodes[i][currentY + j + 1], Gy));
                    }
                }
            }
            currentY += layer.ny;
        }
        // Add conduction between layers
        int offsetY = 0;
        for (int l = 0; l < layers.size() - 1; l++) {
            Layer bottom = layers.get(l);
            Layer top    = layers.get(l + 1);

            double dy_bottom = bottom.thickness / bottom.ny;
            double dy_top    = top.thickness    / top.ny;
            double dx        = bottom.length    / bottom.nx; // nx must match

            // Interface conductance: harmonic mean of the two half-cells
            double G_bottom = bottom.material.getConductivity() * dx / (dy_bottom / 2.0);
            double G_top    = top.material.getConductivity()    * dx / (dy_top    / 2.0);
            double G_interface = 1.0 / (1.0 / G_bottom + 1.0 / G_top);

            int lastJ  = offsetY + bottom.ny - 1;  // last row of bottom layer
            int firstJ = offsetY + bottom.ny;      // first row of top layer

            for (int i = 0; i < bottom.nx; i++) {
                model.addComponent(PhysicsType.THERMAL,new LinearLink(nodes[i][lastJ], nodes[i][firstJ], G_interface));
            }

            offsetY += bottom.ny;
        }
        return nodes;
    }
}
