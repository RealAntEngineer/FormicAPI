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

        int totalNx = layers.get(0).nx;
        int totalNy = 1;
        for (Layer layer : layers) totalNy += layer.ny;

        UnknownNode[][] nodes = new UnknownNode[totalNx][totalNy];
        for (int i = 0; i < totalNx; i++)
            for (int j = 0; j < totalNy; j++) {
                nodes[i][j] = new UnknownNode(PhysicsType.THERMAL, 25);
                model.addNode(nodes[i][j]);
            }

        // ----------------------------------------------------------------
        // Vertical links within each layer
        // Distance between any two adjacent nodes is always dy
        // ----------------------------------------------------------------
        int nodeRowStart = 0;
        for (Layer layer : layers) {
            double dx = layer.length    / layer.nx;
            double dy = layer.thickness / layer.ny;

            for (int j = nodeRowStart; j < nodeRowStart + layer.ny; j++) {
                for (int i = 0; i < layer.nx; i++) {
                    double G = layer.material.getConductivity() * dx / dy;
                    model.addComponent(new LinearLink(
                            nodes[i][j], nodes[i][j + 1], PhysicsType.THERMAL, G));
                }
            }
            nodeRowStart += layer.ny;
        }

        // ----------------------------------------------------------------
        // Horizontal links — interior rows of each layer
        // Edge rows (bottom boundary of layer, top boundary of last layer)
        // use half-cell height dy/2; interior rows use full dy.
        // Interface rows are handled separately below.
        // ----------------------------------------------------------------
        nodeRowStart = 0;
        for (int l = 0; l < layers.size(); l++) {
            Layer  layer = layers.get(l);
            double dx    = layer.length    / layer.nx;
            double dy    = layer.thickness / layer.ny;
            boolean isLastLayer = (l == layers.size() - 1);

            for (int j = nodeRowStart; j <= nodeRowStart + layer.ny; j++) {

                boolean isBottomBoundary = (j == nodeRowStart);
                boolean isTopBoundary    = (j == nodeRowStart + layer.ny);
                boolean isInterface      = isTopBoundary && !isLastLayer;

                // Interface rows are handled in the dedicated block below
                if (isInterface) continue;

                double halfDy = (isBottomBoundary || isTopBoundary) ? 0.5 * dy : dy;

                for (int i = 0; i < layer.nx - 1; i++) {
                    double G = layer.material.getConductivity() * halfDy / dx;
                    model.addComponent(new LinearLink(
                            nodes[i][j], nodes[i + 1][j], PhysicsType.THERMAL, G));
                }
            }
            nodeRowStart += layer.ny;
        }

        // ----------------------------------------------------------------
        // Horizontal links at interface rows
        // The interface node sits between two materials; each side contributes
        // a half-cell (dy/2) in series, giving a harmonic-mean conductance.
        // ----------------------------------------------------------------
        nodeRowStart = 0;
        for (int l = 0; l < layers.size() - 1; l++) {
            Layer bot = layers.get(l);
            Layer top = layers.get(l + 1);
            nodeRowStart += bot.ny;  // global j of the interface row

            double dy_bot = bot.thickness / bot.ny;
            double dy_top = top.thickness / top.ny;
            double dx     = bot.length    / bot.nx;

            // Each half-cell resistance in the horizontal direction:
            // R = distance / (k * area) = (dy/2) / (k * 1)  [per unit depth, dx cancels]
            // But here the section area for horizontal flow IS dy/2 (height) * 1 (depth)
            // and distance is dx, so G = k * (dy/2) / dx  — harmonic mean of the two:
            double G_bot       = bot.material.getConductivity() * (dy_bot / 2.0) / dx;
            double G_top       = top.material.getConductivity() * (dy_top / 2.0) / dx;
            double G_interface = 1.0 / (1.0 / G_bot + 1.0 / G_top);

            for (int i = 0; i < bot.nx - 1; i++) {
                model.addComponent(new LinearLink(
                        nodes[i][nodeRowStart], nodes[i + 1][nodeRowStart],
                        PhysicsType.THERMAL, G_interface));
            }
        }

        return nodes;
    }

}
