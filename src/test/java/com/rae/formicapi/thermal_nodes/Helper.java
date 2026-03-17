package com.rae.formicapi.thermal_nodes;

import com.rae.formicapi.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.simulation.nodal.thermal.PlateNodeHelper;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

public class Helper {

    public static void savePlateHeatmap(UnknownNode[][] nodes, ArrayList<PlateNodeHelper.Layer> layers, String filename) {

        int Nx = nodes.length;
        int Ny = nodes[0].length;

        // ------------------------------------------------
        // Compute physical pixel sizes per column and row
        // ------------------------------------------------

        int pixelsPerMeter = 2000; // tune this to get a reasonable image size

        // ------------------------------------------------
        // Compute cell sizes — fallback to uniform if no layers
        // ------------------------------------------------

        int   cellW;
        int[] cellH = new int[Ny];

        if (layers == null || layers.isEmpty()) {

            int fallback = 50;
            cellW = fallback;
            Arrays.fill(cellH, fallback);

        } else {

            double dx = layers.get(0).length / layers.get(0).nx;
            cellW = Math.max(1, (int) (dx * pixelsPerMeter));

            int rowIndex = 0;
            for (PlateNodeHelper.Layer layer : layers) {
                double dy = layer.thickness / layer.ny;
                int h = Math.max(1, (int) (dy * pixelsPerMeter));
                for (int j = 0; j < layer.ny; j++) {
                    cellH[rowIndex++] = h;
                }
            }
        }

        // ------------------------------------------------
        // Compute total image size
        // ------------------------------------------------

        int imageW = Nx * cellW;
        int imageH = 0;
        for (int h : cellH) imageH += h;

        // ------------------------------------------------
        // Find min / max temperatures
        // ------------------------------------------------

        double T_min = Double.MAX_VALUE;
        double T_max = -Double.MAX_VALUE;

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double T = nodes[i][j].getValue();
                if (T < T_min) T_min = T;
                if (T > T_max) T_max = T;
            }
        }

        double range  = T_max - T_min;
        double T_low  = T_min + 0.05 * range;
        double T_high = T_max - 0.05 * range;

        // ------------------------------------------------
        // Precompute row Y offsets (bottom-up in physical space)
        // ------------------------------------------------

        int[] rowY = new int[Ny + 1]; // rowY[j] = pixel y of bottom edge of row j
        rowY[0] = 0;
        for (int j = 0; j < Ny; j++) {
            rowY[j + 1] = rowY[j] + cellH[j];
        }

        // ------------------------------------------------
        // Create and fill image
        // ------------------------------------------------

        BufferedImage image = new BufferedImage(imageW, imageH, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {

                double T          = nodes[i][j].getValue();
                double normalized = (T - T_low) / (T_high - T_low);
                normalized        = Math.max(0, Math.min(1, normalized));
                normalized        = Math.sqrt(normalized);

                Color color = heatmap(normalized);
                int rgb     = color.getRGB();

                int xStart = i * cellW;
                int xEnd   = xStart + cellW;

                // flip vertically: j=0 is bottom physically
                int yStart = imageH - rowY[j + 1];
                int yEnd   = imageH - rowY[j];

                for (int x = xStart; x < xEnd; x++) {
                    for (int y = yStart; y < yEnd; y++) {
                        image.setRGB(x, y, rgb);
                    }
                }
            }
        }

        // ------------------------------------------------
        // Save image
        // ------------------------------------------------

        try {
            Path path = Paths.get("test-output", filename);
            Files.createDirectories(path.getParent());
            ImageIO.write(image, "png", path.toFile());
            System.out.println("Saved heatmap to " + filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ------------------------------------------------
    // Heatmap gradient
    // blue -> cyan -> green -> yellow -> red
    // ------------------------------------------------

    public static Color heatmap(double t) {

        t = Math.max(0, Math.min(1, t));

        if (t < 0.25) {

            double k = t / 0.25;
            return lerp(new Color(0, 0, 128), new Color(0, 255, 255), k);

        }
        else if (t < 0.5) {

            double k = (t - 0.25) / 0.25;
            return lerp(new Color(0, 255, 255), new Color(0, 255, 0), k);

        }
        else if (t < 0.75) {

            double k = (t - 0.5) / 0.25;
            return lerp(new Color(0, 255, 0), new Color(255, 255, 0), k);

        }
        else {

            double k = (t - 0.75) / 0.25;
            return lerp(new Color(255, 255, 0), new Color(255, 0, 0), k);

        }
    }

    // ------------------------------------------------
    // Linear color interpolation
    // ------------------------------------------------

    public static Color lerp(Color a, Color b, double t) {

        int r = (int) (a.getRed()   + t * (b.getRed()   - a.getRed()));
        int g = (int) (a.getGreen() + t * (b.getGreen() - a.getGreen()));
        int bl = (int) (a.getBlue() + t * (b.getBlue()  - a.getBlue()));

        return new Color(r, g, bl);
    }
}
