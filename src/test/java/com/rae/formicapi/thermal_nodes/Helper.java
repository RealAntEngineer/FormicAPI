package com.rae.formicapi.thermal_nodes;

import com.rae.formicapi.simulation.nodal.ModelType;
import com.rae.formicapi.simulation.nodal.core.UnknownNode;
import com.rae.formicapi.simulation.nodal.linear.thermal.PlateNodeHelper;

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

        int pixelsPerMeter = 2000;
        int supersample    = 2; // render at 2x then downscale

        // ------------------------------------------------
        // Compute cell sizes
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
                for (int j = 0; j < layer.ny; j++) cellH[rowIndex++] = h;
            }
        }

        // ------------------------------------------------
        // Compute total image size
        // ------------------------------------------------

        int imageW = Nx * cellW;
        int imageH = 0;
        for (int h : cellH) imageH += h;

        int renderW = imageW * supersample;
        int renderH = imageH * supersample;

        // ------------------------------------------------
        // Find min / max temperatures
        // ------------------------------------------------

        double T_min = Double.MAX_VALUE;
        double T_max = -Double.MAX_VALUE;

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double T = nodes[i][j].getValue(ModelType.THERMAL);
                if (T < T_min) T_min = T;
                if (T > T_max) T_max = T;
            }
        }

        double range  = T_max - T_min;
        double T_low  = T_min + 0.05 * range;
        double T_high = T_max - 0.05 * range;

        // ------------------------------------------------
        // Precompute row Y offsets and per-pixel node coords
        // ------------------------------------------------

        // cumulative pixel Y of bottom edge of each row (in render space)
        int[] rowY = new int[Ny + 1];
        rowY[0] = 0;
        for (int j = 0; j < Ny; j++) rowY[j + 1] = rowY[j] + cellH[j] * supersample;

        // For each render pixel x, find its fractional node-space i
        double[] pixelToNodeX = new double[renderW];
        int cellWs = cellW * supersample;
        for (int x = 0; x < renderW; x++) {
            pixelToNodeX[x] = (x + 0.5) / cellWs - 0.5; // center of node at i+0.5 in pixel space
        }

        // For each render pixel y (in flipped coords), find fractional node-space j
        double[] pixelToNodeY = new double[renderH];
        for (int y = 0; y < renderH; y++) {
            int yFlipped = renderH - 1 - y; // j=0 is bottom physically
            // find which row this belongs to
            double nodeJ = 0;
            for (int j = 0; j < Ny; j++) {
                int yBot = rowY[j];
                int yTop = rowY[j + 1];
                if (yFlipped >= yBot && yFlipped < yTop) {
                    double frac = (yFlipped - yBot + 0.5) / (yTop - yBot);
                    nodeJ = j + frac - 0.5;
                    break;
                }
            }
            pixelToNodeY[y] = nodeJ;
        }

        // ------------------------------------------------
        // Render with bicubic interpolation
        // ------------------------------------------------

        BufferedImage renderImage = new BufferedImage(renderW, renderH, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < renderH; y++) {
            double fj = pixelToNodeY[y];
            int    j0 = (int) Math.floor(fj);
            double tj = fj - j0;

            for (int x = 0; x < renderW; x++) {
                double fi = pixelToNodeX[x];
                int    i0 = (int) Math.floor(fi);
                double ti = fi - i0;

                // Bicubic: accumulate over 4x4 neighborhood
                double T = 0;
                for (int di = -1; di <= 2; di++) {
                    double wi = cubicWeight(di - ti);
                    int    ci = Math.max(0, Math.min(Nx - 1, i0 + di));
                    for (int dj = -1; dj <= 2; dj++) {
                        double wj = cubicWeight(dj - tj);
                        int    cj = Math.max(0, Math.min(Ny - 1, j0 + dj));
                        T += wi * wj * nodes[ci][cj].getValue(ModelType.THERMAL);
                    }
                }

                double normalized = (T - T_low) / (T_high - T_low);
                normalized = Math.max(0, Math.min(1, normalized));
                normalized = Math.sqrt(normalized);

                renderImage.setRGB(x, y, heatmap(normalized).getRGB());
            }
        }

        // ------------------------------------------------
        // Downscale to final size (average supersampled pixels)
        // ------------------------------------------------

        BufferedImage image = new BufferedImage(imageW, imageH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = image.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(renderImage, 0, 0, imageW, imageH, null);
        g2.dispose();

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

    /**
     * Cubic Hermite / Catmull-Rom kernel weight for distance t.
     * t is the distance from the sample point to the kernel center.
     */
    private static double cubicWeight(double t) {
        double a = Math.abs(t);
        if (a >= 2.0) return 0;
        if (a >= 1.0) return (-0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0);
        return (1.5 * a * a * a - 2.5 * a * a + 1.0);
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
