package com.rae.formicapi;

import com.rae.formicapi.simulation.nodal.core.UnknownNode;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Helper {

    public static void savePlateHeatmap(UnknownNode[][] nodes, String filename) {

        int Nx = nodes.length;
        int Ny = nodes[0].length;

        // ------------------------------------------------
        // Find min and max temperatures
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

        // ------------------------------------------------
        // Clamp range to reduce outlier influence
        // ------------------------------------------------

        double range = T_max - T_min;

        double T_low  = T_min + 0.05 * range;
        double T_high = T_max - 0.05 * range;

        // ------------------------------------------------
        // Create image
        // ------------------------------------------------

        int scale = 50;

        BufferedImage image =
                new BufferedImage(Nx * scale, Ny * scale, BufferedImage.TYPE_INT_RGB);

        // ------------------------------------------------
        // Fill pixels
        // ------------------------------------------------

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {

                double T = nodes[i][j].getValue();

                // normalize
                double normalized = (T - T_low) / (T_high - T_low);

                // clamp
                normalized = Math.max(0, Math.min(1, normalized));

                // nonlinear scaling (optional but recommended)
                normalized = Math.sqrt(normalized);

                Color color = heatmap(normalized);

                // draw block
                for (int x = i * scale; x < (i + 1) * scale; x++) {
                    for (int y = (Ny - j - 1) * scale; y < (Ny - j) * scale; y++) {

                        image.setRGB(x, y, color.getRGB());

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
