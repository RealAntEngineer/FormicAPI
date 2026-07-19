package com.rae.formicapi.math_tests.solvers;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Field2DRenderer {

    public enum Interpolation {
        NEAREST,
        BILINEAR
    }

    public static void saveHeatmap(double[][] field, String filename) {
        saveHeatmap(field, filename, Interpolation.BILINEAR);
    }

    public static void saveHeatmap(double[][] field, String filename, Interpolation interpolation) {

        int nx = field.length;
        int ny = field[0].length;

        int pixelsPerCell = 20;
        int supersample   = 2;

        int imageW = nx * pixelsPerCell;
        int imageH = ny * pixelsPerCell;

        int renderW = imageW * supersample;
        int renderH = imageH * supersample;

        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for (double[] doubles : field) {
            for (int j = 0; j < ny; j++) {
                double v = doubles[j];

                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        BufferedImage render = new BufferedImage(renderW, renderH, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < renderH; y++) {

            double fy = ((double) y / renderH) * (ny - 1);
            int    j0 = (int) fy;
            double ty = fy - j0;

            for (int x = 0; x < renderW; x++) {
                double fx = ((double) x / renderW) * (nx - 1);
                int    i0 = (int) fx;
                double tx = fx - i0;

                double value = switch (interpolation) {
                    case NEAREST -> nearest(field, i0, j0, tx, ty);
                    case BILINEAR -> bilinear(field, i0, j0, tx, ty);
                };

                double normalized = (value - min) / (max - min);

                normalized = Math.clamp(normalized, 0, 1);

                // flip Y so row 0 is bottom
                render.setRGB(x, renderH - 1 - y, heatmap(normalized).getRGB());
            }
        }


        BufferedImage image =
                new BufferedImage(imageW, imageH, BufferedImage.TYPE_INT_RGB);


        Graphics2D g = image.createGraphics();

        // For NEAREST field sampling, also drop the AWT downscale to
        // nearest-neighbor so the blocky per-cell look survives the final
        // resize instead of being smoothed away by bilinear resampling.
        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                interpolation == Interpolation.NEAREST
                        ? RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
                        : RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );

        g.drawImage(render, 0, 0, imageW, imageH, null);
        g.dispose();

        try {
            Path path = Paths.get("test-output", filename);
            Files.createDirectories(path.getParent());
            ImageIO.write(image, "png", path.toFile());
            System.out.println("Saved heatmap to " + path.toAbsolutePath());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    private static double bilinear(double[][] f, int x, int y, double tx, double ty) {

        int nx = f.length;
        int ny = f[0].length;

        int x1 = Math.min(nx - 1, x + 1);
        int y1 = Math.min(ny - 1, y + 1);

        double a = f[x][y];
        double b = f[x1][y];
        double c = f[x][y1];
        double d = f[x1][y1];

        return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
    }

    private static double nearest(double[][] f, int x, int y, double tx, double ty) {

        int nx = f.length;
        int ny = f[0].length;

        int xi = tx < 0.5 ? x : Math.min(nx - 1, x + 1);
        int yi = ty < 0.5 ? y : Math.min(ny - 1, y + 1);

        return f[xi][yi];
    }


    public static Color heatmap(double t) {

        if (t < 0.25)
            return lerp(new Color(0, 0, 128),
                    new Color(0, 255, 255),
                    t * 4);

        if (t < 0.5)
            return lerp(new Color(0, 255, 255),
                    new Color(0, 255, 0),
                    (t - 0.25) * 4);

        if (t < 0.75)
            return lerp(new Color(0, 255, 0),
                    new Color(255, 255, 0),
                    (t - 0.5) * 4);

        return lerp(new Color(255, 255, 0),
                new Color(255, 0, 0),
                (t - 0.75) * 4);
    }


    private static Color lerp(Color a, Color b, double t) {
        return new Color(
                (int) (a.getRed() + t * (b.getRed() - a.getRed())),
                (int) (a.getGreen() + t * (b.getGreen() - a.getGreen())),
                (int) (a.getBlue() + t * (b.getBlue() - a.getBlue()))
        );
    }
}