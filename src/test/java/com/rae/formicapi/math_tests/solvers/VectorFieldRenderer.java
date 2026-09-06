package com.rae.formicapi.math_tests.solvers;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static com.rae.formicapi.math_tests.solvers.Field2DRenderer.*;

public class VectorFieldRenderer {

    public static void saveVectorField(double[][] u, double[][] v, String filename) {
        saveVectorField(u, v, filename, 8, 0.45, 0.3);
    }

    public static void saveVectorField(double[][] u, double[][] v, String filename,
                                       int vectorSpacing, double arrowLength, double arrowHeadSize) {
        int nx = u.length;
        int ny = u[0].length;

        if (v.length != nx || v[0].length != ny) {
            throw new IllegalArgumentException("u and v must have the same dimensions");
        }

        int pixelsPerCell = 1;
        int scaleW = 80;

        int fieldW = nx * pixelsPerCell;
        int fieldH = ny * pixelsPerCell;

        int imageW = fieldW + scaleW;
        int imageH = fieldH;

        double[][] magnitude = new double[nx][ny];

        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double speed = Math.hypot(u[i][j], v[i][j]);

                magnitude[i][j] = speed;

                if (Double.isFinite(speed)) {
                    min = Math.min(min, speed);
                    max = Math.max(max, speed);
                }
            }
        }

        BufferedImage image =
                new BufferedImage(imageW, imageH, BufferedImage.TYPE_INT_RGB);

        Graphics2D g = image.createGraphics();

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);

        drawMagnitudeField(g, magnitude, min, max, scaleW, fieldW, fieldH);

        drawVectorField(g, u, v, scaleW, fieldH,
                vectorSpacing, arrowLength, arrowHeadSize);

        drawScale(g, scaleW, imageH, min, max);

        g.dispose();

        try {
            Path path = Paths.get("test-output", filename);
            Files.createDirectories(path.getParent());
            ImageIO.write(image, "png", path.toFile());

            System.out.println("Saved vector field to " + path.toAbsolutePath());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void drawMagnitudeField(Graphics2D g, double[][] magnitude,
                                           double min, double max,
                                           int scaleW, int fieldW, int fieldH) {
        int nx = magnitude.length;
        int ny = magnitude[0].length;

        BufferedImage render =
                new BufferedImage(nx, ny, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                double value = magnitude[x][y];

                if (!Double.isFinite(value)) {
                    render.setRGB(x, ny - 1 - y, Color.WHITE.getRGB());
                    continue;
                }

                double normalized =
                        max == min ? 0.0 : (value - min) / (max - min);

                normalized = Math.clamp(normalized, 0.0, 1.0);

                render.setRGB(x, ny - 1 - y,
                        heatmap(normalized).getRGB());
            }
        }

        g.drawImage(render, scaleW, 0, fieldW, fieldH, null);
    }

    private static void drawVectorField(Graphics2D g, double[][] u, double[][] v,
                                        int scaleW, int imageH,
                                        int spacing, double arrowLength,
                                        double arrowHeadSize) {
        int nx = u.length;
        int ny = u[0].length;

        g.setColor(Color.BLACK);
        g.setStroke(new BasicStroke(1.0f,
                BasicStroke.CAP_ROUND,
                BasicStroke.JOIN_ROUND));

        double maxSpeed = 0.0;

        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                double s = Math.hypot(u[x][y], v[x][y]);

                if (Double.isFinite(s)) {
                    maxSpeed = Math.max(maxSpeed, s);
                }
            }
        }

        if (maxSpeed <= 1e-12) {
            return;
        }

        for (int y = 0; y < ny; y += spacing) {
            for (int x = 0; x < nx; x += spacing) {

                double[] velocity = sampleVelocity(u, v, x, y);

                double vx = velocity[0];
                double vy = velocity[1];

                if (!Double.isFinite(vx) || !Double.isFinite(vy)) {
                    continue;
                }

                double speed = Math.hypot(vx, vy);

                if (speed <= 1e-12) {
                    continue;
                }

                double nxv = vx / speed;
                double nyv = vy / speed;

                double length = arrowLength * spacing;

                double x0 = scaleW + x;
                double y0 = imageH - 1 - y;

                double x1 = x0 + nxv * length;
                double y1 = y0 - nyv * length;

                drawArrow(g, x0, y0, x1, y1, arrowHeadSize * spacing);
            }
        }
    }

    private static void drawArrow(Graphics2D g,
                                  double x0, double y0,
                                  double x1, double y1,
                                  double headSize) {
        g.drawLine(
                (int) Math.round(x0),
                (int) Math.round(y0),
                (int) Math.round(x1),
                (int) Math.round(y1)
        );

        double angle = Math.atan2(y1 - y0, x1 - x0);

        double leftAngle = angle + Math.PI * 0.8;
        double rightAngle = angle - Math.PI * 0.8;

        int lx = (int) Math.round(x1 + Math.cos(leftAngle) * headSize);
        int ly = (int) Math.round(y1 + Math.sin(leftAngle) * headSize);

        int rx = (int) Math.round(x1 + Math.cos(rightAngle) * headSize);
        int ry = (int) Math.round(y1 + Math.sin(rightAngle) * headSize);

        g.drawLine((int) Math.round(x1), (int) Math.round(y1), lx, ly);
        g.drawLine((int) Math.round(x1), (int) Math.round(y1), rx, ry);
    }

    private static double[] sampleVelocity(double[][] u, double[][] v,
                                            double x, double y) {
        int nx = u.length;
        int ny = u[0].length;

        x = Math.clamp(x, 0.0, nx - 1.0);
        y = Math.clamp(y, 0.0, ny - 1.0);

        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);

        int x1 = Math.min(nx - 1, x0 + 1);
        int y1 = Math.min(ny - 1, y0 + 1);

        double tx = x - x0;
        double ty = y - y0;

        double ux =
                u[x0][y0] * (1 - tx) * (1 - ty) +
                u[x1][y0] * tx * (1 - ty) +
                u[x0][y1] * (1 - tx) * ty +
                u[x1][y1] * tx * ty;

        double vy =
                v[x0][y0] * (1 - tx) * (1 - ty) +
                v[x1][y0] * tx * (1 - ty) +
                v[x0][y1] * (1 - tx) * ty +
                v[x1][y1] * tx * ty;

        return new double[]{ux, vy};
    }
}