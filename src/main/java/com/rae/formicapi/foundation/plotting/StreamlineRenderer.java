package com.rae.formicapi.foundation.plotting;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static com.rae.formicapi.foundation.plotting.Field2DRenderer.*;

public class StreamlineRenderer {

    public static void saveStreamlines(double[][] u, double[][] v, String filename) {
        saveStreamlines(u, v, filename, 0.25, 5000, 10.0);
    }


    public static void saveStreamlines(double[][] u, double[][] v, String filename, double stepSize,
                                       int maxSteps, double minStreamlineDistance) {
        int nx = u.length;
        int ny = u[0].length;

        if (v.length != nx || v[0].length != ny) {
            throw new IllegalArgumentException("u and v must have the same dimensions");
        }

        int pixelsPerCell = 1;
        int supersample   = 1;
        int scaleW    = 80;

        int fieldW = nx * pixelsPerCell;
        int fieldH = ny * pixelsPerCell;

        int imageW = fieldW + scaleW;
        int imageH = fieldH;

        int renderW = fieldW * supersample;
        int renderH = fieldH * supersample;

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

        BufferedImage render =
                new BufferedImage(renderW, renderH, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < renderH; y++) {

            double fy = (double) y / (renderH - 1) * (ny - 1);
            int    j0 = (int) fy;
            double ty = fy - j0;

            for (int x = 0; x < renderW; x++) {

                double fx = (double) x / (renderW - 1) * (nx - 1);
                int    i0 = (int) fx;
                double tx = fx - i0;

                double value = bilinear(magnitude, i0, j0, tx, ty);

                double normalized =
                        max == min ? 0.0 : (value - min) / (max - min);

                normalized = Math.clamp(normalized, 0, 1);

                render.setRGB(x, renderH - 1 - y, heatmap(normalized).getRGB()
                );
            }
        }

        BufferedImage image =
                new BufferedImage(imageW, imageH, BufferedImage.TYPE_INT_RGB);

        Graphics2D g = image.createGraphics();

        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );

        g.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
        );

        // Background field.
        g.drawImage(render, scaleW, 0, fieldW, fieldH, null);

        // Magnitude scale.
        drawScale(g, scaleW, imageH, min, max);

        /*
         * Points from existing streamlines.
         *
         * These are used ONLY for deciding where additional
         * streamlines should be seeded.
         */
        List<Point2D> occupied = new ArrayList<>();

        /*
         * First populate from the boundary.
         */
        List<BoundarySeed> boundarySeeds = generateBoundarySeeds(nx, ny, minStreamlineDistance);

        for (BoundarySeed seed : boundarySeeds) {

            if (!entersDomain(u, v, seed.x, seed.y, seed.edge)) {
                continue;
            }

            addStreamline(g, u, v, seed.x, seed.y, stepSize, maxSteps,
                    minStreamlineDistance, occupied, scaleW,fieldH
            );
        }

        /*
         * Now fill low-density regions inside the domain.
         *
         * This is what catches recirculation regions that have no
         * streamline entering from the boundary.
         */
        populateLowDensityRegions(g, u, v, stepSize, maxSteps, minStreamlineDistance,
                occupied, scaleW, fieldH);

        g.dispose();

        try {
            Path path = Paths.get("test-output", filename);
            Files.createDirectories(path.getParent());
            ImageIO.write(image, "png", path.toFile());

            System.out.println(
                    "Saved streamlines to " + path.toAbsolutePath()
            );
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    private static boolean validVelocity(double[][] u, double[][] v, double x, double y) {
        if (x < 0 ||
                x > u.length - 1 ||
                y < 0 ||
                y > u[0].length - 1) {
            return false;
        }

        double[] velocity =
                sampleVelocity(u, v, x, y);

        return Double.isFinite(velocity[0])
                && Double.isFinite(velocity[1]);
    }

    private static double speed(double[][] u, double[][] v, double x, double y) {
        double[] velocity =
                sampleVelocity(u, v, x, y);

        return Math.hypot(
                velocity[0],
                velocity[1]
        );
    }

    /*
     * Integrate a streamline using RK4.
     */
    private static List<Point2D> integrate(double[][] u, double[][] v, double x, double y, double ds, int maxSteps) {
        int nx = u.length;
        int ny = u[0].length;

        List<Point2D> result = new ArrayList<>();

        for (int step = 0; step < maxSteps; step++) {

            if (x < 0 || x > nx - 1 || y < 0 || y > ny - 1) {
                break;
            }

            double[] k1 = velocity(u, v, x, y);
            double[] k2 = velocity(u, v, x + 0.5 * ds * k1[0], y + 0.5 * ds * k1[1]);
            double[] k3 = velocity(u, v, x + 0.5 * ds * k2[0], y + 0.5 * ds * k2[1]);
            double[] k4 = velocity(u, v, x + ds * k3[0], y + ds * k3[1]);

            result.add(new Point2D(x, y));

            x += ds / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);

            y += ds / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        }

        return result;
    }

    private static void drawStreamline(Graphics2D g, List<Point2D> streamline, int scaleW, int imageH) {
        if (streamline.size() < 2) {
            return;
        }


        //Use speed to slightly vary the line thickness.
        g.setStroke(new BasicStroke(1.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

        g.setColor(Color.BLACK);

        for (int i = 1; i < streamline.size(); i++) {

            Point2D a = streamline.get(i - 1);
            Point2D b = streamline.get(i);

            int x0 = (int) Math.round(scaleW + a.x);
            int y0 = (int) Math.round(imageH - 1 - a.y);

            int x1 = (int) Math.round(scaleW + b.x);
            int y1 = (int) Math.round(imageH - 1 - b.y);

            g.drawLine(x0, y0, x1, y1);
        }
    }

    /*
     * Keep only periodically spaced points from a streamline for the occupancy test.
     */
    private static List<Point2D> sampleStreamline(List<Point2D> streamline, double spacing) {
        List<Point2D> result = new ArrayList<>();

        double spacingSquared = spacing * spacing;

        for (Point2D p : streamline) {

            boolean accept = true;

            for (Point2D q : result) {

                double dx = p.x - q.x;
                double dy = p.y - q.y;

                if (dx * dx + dy * dy < spacingSquared) {
                    accept = false;
                    break;
                }
            }

            if (accept) {
                result.add(p);
            }
        }

        return result;
    }

    private static void populateLowDensityRegions(Graphics2D g, double[][] u, double[][] v, double stepSize, int maxSteps,
                                                  double minDistance, List<Point2D> occupied, int scaleW, int imageH) {
        int nx = u.length;
        int ny = u[0].length;

        /*
         * Use a grid finer than the desired streamline spacing so
         * we don't miss narrow low-density regions.
         */
        double candidateSpacing = minDistance * 0.5;

        boolean added;

        do {
            added = false;

            /*
             * Scan the domain for the most under-populated location.
             *
             * This is better than simply taking the first available
             * point because it progressively fills the largest holes.
             */
            Point2D bestSeed     = null;
            double  bestDistance = minDistance;

            for (double y = 0; y <= ny - 1; y += candidateSpacing) {
                for (double x = 0; x <= nx - 1; x += candidateSpacing) {

                    if (!validVelocity(u, v, x, y)) {
                        continue;
                    }

                    if (speed(u, v, x, y) <= 1e-12) {
                        continue;
                    }

                    double distance = distanceToStreamlines(x, y, occupied);
                    if (distance > bestDistance) {
                        bestDistance = distance;
                        bestSeed = new Point2D(x, y);
                    }
                }
            }

            if (bestSeed != null) {

                int before = occupied.size();

                addStreamline(g, u, v, bestSeed.x, bestSeed.y, stepSize, maxSteps, minDistance, occupied, scaleW,imageH);

                added = occupied.size() > before;
            }

        } while (added);
    }

    private static List<BoundarySeed> generateBoundarySeeds(int nx, int ny, double spacing) {
        List<BoundarySeed> seeds = new ArrayList<>();

        for (double x = 0; x <= nx - 1; x += spacing) {
            seeds.add(new BoundarySeed(x, 0, Edge.BOTTOM));
            seeds.add(new BoundarySeed(x, ny - 1, Edge.TOP));
        }

        for (double y = spacing; y < ny - 1; y += spacing) {
            seeds.add(new BoundarySeed(0, y, Edge.LEFT));
            seeds.add(new BoundarySeed(nx - 1, y, Edge.RIGHT));
        }

        return seeds;
    }

    private static boolean entersDomain(double[][] u, double[][] v, double x, double y, Edge edge) {
        double[] velocity = sampleVelocity(u, v, x, y);

        double speed = Math.hypot(velocity[0], velocity[1]);

        if (!Double.isFinite(speed) || speed <= 1e-12) {
            return false;
        }

        /*
         * Use a relative epsilon so "parallel" is rejected,
         * while numerical noise isn't treated as an inward direction.
         */
        double epsilon = speed * 1e-8;

        return switch (edge) {
            case BOTTOM -> velocity[1] > epsilon;
            case TOP -> velocity[1] < -epsilon;
            case LEFT -> velocity[0] > epsilon;
            case RIGHT -> velocity[0] < -epsilon;
        };
    }

    private static void addStreamline(Graphics2D g, double[][] u, double[][] v, double x, double y, double stepSize,
                                      int maxSteps, double minDistance, List<Point2D> occupied, int scaleW, int imageH) {
        if (!validVelocity(u, v, x, y)) {
            return;
        }

        if (speed(u, v, x, y) <= 1e-12) {
            return;
        }

        /*
         * Distance is only a SEEDING criterion.
         *
         * Once this streamline exists, it is allowed to approach
         * another streamline arbitrarily closely.
         */
        if (tooCloseToExistingStreamline(x, y, occupied, minDistance)) {
            return;
        }

        List<Point2D> backward = integrate(u, v, x, y, -stepSize, maxSteps);
        List<Point2D> forward  = integrate(u, v, x, y, stepSize, maxSteps);

        List<Point2D> streamline = new ArrayList<>(backward.size() + forward.size() + 1);

        for (int i = backward.size() - 1; i >= 0; i--) {
            streamline.add(backward.get(i));
        }

        streamline.add(new Point2D(x, y));
        streamline.addAll(forward);

        if (streamline.size() < 2) {
            return;
        }

        drawStreamline(g, streamline, scaleW,imageH);

        /*
         * Only after the streamline has been accepted do we add
         * its points to the density map.
         */
        occupied.addAll(sampleStreamline(streamline, minDistance));
    }

    private static double distanceToStreamlines(double x, double y, List<Point2D> occupied) {
        if (occupied.isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }

        double bestSquared = Double.POSITIVE_INFINITY;

        for (Point2D p : occupied) {

            double dx = x - p.x;
            double dy = y - p.y;

            double distanceSquared = dx * dx + dy * dy;

            bestSquared = Math.min(bestSquared, distanceSquared);
        }

        return Math.sqrt(bestSquared);
    }

    private static boolean tooCloseToExistingStreamline(double x, double y, List<Point2D> occupied, double minDistance) {
        double minDistanceSquared = minDistance * minDistance;

        for (Point2D p : occupied) {

            double dx = x - p.x;
            double dy = y - p.y;

            if (dx * dx + dy * dy < minDistanceSquared) {
                return true;
            }
        }

        return false;
    }

    /*
     * Bilinear velocity interpolation.
     */
    private static double[] sampleVelocity(double[][] u, double[][] v, double x, double y) {
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

        double ux = u[x0][y0] * (1 - tx) * (1 - ty) + u[x1][y0] * tx * (1 - ty) + u[x0][y1] * (1 - tx) * ty + u[x1][y1] * tx * ty;
        double vy = v[x0][y0] * (1 - tx) * (1 - ty) + v[x1][y0] * tx * (1 - ty) + v[x0][y1] * (1 - tx) * ty + v[x1][y1] * tx * ty;

        return new double[]{ux, vy};
    }

    private static double[] velocity(double[][] u, double[][] v, double x, double y) {
        double[] uv = sampleVelocity(u, v, x, y);
        return new double[]{uv[0], uv[1]};
    }

    private enum Edge {
        LEFT,
        RIGHT,
        BOTTOM,
        TOP
    }

    private static class BoundarySeed extends Point2D {

        final Edge edge;

        BoundarySeed(double x, double y, Edge edge) {
            super(x, y);
            this.edge = edge;
        }
    }

    private static class Point2D {

        final double x;
        final double y;

        Point2D(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }
}