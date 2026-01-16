package com.rae.formicapi.plotting;

import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

/**
 * Minimalist Matplotlib-style plotting utility for Forge mods.
 * Supports multiple series, logarithmic axes, and PNG output.
 */
@OnlyIn(Dist.CLIENT)
public class SimplePlot {

    public static class Series {
        public String label;
        public List<Double> x;
        public List<Double> y;
        public Color color;

        public Series(String label, List<Double> x, List<Double> y, Color color) {
            this.label = label;
            this.x = x;
            this.y = y;
            this.color = color;
        }
    }

    private final List<Series> seriesList = new ArrayList<>();
    private boolean xLog = false;
    private boolean yLog = false;
    private boolean showLegend = false;
    private String xLabel = "";
    private String yLabel = "";
    private String title = "";

    private int width = 800;
    private int height = 600;
    private int margin = 60;

    private double xMin = Double.POSITIVE_INFINITY;
    private double xMax = Double.NEGATIVE_INFINITY;
    private double yMin = Double.POSITIVE_INFINITY;
    private double yMax = Double.NEGATIVE_INFINITY;

    // ===== Setters =====

    public SimplePlot width(int width) { this.width = width; return this; }
    public SimplePlot height(int height) { this.height = height; return this; }
    public SimplePlot title(String title) { this.title = title; return this; }
    public SimplePlot xlabel(String label) { this.xLabel = label; return this; }
    public SimplePlot ylabel(String label) { this.yLabel = label; return this; }
    public SimplePlot xLog(boolean log) { this.xLog = log; return this; }
    public SimplePlot yLog(boolean log) { this.yLog = log; return this; }
    public SimplePlot showLegend(boolean showLegend){this.showLegend = showLegend;return this;}

    public void addSeries(String label, List<Double> x, List<Double> y) {
        Color color = randomColor(seriesList.size());
        seriesList.add(new Series(label, x, y, color));

        for (double val : x) { if (!Double.isNaN(val)) { xMin = Math.min(xMin, val); xMax = Math.max(xMax, val); } }
        for (double val : y) { if (!Double.isNaN(val)) { yMin = Math.min(yMin, val); yMax = Math.max(yMax, val); } }
    }

    // ===== Rendering =====

    public void save(String fileName) {
        try {
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = image.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Background
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, width, height);

            // Axes
            g.setColor(Color.BLACK);
            g.drawLine(margin, height - margin, width - margin, height - margin); // X
            g.drawLine(margin, margin, margin, height - margin); // Y

            // Axis labels
            g.setFont(new Font("SansSerif", Font.PLAIN, 14));
            g.drawString(xLabel, width / 2 - xLabel.length() * 3, height - 10);
            g.drawString(yLabel, 10, height / 2);

            // Title
            g.setFont(new Font("SansSerif", Font.BOLD, 16));
            g.drawString(title, width / 2 - title.length() * 3, margin / 2);

            // Draw each series
            for (Series s : seriesList) {
                g.setColor(s.color);
                int prevX = -1, prevY = -1;
                for (int i = 0; i < s.x.size(); i++) {
                    double xv = s.x.get(i);
                    double yv = s.y.get(i);
                    if (Double.isNaN(xv) || Double.isNaN(yv)) continue;

                    int px = margin + (int) ((width - 2*margin) * normalize(xv, xMin, xMax, xLog));
                    int py = height - margin - (int) ((height - 2*margin) * normalize(yv, yMin, yMax, yLog));

                    if (prevX >= 0 && prevY >= 0) {
                        g.drawLine(prevX, prevY, px, py);
                    }
                    prevX = px;
                    prevY = py;
                }
            }

            // Legend
            if (showLegend) {
                int legendX = width - margin - 120;
                int legendY = margin;
                g.setFont(new Font("SansSerif", Font.PLAIN, 12));
                for (Series s : seriesList) {
                    g.setColor(s.color);
                    g.fillRect(legendX, legendY, 10, 10);
                    g.setColor(Color.BLACK);
                    g.drawString(s.label, legendX + 15, legendY + 10);
                    legendY += 20;
                }
            }

            drawGraduations(g, xMin, xMax, yMin, yMax);
            g.dispose();

            File dir = new File("plots");
            dir.mkdirs();
            File out = new File(dir, fileName);
            ImageIO.write(image, "PNG", out);
            System.out.println("Saved plot to " + out.getAbsolutePath());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ===== Utilities =====
    private void drawGraduations(Graphics2D g, double xMin, double xMax, double yMin, double yMax) {
        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g.setColor(Color.BLACK);

        int tickSize = 5;

        // X axis ticks
        int numXTicks = 10;
        for (int i = 0; i <= numXTicks; i++) {
            double val;
            if (xLog) {
                double logMin = Math.log10(xMin);
                double logMax = Math.log10(xMax);
                val = Math.pow(10, logMin + i*(logMax-logMin)/numXTicks);
            } else val = xMin + i*(xMax-xMin)/numXTicks;

            int px = margin + (int)((width - 2*margin) * normalize(val, xMin, xMax, xLog));
            g.drawLine(px, height - margin, px, height - margin + tickSize);
            String label = formatNumber(val);
            g.drawString(label, px - g.getFontMetrics().stringWidth(label)/2, height - margin + 20);
        }

        // Y axis ticks
        int numYTicks = 10;
        for (int i = 0; i <= numYTicks; i++) {
            double val;
            if (yLog) {
                double logMin = Math.log10(yMin);
                double logMax = Math.log10(yMax);
                val = Math.pow(10, logMin + i*(logMax-logMin)/numYTicks);
            } else val = yMin + i*(yMax-yMin)/numYTicks;

            int py = height - margin - (int)((height - 2*margin) * normalize(val, yMin, yMax, yLog));
            g.drawLine(margin - tickSize, py, margin, py);
            String label = formatNumber(val);
            g.drawString(label, margin - tickSize - g.getFontMetrics().stringWidth(label) - 2, py + 4);
        }
    }
    private static double normalize(double value, double min, double max, boolean log) {
        if (log) {
            value = Math.log10(value);
            min = Math.log10(min);
            max = Math.log10(max);
        }
        return (value - min) / (max - min);
    }

    private static Color randomColor(int seed) {
        int r = (int) ((Math.sin(seed + 1) * 10000) % 255);
        int g = (int) ((Math.cos(seed + 2) * 10000) % 255);
        int b = (int) ((Math.sin(seed + 3) * 10000) % 255);
        return new Color(Math.abs(r), Math.abs(g), Math.abs(b));
    }

    private static String formatNumber(double val) {
        if (Math.abs(val) >= 1e4 || Math.abs(val) < 1e-2) return String.format("%.1e", val);
        if (Math.abs(val) >= 10) return String.format("%.0f", val);
        return String.format("%.3f", val);
    }
}

