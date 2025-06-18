package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.PengRobinsonEOS;
import com.rae.formicapi.thermal_utilities.WaterCubicEOSTransformationHelper;
import net.createmod.catnip.data.Couple;
import org.knowm.xchart.*;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class HeatScatterPlot {

    public static final PengRobinsonEOS PR_WATER_EOS = EOSLibrary.getPRWaterEOS();

    public static void main(String[] args) {
        plotEntropyTV();
    }

    private static void plotTemperaturePH() {
        // Define ranges for P and H
        double pMin = 1.0, pMax = 10.0;
        double hMin = 1e3, hMax = 1e6;

        int pSteps = 20;
        int hSteps = 20;

        // Collect all T values to normalize for color gradient
        List<PointData> points = new ArrayList<>();

        for (int i = 0; i <= pSteps; i++) {
            double P = pMin + i * (pMax - pMin) / pSteps;
            for (int j = 0; j <= hSteps; j++) {
                double H = hMin + j * (hMax - hMin) / hSteps;
                double T = WaterCubicEOSTransformationHelper.get_T((float) P, (float) H);
                points.add(new PointData(P, H, T));
            }
        }

        // Find min/max T for color scaling
        double minT = 0;
        double maxT = 800;

        // Create chart
        XYChart chart = new XYChartBuilder().width(800).height(600).title("Temperature Heatmap Scatter Plot").xAxisTitle("Pressure (P)").yAxisTitle("Enthalpy (H)").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setMarkerSize(8);
        chart.getStyler().setLegendVisible(false);

        // Add one series per point (to assign different color)
        int pointIndex = 0;
        System.out.println(points.size());
        for (PointData pt : points) {
            Color color = getColorFromValue(pt.Z, minT, maxT);
            XYSeries series = chart.addSeries("pt" + (pointIndex++), new double[]{pt.X}, new double[]{pt.Y});
            series.setMarker(SeriesMarkers.CIRCLE);
            series.setMarkerColor(color);
        }

        // Show chart
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotEntropyTV(){
        // Define ranges for P and H
        double TMin = 100, TMax = 630;

        double vMin = PR_WATER_EOS.getB()*1.01, vMax = 5e-3;

        int TSteps = 100;
        int vSteps = 50;

        // Collect all T values to normalize for color gradient
        List<PointData> SPoints = new ArrayList<>();
        List<PointData> XPoints = new ArrayList<>();
        List<PointData> SaturationPoints = new ArrayList<>();

        for (int i = 0; i <= TSteps; i++) {
            double T = TMin + i * (TMax - TMin) / TSteps;
            Couple<Double> Vs = PR_WATER_EOS.getSaturationVolumes(T);
            SaturationPoints.add(new PointData(Vs.getFirst(), PR_WATER_EOS.pressure(T, Vs.getFirst()), 0));
            SaturationPoints.add(new PointData(Vs.getSecond(), PR_WATER_EOS.pressure(T, Vs.getSecond()), 1));
            for (int j = 0; j <= vSteps; j++) {
                double V = vMin + j * (vMax - vMin) / vSteps;
                double P = PR_WATER_EOS.pressure(T, V);
                double S = PR_WATER_EOS.totalEntropy((float) T, (float) V);
                double x = WaterCubicEOSTransformationHelper.get_x_from_entropy((float) S, (float) T);
                SPoints.add(new PointData(V, P, S));
                XPoints.add(new PointData(V, P, x));
            }
        }
        System.out.println(SaturationPoints);
        // Find min/max T for color scaling
        XYChart chart1 = new XYChartBuilder().width(800).height(600)
                .title("Entropy Plot")
                .yAxisTitle("Pressure (Pa)").xAxisTitle("Molar Volume (V)").build();
        chart1.getStyler().setXAxisLogarithmic(true);
        chart1.getStyler().setXAxisMin(vMin);
        chart1.getStyler().setXAxisMax(vMax);
        chart1.getStyler().setYAxisMin(-10e6);
        chart1.getStyler().setYAxisMax(30e6);
        plotWithColor(SPoints, chart1, "pt");
        plotWithColor(SaturationPoints, chart1, "sat");
        new SwingWrapper<>(chart1).displayChart();

        XYChart chart2 = new XYChartBuilder().width(800).height(600)
                .title("vapor quality Plot")
                .yAxisTitle("Pressure (Pa)").xAxisTitle("Molar Volume (V)").build();
        chart2.getStyler().setXAxisLogarithmic(true);
        chart2.getStyler().setXAxisMin(vMin);
        chart2.getStyler().setXAxisMax(vMax);
        chart2.getStyler().setYAxisMin(-10e6);
        chart2.getStyler().setYAxisMax(30e6);
        plotWithColor(XPoints, chart2, "pt");
        plotWithColor(SaturationPoints, chart2, "sat");
        new SwingWrapper<>(chart2).displayChart();


        XYChart chart3 = new XYChartBuilder().width(800).height(600)
                .title("Saturation Plot")
                .yAxisTitle("Pressure (Pa)").xAxisTitle("Molar Volume (V)").build();
        chart3.getStyler().setXAxisLogarithmic(true);
        chart3.getStyler().setXAxisMin(vMin);
        chart3.getStyler().setXAxisMax(vMax);
        chart3.getStyler().setYAxisMin(-10e6);
        chart3.getStyler().setYAxisMax(30e6);
        plotWithColor(SaturationPoints, chart3, "sat");

        new SwingWrapper<>(chart3).displayChart();
    }

    private static void plotWithColor(List<PointData> points, XYChart chart, String seriesName) {
        double minT = points.stream().map(pt -> pt.Z).min(Double::compareTo).orElse(0.0);
        double maxT = points.stream().map(pt -> pt.Z).max(Double::compareTo).orElse(1.0);
        // Create chart
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendVisible(false);

        // Add one series per point (to assign different color)
        int pointIndex = 0;
        for (PointData pt : points) {
            Color color = getColorFromValue(pt.Z, minT, maxT);
            XYSeries series = chart.addSeries(seriesName + (pointIndex++), new double[]{pt.X}, new double[]{pt.Y});
            series.setMarker(SeriesMarkers.CIRCLE);
            series.setMarkerColor(color);
        }

        // Show chart

    }

    private static void plotEnthalpyTV() {
        // Define ranges for P and H
        double tMin = 273, tMax = 600;
        double vMin = EOSLibrary.getPRWaterEOS().getB()*1.3, vMax = 0.0002;

        int pSteps = 200;
        int vSteps = 400;

        // Collect all T values to normalize for color gradient
        List<PointData> points = new ArrayList<>();

        for (int i = 0; i <= pSteps; i++) {
            double T = tMin + i * (tMax - tMin) / pSteps;
            for (int j = 0; j <= vSteps; j++) {
                double V = vMin + j * (vMax - vMin) / vSteps;
                double H = EOSLibrary.getPRWaterEOS().totalEnthalpy((float) T, (float) V);
                points.add(new PointData(T, V, H));
            }
        }

        // Find min/max T for color scaling
        double minT = points.stream().map(pt -> pt.Z).min(Double::compareTo).orElse(0.0);
        double maxT = points.stream().map(pt -> pt.Z).max(Double::compareTo).orElse(1.0);
        System.out.println("minT: " + minT + " maxT: " + maxT);
        // Create chart
        XYChart chart = new XYChartBuilder().width(800).height(600)
                .title("Enthalpy Heatmap Scatter Plot")
                .xAxisTitle("Temperature (T)").yAxisTitle("Molar Volume (H)").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendVisible(false);

        // Add one series per point (to assign different color)
        int pointIndex = 0;
        for (PointData pt : points) {
            Color color = getColorFromValue(pt.Z, minT, maxT);
            XYSeries series = chart.addSeries("pt" + (pointIndex++), new double[]{pt.X}, new double[]{pt.Y});
            series.setMarker(SeriesMarkers.CIRCLE);
            series.setMarkerColor(color);
        }

        // Show chart
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotEnthalpyPT() {
        // Define ranges for P and H
        double tMin = 620, tMax = 800;
        double pMin = 1e3, pMax = 1e5;

        int tSteps = 120;
        int pSteps = 320;

        // Collect all T values to normalize for color gradient
        List<PointData> points = new ArrayList<>();

        for (int i = 0; i <= tSteps; i++) {
            double T = tMin + i * (tMax - tMin) / tSteps;
            for (int j = 0; j <= pSteps; j++) {
                double P = pMin + j * (pMax - pMin) / pSteps;
                double H = WaterCubicEOSTransformationHelper.get_h(0,(float) T, (float) P);
                points.add(new PointData(P,T, H));
            }
        }

        // Find min/max T for color scaling
        double minT = points.stream().map(pt -> pt.Z).min(Double::compareTo).orElse(0.0);
        double maxT = points.stream().map(pt -> pt.Z).max(Double::compareTo).orElse(1.0);
        System.out.println("minT: " + minT + " maxT: " + maxT);


        // Create chart
        XYChart chart = new XYChartBuilder().width(800).height(600)
                .title("Enthalpy Heatmap Scatter Plot")
                .yAxisTitle("Temperature (T)").xAxisTitle("Pressure (H)").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendVisible(false);

        // Add one series per point (to assign different color)
        int pointIndex = 0;
        for (PointData pt : points) {
            Color color = getColorFromValue(pt.Z, minT, maxT);
            XYSeries series = chart.addSeries("pt" + (pointIndex++), new double[]{pt.X}, new double[]{pt.Y});
            series.setMarker(SeriesMarkers.CIRCLE);
            series.setMarkerColor(color);
        }

        // Show chart
        new SwingWrapper<>(chart).displayChart();
    }

    private static void plotPressureTV() {
        // Define ranges for P and H
        double tMin = 273, tMax = 600;
        double vMin = EOSLibrary.getPRWaterEOS().getB()*1.1, vMax = 0.005;

        int pSteps = 200;
        int vSteps = 400;

        // Collect all T values to normalize for color gradient
        List<PointData> points = new ArrayList<>();

        for (int i = 0; i <= pSteps; i++) {
            double T = tMin + i * (tMax - tMin) / pSteps;
            for (int j = 0; j <= vSteps; j++) {
                double V = vMin + j * (vMax - vMin) / vSteps;
                double H = EOSLibrary.getPRWaterEOS().pressure((float) T, (float) V);
                points.add(new PointData(T, V, H));
            }
        }

        // Find min/max T for color scaling
        plotWithColor(points, new XYChartBuilder().width(800).height(600)
                .title("Pressure Heatmap Scatter Plot")
                .xAxisTitle("Temperature (T)").yAxisTitle("Molar Volume (H)").build(), "pt");
    }

    // Normalize value between min and max and map to gradient blue->red
    private static Color getColorFromValue(double value, double min, double max) {
        double ratio = (value - min) / (max - min);
        ratio = Math.min(Math.max(ratio, 0.0), 1.0); // clamp 0..1

        // Simple gradient from blue (cold) to red (hot)
        int red = (int) (ratio * 255);
        int blue = (int) ((1 - ratio) * 255);
        return new Color(red, 0, blue);
    }

    // Simple struct for points
    private static class PointData {
        double X, Y, Z;

        PointData(double x, double y, double z) {
            this.X = x;
            this.Y = y;
            this.Z = z;
        }

        @Override
        public String toString() {
            return "PointData{" +
                    "X=" + X +
                    ", Y=" + Y +
                    ", Z=" + Z +
                    '}';
        }
    }
}

