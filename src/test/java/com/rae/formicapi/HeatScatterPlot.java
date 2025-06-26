package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.SpecificRealGazState;
import com.rae.formicapi.thermal_utilities.eos.PengRobinsonEOS;
import com.rae.formicapi.thermal_utilities.helper.WaterCubicEOS;
import net.createmod.catnip.data.Couple;
import org.knowm.xchart.*;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class HeatScatterPlot {

    public static final PengRobinsonEOS PR_WATER_EOS = EOSLibrary.getPRWaterEOS();

    public static void main(String[] args) {
        plotEntropyPH();
    }

    private static void plotTemperaturePH() {
        // Define ranges for P and H
        double pMin = 1e5, pMax = 1e7;
        double hMin = 1e5, hMax = 1e6;

        int pSteps = 50;
        int hSteps = 50;

        // Collect all T values to normalize for color gradient
        List<PointData> points = new ArrayList<>();

        for (int i = 0; i <= pSteps; i++) {
            double P = pMin + i * (pMax - pMin) / pSteps;
            for (int j = 0; j <= hSteps; j++) {
                double H = hMin + j * (hMax - hMin) / hSteps;
                double T = WaterCubicEOS.get_T((float) P, (float) H);
                points.add(new PointData(P, H, T));
            }
        }

        // Find min/max T for color scaling
        double minT = points.stream().map(p -> p.Z).min(Double::compareTo).get();
        double maxT = points.stream().map(p -> p.Z).max(Double::compareTo).get();
        System.out.println("minT: " + minT);
        System.out.println("maxT: " + maxT);
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
    private static void plotEntropyPV(){
        // Define ranges for P and H
        double TMin = 273, TMax = 1000;

        double vMin = 2e-5, vMax = 5e-4;

        int TSteps = 500;
        int vSteps = 200;

        // Collect all T values to normalize for color gradient
        List<PointData> SPoints = new ArrayList<>();
        List<PointData> XPoints = new ArrayList<>();

        for (int i = 0; i <= TSteps; i++) {
            double T = TMin + i * (TMax - TMin) / TSteps;
            Couple<Double> Vs = PR_WATER_EOS.getSaturationVolumes(T);
            for (int j = 0; j <= vSteps; j++) {
                double V = vMin + j * (vMax - vMin) / vSteps;
                double P = PR_WATER_EOS.pressure(T, V);
                if (V > Vs.getFirst() && V < Vs.getSecond()) {
                    P = PR_WATER_EOS.saturationPressure(T);

                }
                double S = PR_WATER_EOS.totalEntropy((float) T, (float) V);
                double x = WaterCubicEOS.get_x_from_entropy((float) S, (float) T, (float) P);
                SPoints.add(new PointData(V, P, S));
                XPoints.add(new PointData(V, P, x));
            }
        }
        // Find min/max T for color scaling
        XYChart chart1 = new XYChartBuilder().width(800).height(600)
                .title("Entropy Plot")
                .yAxisTitle("Pressure (Pa)").xAxisTitle("Molar Volume (V)").build();
        chart1.getStyler().setXAxisLogarithmic(true);
        chart1.getStyler().setXAxisMin(vMin);
        chart1.getStyler().setXAxisMax(vMax);
        chart1.getStyler().setYAxisMin(1d);
        chart1.getStyler().setYAxisMax(30e6);
        plotWithColor(SPoints, chart1, "pt", 0d,20d);

        new SwingWrapper<>(chart1).displayChart();

        XYChart chart2 = new XYChartBuilder().width(800).height(600)
                .title("vapor quality Plot")
                .yAxisTitle("Pressure (Pa)").xAxisTitle("Molar Volume (V)").build();
        chart2.getStyler().setXAxisLogarithmic(true);
        chart2.getStyler().setXAxisMin(vMin);
        chart2.getStyler().setXAxisMax(vMax);
        chart2.getStyler().setYAxisMin(1d);
        chart2.getStyler().setYAxisMax(30e6);
        plotWithColor(XPoints, chart2, "pt", 0d, 1d);

        new SwingWrapper<>(chart2).displayChart();

    }
    private static void plotXPH(){
        // Define ranges for P and H
        double TMin = 273, TMax = 2000;

        double vMin = 2e-5, vMax = 1;

        int TSteps = 40;


        // Collect all T values to normalize for color gradient
        List<PointData> SPoints = new ArrayList<>();

        for (int i = 0; i <= TSteps; i++) {
            double T = TMin + i * (TMax - TMin) / TSteps;
            Couple<Double> Vs = PR_WATER_EOS.getSaturationVolumes(T);
            for (double V = vMin; V <= vMax; V*=1.005f) {
                double P = PR_WATER_EOS.pressure(T, V);
                if (V > Vs.getFirst() && V < Vs.getSecond()) {
                    P = PR_WATER_EOS.saturationPressure(T);

                }
                double S = PR_WATER_EOS.totalEntropy((float) T, (float) V);
                double H = PR_WATER_EOS.totalEnthalpy((float) T, (float) V);
                double X = WaterCubicEOS.get_x((float) H, (float) T, (float) P);
                SPoints.add(new PointData(H/1e3,P/1e5, X));

            }
        }

        double minS = SPoints.stream().map(p -> p.Z).min(Double::compareTo).get();
        double maxS = SPoints.stream().map(p -> p.Z).max(Double::compareTo).get();
        System.out.println("minX " + minS);
        System.out.println("maxX: " + maxS);
        // Find min/max T for color scaling
        XYChart chart1 = new XYChartBuilder().width(800).height(600)
                .title("Vapor fraction Plot")
                .xAxisTitle("Enthalpy (KJ/Kg)").yAxisTitle("Pressure (bar)").build();
        chart1.getStyler().setYAxisLogarithmic(true);
        chart1.getStyler().setYAxisMax(1e4);
        chart1.getStyler().setXAxisMax(4500d);
        plotWithColor(SPoints, chart1, "pt", minS,maxS);
        new SwingWrapper<>(chart1).displayChart();
    }

    private static void plotEntropyPH(){
        // Define ranges for P and H
        double TMin = 273 + 50, TMax = 800;

        double vMin = 2e-5, vMax = 1;

        int TSteps = 40;


        // Collect all T values to normalize for color gradient
        List<PointData> SPoints = new ArrayList<>();

        for (int i = 0; i <= TSteps; i++) {
            double T = TMin + i * (TMax - TMin) / TSteps;
            Couple<Double> Vs = PR_WATER_EOS.getSaturationVolumes(T);
            for (double V = vMin; V <= vMax; V*=1.005f) {
                double P = PR_WATER_EOS.pressure(T, V);
                if (V > Vs.getFirst() && V < Vs.getSecond()) {
                    P = PR_WATER_EOS.saturationPressure(T);

                }
                double S = PR_WATER_EOS.totalEntropy((float) T, (float) V);
                double H = PR_WATER_EOS.totalEnthalpy((float) T, (float) V);
                SPoints.add(new PointData(H/1e3,P/1e5, S));

            }
        }

        double minS = SPoints.stream().map(p -> p.Z).min(Double::compareTo).get();
        double maxS = SPoints.stream().map(p -> p.Z).max(Double::compareTo).get();
        System.out.println("minS " + minS);
        System.out.println("maxS: " + maxS);
        // Find min/max T for color scaling
        XYChart chart1 = new XYChartBuilder().width(800).height(600)
                .title("Entropy Plot")
                .xAxisTitle("Enthalpy (KJ/Kg)").yAxisTitle("Pressure (bar)").build();
        chart1.getStyler().setMarkerSize(6);
        chart1.getStyler().setLegendVisible(false);
        chart1.getStyler().setYAxisLogarithmic(true);
        chart1.getStyler().setYAxisMax(1e4);
        chart1.getStyler().setXAxisMax(4500d);
        plotWithColor(SPoints, chart1, "pt", minS,maxS);

        float heatingPower =5e6f;
        float compressionFactor = 1000f;
        SpecificRealGazState startPR = WaterCubicEOS.DEFAULT_STATE;

        ArrayList<SpecificRealGazState> compressionStatesPr = new ArrayList<>();
        int nbrOfStep = 500;
        float factorStep = (compressionFactor - 1) / nbrOfStep;
        for (float factor = 1; factor <= compressionFactor; factor += factorStep) {
            compressionStatesPr.add(WaterCubicEOS.isentropicCompression(startPR,factor));
        }
        chart1.addSeries("compression for PR", compressionStatesPr.stream().map((s) -> s.specificEnthalpy()/1e3).toList(),
                        compressionStatesPr.stream().map((s) -> s.pressure()/1e5).toList())
                .setMarker(SeriesMarkers.CIRCLE)
                .setLineStyle(SeriesLines.SOLID);


        SpecificRealGazState endOfHeatingPR = WaterCubicEOS.isobaricTransfer(
                compressionStatesPr.get(nbrOfStep-1),heatingPower);


        chart1.addSeries("heating for PR", List.of(compressionStatesPr.get(nbrOfStep-1).specificEnthalpy()/1e3, endOfHeatingPR.specificEnthalpy()/1e3),
                        List.of(compressionStatesPr.get(nbrOfStep-1).pressure()/1e5, endOfHeatingPR.pressure()/1e5))
                .setMarker(SeriesMarkers.CIRCLE)
                .setLineStyle(SeriesLines.SOLID);

        ArrayList<SpecificRealGazState> expansionStatesPr = new ArrayList<>(List.of(endOfHeatingPR));
        for (float factor = 1 + factorStep; factor <= compressionFactor; factor += factorStep) {
            //expansionStatesPr.add(WaterCubicEOSTransformationHelper.isentropicExpansion(expansionStatesPr.get(expansionStatesPr.size() - 1),factor/(factor - factorStep)));
            expansionStatesPr.add(WaterCubicEOS.isentropicExpansion(endOfHeatingPR,factor));

        }
        chart1.addSeries("expansion for PR",
                        expansionStatesPr.stream().map((s) -> s.specificEnthalpy()/1e3).toList(),
                        expansionStatesPr.stream().map((s) -> s.pressure()/1e5).toList())
                .setMarker(SeriesMarkers.CIRCLE)
                .setLineStyle(SeriesLines.SOLID);
        chart1.getStyler().setYAxisLogarithmic(true);

        new SwingWrapper<>(chart1).displayChart();
    }

    private static void plotWithColor(List<PointData> points, XYChart chart, String seriesName, Double minValue, Double maxValue) {
        // Create chart
        //chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);


        // Add one series per point (to assign different color)
        int pointIndex = 0;
        for (PointData pt : points) {
            Color color = getColorFromValue(pt.Z, minValue, maxValue);
            chart.addSeries(seriesName + (pointIndex++), new double[]{pt.X}, new double[]{pt.Y})
                    .setMarker(SeriesMarkers.CIRCLE)
                    .setMarkerColor(color).setLineStyle(SeriesLines.NONE);
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
                double H = WaterCubicEOS.get_h(0,(float) T, (float) P);
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
                .xAxisTitle("Temperature (T)").yAxisTitle("Molar Volume (H)").build(), "pt", points.stream().map(pt -> pt.Z).min(Double::compareTo).orElse(0.0), points.stream().map(pt -> pt.Z).max(Double::compareTo).orElse(1.0));
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

