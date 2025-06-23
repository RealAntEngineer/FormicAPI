package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.eos.EquationOfState;

import com.rae.formicapi.thermal_utilities.eos.PengRobinsonEOS;
import net.createmod.catnip.data.Couple;
import org.knowm.xchart.*;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.None;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.util.*;


public class MakeCharts {
    public static void plotIsotherms(EquationOfState eos,String name, double[] temperatures,double pMin,double pMax, double vMin, double vMax) {
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title(name)
                .xAxisTitle("Molar Volume (m³/mol)")
                .yAxisTitle("Pressure (Pa)")
                .build();

        chart.getStyler().setLegendVisible(true);
        chart.getStyler().setXAxisLogarithmic(true);
        chart.getStyler().setMarkerSize(20);
        chart.getStyler().setYAxisMin(pMin);
        chart.getStyler().setYAxisMax(pMax);

        for (double T : temperatures) {
            List<Double> volumes = new ArrayList<>();
            List<Double> pressures = new ArrayList<>();

            for (double V = vMin; V <= vMax; V *= 1.01) {
                try {
                    double P = eos.pressure(T, V);
                    //if (P > 0) {
                        volumes.add(V);
                        pressures.add(P);
                    //}
                } catch (Exception ignored) {
                }
            }
            if (!volumes.isEmpty())
                chart.addSeries(String.format("T = %.0f K", T), volumes, pressures)
                    .setMarker(new None());

            if (eos instanceof PengRobinsonEOS pr) {
                try {
                    List<Double> spinodals = findSpinodalPoints(eos, T, vMin, vMax);
                    if (!spinodals.isEmpty()) {
                        //TODO derivation of cubic state equation = we just find the root of the derivative t
                        double Pcoex = computeCoexistencePressure(pr, T, Math.max(0,eos.pressure(T, spinodals.get(0))), eos.pressure(T, spinodals.get(1))); // Pa
                        double Vl = pr.volumeMolar(T, Pcoex, 0.0); // saturated liquid
                        double Vv = pr.volumeMolar(T, Pcoex, 1.0); // saturated vapor
                        if (Vl < 0 || Vv < 0) throw new IllegalStateException("Vl or Vv is negative");
                        chart.addSeries(String.format("Saturated Liquid @ %.0f K", T), List.of(Vl), List.of(Pcoex))
                                .setMarker(SeriesMarkers.CIRCLE)
                                .setLineStyle(SeriesLines.NONE);

                        chart.addSeries(String.format("Saturated Vapor @ %.0f K", T), List.of(Vv), List.of(Pcoex))
                                .setMarker(SeriesMarkers.SQUARE)
                                .setLineStyle(SeriesLines.NONE);
                        chart.addSeries(
                                        String.format("Equilibrium @ %.0f K", T),
                                        Arrays.asList(Vl, Vv),
                                        Arrays.asList(Pcoex, Pcoex)
                                ).setMarker(SeriesMarkers.NONE)
                                .setLineStyle(SeriesLines.DASH_DASH);
                    }
                } catch (Exception ignored) {
                    System.out.println(ignored);
                }
            }
        }


        new SwingWrapper<>(chart).displayChart();
    }
    public static List<Double> findSpinodalPoints(EquationOfState eos, double T, double vMin, double vMax) {
        List<Double> spinodals = new ArrayList<>();

        double previousV = vMin;
        double previousP = eos.pressure(T, previousV);
        double previousSlope = Double.NaN;

        for (double V = vMin * 1.01; V <= vMax; V *= 1.01) {
            double P = eos.pressure(T, V);
            double slope = (P - previousP) / (V - previousV);

            if (!Double.isNaN(previousSlope) && previousSlope * slope <= 0) {
                // slope changed sign → inflection/spinodal
                spinodals.add(V);
            }

            previousV = V;
            previousP = P;
            previousSlope = slope;
        }

        return spinodals;
    }
    public static double computeCoexistencePressure(PengRobinsonEOS eos, double T, double pMin, double pMax) {
        /*Function<Float, Float> areaDifferenceFunction = (Float Guess) -> {
            double P = Guess;
            double[] roots = eos.getZFactors(T, P);
            if (roots.length < 2) throw new RuntimeException("Can't compute coexistence pressure outside LV phase");

            double Zl = Arrays.stream(roots).min().getAsDouble();
            double Zv = Arrays.stream(roots).max().getAsDouble();

            double Vl = Zl * PengRobinsonEOS.R * T / P;
            double Vv = Zv * PengRobinsonEOS.R * T / P;

            if (Vv - Vl < 1e-9) throw new RuntimeException("liquid and gaz state are too close");

            int n = 1000;
            double dV = (Vv - Vl) / n;
            //it's an integral and we already have the eos so why not doing it directly with a formula ?
            double area = 0;
            for (int i = 0; i < n; i++) {
                double V1 = Vl + i * dV;
                double V2 = V1 + dV;
                double p1 = eos.pressure(T, V1);
                double p2 = eos.pressure(T, V2);
                area += 0.5 * (p1 + p2) * dV;
            }

            double rect = P * (Vv - Vl);
            return (float)(area - rect);
        };

        double epsilon = (pMax - pMin) * 0.01f; // acceptable pressure precision (Pa)
        float PCoexistence = Solvers.dichotomy(areaDifferenceFunction, (float)pMin, (float)pMax, (float) epsilon);

        if (Float.isNaN(PCoexistence)) {
            throw new RuntimeException("Dichotomy failed: no sign change or invalid region.");
        }

        return PCoexistence;*/

        return eos.saturationPressure(T);
    }


    public static void main(String[] args) {
        PengRobinsonEOS waterPrEOS = EOSLibrary.getPRWaterEOS();
        //VanDerWaalsEOS water2EOS = EOSLibrary.getVanDerWaalsWaterEOS();

        double[] temperatures = new double[]{300, 480}; // Kelvin
        double vMin = 2.5e-5;      // m³/mol
        double vMax = 2e-2;      // m³/mol
        double Pc = 22e6;
        plotIsotherms(waterPrEOS,"Peng-Robinson", temperatures,-Pc*1f,Pc * 1.5, vMin, vMax);
        plotSaturation();
    }

    private static void plotSaturation() {
        PengRobinsonEOS waterPrEOS = EOSLibrary.getPRWaterEOS();
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Saturation")
                .xAxisTitle("Molar Volume")
                .yAxisTitle("Pressure")
                .build();

        chart.getStyler().setXAxisLogarithmic(false);
        chart.getStyler().setLegendVisible(false);
        for (float T = 300 ; T <= 645; T+=5){
            Couple<Double> Vs = waterPrEOS.getSaturationVolumes(T);
            System.out.println(Vs.getFirst());
            if (Vs.get(true) > waterPrEOS.getB() && Vs.get(true) < 10) {

                chart.addSeries("liquid " + T, List.of(Vs.get(true)), List.of(waterPrEOS.pressure(T, Vs.get(true))))
                        .setMarker(SeriesMarkers.CIRCLE)
                        .setLineStyle(SeriesLines.NONE);
            }
            if (Vs.get(false) >  waterPrEOS.getB()  && Vs.get(false) < 10) {

                chart.addSeries("vapor " + T, List.of(Vs.get(false)), List.of(waterPrEOS.pressure(T, Vs.get(false))))
                        .setMarker(SeriesMarkers.SQUARE)
                        .setLineStyle(SeriesLines.NONE);
            }
        }
        new SwingWrapper<>(chart).displayChart();
    }
}
