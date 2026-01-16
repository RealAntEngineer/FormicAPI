package com.rae.formicapi;

import com.rae.formicapi.new_thermalmodels.EquationOfState;
import com.rae.formicapi.new_thermalmodels.PengRobinsonEOS;
import net.createmod.catnip.data.Couple;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.None;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MakeChartsV2 {
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
                    List<Double> spinodals = pr.findSpinodalPoints(T);
                    System.out.println("pressure "+ pr.saturationPressure(T));

                    if (!spinodals.isEmpty()) {
                        //TODO derivation of cubic state equation = we just find the root of the derivative t
                        double Pcoex = pr.saturationPressure(T);//computeCoexistencePressure(pr, T, Math.max(0,eos.pressure(T, spinodals.get(0))), eos.pressure(T, spinodals.get(1))); // Pa
                        double Vl = pr.liquidVolume(T, Pcoex); // saturated liquid
                        double Vv = pr.vaporVolume(T, Pcoex); // saturated vapor
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
                    } else {
                        System.out.println("spinodals not found "+ pr.saturationPressure(T));
                    }
                } catch (Exception ignored) {
                    System.out.println(ignored);
                }
            }
        }


        new SwingWrapper<>(chart).displayChart();
    }

    public static void main(String[] args) {
        double Tc = 647.1;//critical temperature in Kelvin
        double Pc = 22.064e6;//critical pressure in Pascals
        double omega = 0.344;//omega.... What is it exactly ?
        double M = 18.01528e-3f;
        PengRobinsonEOS waterPrEOS = new PengRobinsonEOS(Tc, Pc, omega, M);
        //VanDerWaalsEOS water2EOS = EOSLibrary.getVanDerWaalsWaterEOS();

        double[] temperatures = new double[]{300, 480}; // Kelvin
        double vMin = 2.5e-5/M;      // m³/Kg
        double vMax = 2e-2/M;      // m³/Kg
        plotIsotherms(waterPrEOS,"Peng-Robinson", temperatures,-Pc*1f,Pc * 1.5, vMin, vMax);
        plotSaturation();
    }

    private static void plotSaturation() {
        double Tc = 647.1;//critical temperature in Kelvin
        double Pc = 22.064e6;//critical pressure in Pascals
        double omega = 0.344;//omega.... What is it exactly ?
        double M = 18.01528e-3f;
        PengRobinsonEOS waterPrEOS = new PengRobinsonEOS(Tc, Pc, omega, M);
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
            Couple<Double> Vs = waterPrEOS.saturationVolumes(T);
            //System.out.println(Vs.getFirst());
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
