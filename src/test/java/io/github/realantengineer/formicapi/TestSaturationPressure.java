package io.github.realantengineer.formicapi;

import io.github.realantengineer.formicapi.thermal_utilities.EOSLibrary;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.util.ArrayList;
import java.util.List;

public class TestSaturationPressure {
    public static void main(String[] args) {
        plotSaturationPressure();
    }
    private static void plotPressureAndSpinodals(){
        double TMin = 273.15;
        double TMax = 640;
        double dT = (TMax - TMin)/3;

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Saturation")
                .xAxisTitle("Temperature (°C)")
                .yAxisTitle("Pressure")
                .build();
        double VMin = EOSLibrary.getPRWaterEOS().getB()*1.01, VMax = 1e-2;
        for (double i = TMin; i < TMax; i+=  dT) {
            ArrayList<Double> V_List = new ArrayList<>();
            ArrayList<Double> P_List = new ArrayList<>();
            for (double V = VMin; V < VMax; V *= 1.01) {
                V_List.add(V);
                P_List.add(EOSLibrary.getPRWaterEOS().pressure(i, V));
                //P_List.add(EOSLibrary.getPRWaterEOS().computeSaturationPressure(i) / 101300);

            }
            chart.addSeries("pressure " + i, V_List, P_List)
                    .setMarker(SeriesMarkers.NONE);
            ;
            try {

                List<Double> Vms = EOSLibrary.getPRWaterEOS().findSpinodalPoints(i);

                chart.addSeries("spinodals " + i, List.of(Vms.get(0), Vms.get(1)),
                                List.of(EOSLibrary.getPRWaterEOS().pressure(i, Vms.get(0)), EOSLibrary.getPRWaterEOS().pressure(i, Vms.get(1))))
                        .setLineStyle(SeriesLines.NONE);

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        chart.getStyler().setXAxisLogarithmic(true);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setXAxisMin(VMin);
        chart.getStyler().setYAxisMin(-2e7);
        chart.getStyler().setXAxisMax(VMax);
        chart.getStyler().setYAxisMax(1e7);
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotSaturationPressure() {
        double TMin = 273.15;
        double TMax = 640;
        double dT = (TMax - TMin)/100;
        ArrayList<Double> T_List = new ArrayList<>();
        ArrayList<Double> P_List = new ArrayList<>();
        for (double i = TMin; i < TMax; i+=  dT) {
            try {
                T_List.add(i - 273.15);
                P_List.add(EOSLibrary.getPRWaterEOS().saturationPressure(i) / 101300);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Saturation")
                .xAxisTitle("Temperature (°C)")
                .yAxisTitle("Pressure")
                .build();

        chart.getStyler().setXAxisLogarithmic(false);
        chart.getStyler().setYAxisLogarithmic(true);
        chart.getStyler().setLegendVisible(false);

        chart.addSeries("saturation ", T_List, P_List);
        new SwingWrapper<>(chart).displayChart();
    }
}
