package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.SpecificRealGazState;
import com.rae.formicapi.thermal_utilities.WaterAsRealGazTransformationHelper;
import com.rae.formicapi.thermal_utilities.WaterCubicEOSTransformationHelper;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.util.ArrayList;
import java.util.List;

public class MakeThermalCycleCharts {



    public static void main(String[] args) {
        cycle();
        //smallTest();

    }

    private static void smallTest() {
        SpecificRealGazState startPR = WaterCubicEOSTransformationHelper.DEFAULT_STATE;
        System.out.println("Saturation pressure: " + EOSLibrary.getPRWaterEOS().computeSaturationPressure(300));
        System.out.println("Starting PR: " + startPR);
        SpecificRealGazState compressionStatesPr = WaterCubicEOSTransformationHelper.isentropicCompression(startPR,3);
        System.out.println("Compression States: " + compressionStatesPr);

         SpecificRealGazState endOfHeatingPR = WaterCubicEOSTransformationHelper.isobaricTransfer(
                compressionStatesPr,4e6f);
        System.out.println("End of Heating States: " + endOfHeatingPR);
        SpecificRealGazState expansionStatesPr = WaterCubicEOSTransformationHelper.isentropicExpansion(endOfHeatingPR,3);
        System.out.println("Expansion States: " + expansionStatesPr);
    }

    private static void cycle() {
        XYChart PHchart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("P-H")
                .xAxisTitle("Massic enthalpy (J/Kg)")
                .yAxisTitle("Pressure (Pa)")
                .build();
        float heatingPower = 2e6f;
        float compressionFactor = 10f;
        SpecificRealGazState startPR = WaterCubicEOSTransformationHelper.DEFAULT_STATE;
        SpecificRealGazState startOld = WaterAsRealGazTransformationHelper.DEFAULT_STATE;
        System.out.println(startPR);
        ArrayList<SpecificRealGazState> compressionStatesPr = new ArrayList<>();
        ArrayList<SpecificRealGazState> compressionStatesOld = new ArrayList<>();
        for ( float factor = 1; factor <= compressionFactor; factor +=  (compressionFactor-1)/100) {
            compressionStatesPr.add(WaterCubicEOSTransformationHelper.isentropicCompression(startPR,factor));
            compressionStatesOld.add(WaterAsRealGazTransformationHelper.standardCompression(startOld,factor));
        }
        PHchart.addSeries("compression for PR", compressionStatesPr.stream().map(SpecificRealGazState::specificEnthalpy).toList(),
                compressionStatesPr.stream().map(SpecificRealGazState::pressure).toList())
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);
        PHchart.addSeries("compression for Old", compressionStatesOld.stream().map(SpecificRealGazState::specificEnthalpy).toList(),
                        compressionStatesOld.stream().map(SpecificRealGazState::pressure).toList())
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);
        System.out.println(compressionStatesPr);
        SpecificRealGazState endOfHeatingPR = WaterCubicEOSTransformationHelper.isobaricTransfer(
                compressionStatesPr.get(99),heatingPower);
        SpecificRealGazState endOfHeatingOld = WaterAsRealGazTransformationHelper.isobaricTransfert(
                compressionStatesOld.get(99),heatingPower);
        System.out.println(endOfHeatingPR);
        PHchart.addSeries("heating for PR", List.of(compressionStatesPr.get(99).specificEnthalpy(), endOfHeatingPR.specificEnthalpy()),
                        List.of(compressionStatesPr.get(99).pressure(), endOfHeatingPR.pressure()))
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);
        PHchart.addSeries("heating for Old", List.of(compressionStatesOld.get(99).specificEnthalpy(), endOfHeatingOld.specificEnthalpy()),
                        List.of(compressionStatesOld.get(99).pressure(), endOfHeatingOld.pressure()))
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);

        ArrayList<SpecificRealGazState> expansionStatesPr = new ArrayList<>();
        ArrayList<SpecificRealGazState> expansionStatesOld = new ArrayList<>();
        for ( float factor = 1; factor <= compressionFactor; factor +=  (compressionFactor-1)/100) {
            expansionStatesPr.add(WaterCubicEOSTransformationHelper.isentropicExpansion(endOfHeatingPR,factor));
            expansionStatesOld.add(WaterAsRealGazTransformationHelper.standardExpansion(endOfHeatingOld,factor));
        }
        PHchart.addSeries("expansion for PR",
                        expansionStatesPr.stream().map(SpecificRealGazState::specificEnthalpy).toList(),
                        expansionStatesPr.stream().map(SpecificRealGazState::pressure).toList())
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);
        PHchart.addSeries("expansion for Old",
                        expansionStatesOld.stream().map(SpecificRealGazState::specificEnthalpy).toList(),
                        expansionStatesOld.stream().map(SpecificRealGazState::pressure).toList())
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);

        System.out.println(expansionStatesPr);

        new SwingWrapper<>(PHchart).displayChart();
    }
}
