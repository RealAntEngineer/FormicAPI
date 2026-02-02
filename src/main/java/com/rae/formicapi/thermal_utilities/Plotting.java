package com.rae.formicapi.thermal_utilities;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.plotting.SimplePlot;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;

import java.util.ArrayList;
import java.util.List;

/**
 * Plotting utilities using the SimplePlot API
 */
@OnlyIn(Dist.CLIENT)
public class Plotting {

    /**
     * Plot isentropes (Pressure vs Enthalpy) using SimplePlot
     */
    public static void plotIsentropesPH(String name, double[] entropies, float pMin, float pMax) {

        SimplePlot plot = new SimplePlot()
                .title(name)
                .xlabel("Enthalpy (J/Kg)")
                .ylabel("Pressure (Pa)")
                .yLog(true)
                .width(800)
                .height(600);

        for (double s : entropies) {
            List<Double> pressures = new ArrayList<>();
            List<Double> enthalpies = new ArrayList<>();

            for (float p = pMin; p <= pMax; p *= 1.01f) {
                try {
                    float h = FullTableBased.getH((float) s, p);
                    if (!Float.isFinite(h)) continue;

                    pressures.add((double)p);
                    enthalpies.add((double)h);
                } catch (Exception ignored) {}
            }

            plot.addSeries(String.format("s=%.2f kJ/kg·K", s), enthalpies, pressures);
        }

        plot.save("isentropes_ph.png");
    }

    /**
     * Plot isentropes (Pressure vs Temperature) using SimplePlot
     */
    public static void plotIsentropesPT(String name, double[] entropies, float pMin, float pMax) {

        SimplePlot plot = new SimplePlot()
                .title(name)
                .xlabel("Temperature (K)")
                .ylabel("Pressure (Pa)")
                .yLog(true)
                .width(800)
                .height(600);

        for (double s : entropies) {
            List<Double> pressures = new ArrayList<>();
            List<Double> temperatures = new ArrayList<>();

            for (float p = pMin; p <= pMax; p *= 1.01f) {
                try {
                    float T = FullTableBased.getT(p, (float)FullTableBased.getH((float) s, p));
                    if (!Float.isFinite(T)) continue;

                    pressures.add((double)p);
                    temperatures.add((double)T);
                } catch (Exception ignored) {}
            }

            plot.addSeries(String.format("s=%.2f kJ/kg·K", s), temperatures, pressures);
        }
        List<Double> pressures = new ArrayList<>();
        List<Double> temperatures = new ArrayList<>();

        /*for (float p = pMin; p <= pMax; p *= 1.01f) {
            try {
                float T = EOSLibrary.getPRWaterEOS().saturationTemperature(p);
                if (!Float.isFinite(T)) continue;

                pressures.add((double)p);
                temperatures.add((double)T);
            } catch (Exception ignored) {}
        }

        plot.addSeries(String.format("saturation"), temperatures, pressures);*/



        plot.save("isentropes_pt.png");
    }
}
