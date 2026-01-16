package com.rae.formicapi.new_thermalmodels;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.data.managers.TwoDSparceTabulatedFunctionLoader;
import com.rae.formicapi.thermal_utilities.SpecificRealGazState;
import net.minecraftforge.event.AddReloadListenerEvent;
import org.jetbrains.annotations.NotNull;

public class FullTableBased {
    private static final TwoDSparceTabulatedFunctionLoader WATER_HP_T = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID,"water/hp_to_t");
    private static final TwoDSparceTabulatedFunctionLoader WATER_HP_S = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID,"water/hp_to_s");
    private static final TwoDSparceTabulatedFunctionLoader WATER_HP_X = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID,"water/hp_to_x");
    private static final TwoDSparceTabulatedFunctionLoader WATER_SP_H = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID,"water/sp_to_h");

    public static float getH(float S, float P){
        return WATER_SP_H.getValue(S,P);
    };

    public static float getS(float H, float P){
        return WATER_HP_S.getValue(H,P);
    };

    public static float getT(float H, float P){
        return WATER_HP_T.getValue(H,P);
    };

    public static float getX(float H, float P){
        return WATER_HP_X.getValue(H,P);
    };


    private static @NotNull SpecificRealGazState isentropicPressureChange(float H1, float P1, float finalPressure) {
        float sTarget = (float) getS(H1, P1);

        float hFinal = getH(sTarget, finalPressure);
        float Tfinal = getT(hFinal,finalPressure);
        float xFinal = getX(hFinal,Tfinal);
        return new SpecificRealGazState(Tfinal, finalPressure, hFinal, xFinal);
    }
    /**
     * adiabatic reversible expansion
     * @param initial : initial state
     * @param expansionFactor the initial pressure over the pressure of the fluid at the end of the turbine
     * @return the new fluid state
     */
    public static SpecificRealGazState isentropicExpansion(SpecificRealGazState initial, float expansionFactor) {

        return isentropicPressureChange(initial.specificEnthalpy(), initial.pressure(), initial.pressure() / expansionFactor);
    }

    /**
     * adiabatic reversible compression
     * @param initial : initial state
     * @param compressionFactor :the pressure of the fluid at the end of the turbine over the initial pressure
     * @return the new fluid state
     */
    public static SpecificRealGazState isentropicCompression(SpecificRealGazState initial, float compressionFactor) {

        return isentropicPressureChange(initial.specificEnthalpy(), initial.pressure(), initial.pressure() *compressionFactor);

    }

    public static SpecificRealGazState isobaricTransfer(SpecificRealGazState fluidState, float specific_heat) {
        if (specific_heat == 0) {
            return fluidState;
        }
        else {
            float newH = fluidState.specificEnthalpy() + specific_heat;
            float newPressure = fluidState.pressure();
            float newT = getT(newH, newPressure);
            float newVaporQuality = getX(newH,newPressure);
            return new SpecificRealGazState(newT, newPressure, newH, newVaporQuality);
        }
    }



    public static void addReloadListeners(AddReloadListenerEvent event) {
        event.addListener(WATER_HP_T);
        event.addListener(WATER_HP_S);
        event.addListener(WATER_HP_X);
        event.addListener(WATER_SP_H);
    }
}