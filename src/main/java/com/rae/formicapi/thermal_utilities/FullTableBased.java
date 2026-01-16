package com.rae.formicapi.thermal_utilities;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.data.managers.TwoDSparceTabulatedFunctionLoader;
import net.minecraftforge.event.AddReloadListenerEvent;
import org.jetbrains.annotations.NotNull;

public class FullTableBased {
    public static final SpecificRealGazState DEFAULT_STATE = new SpecificRealGazState(300f, 101300f, 112665f,0f);

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


    //TODO -> it seems to not be working when amount are too low -> protection against 0 values ?
    public static SpecificRealGazState mix(SpecificRealGazState first, float firstAmount, SpecificRealGazState second, float secondAmount){
        //System.out.println("first "+first+ " second"+second);
        if (firstAmount == 0) return second;
        if (secondAmount == 0) return first;
        float P = first.pressure()*firstAmount/(firstAmount+ secondAmount) + second.pressure()*secondAmount/(firstAmount+ secondAmount);
        float h = first.specificEnthalpy()*firstAmount/(firstAmount+ secondAmount) + second.specificEnthalpy()*secondAmount/(firstAmount+ secondAmount);
        //float x = first.vaporQuality()*firstAmount/(firstAmount+ secondAmount) + second.vaporQuality()*secondAmount/(firstAmount+ secondAmount);
        SpecificRealGazState state = new SpecificRealGazState(
                getT(h,P), P, h, getX(h,P));
        //System.out.println(state);
        if (first.temperature().isNaN() || second.temperature().isNaN()){
            return DEFAULT_STATE;
        }
        if (first.temperature() > 20000 || second.temperature() > 20000){
            return DEFAULT_STATE;
        }
        return state;
    }

    public static void addReloadListeners(AddReloadListenerEvent event) {
        event.addListener(WATER_HP_T);
        event.addListener(WATER_HP_S);
        event.addListener(WATER_HP_X);
        event.addListener(WATER_SP_H);
    }
}