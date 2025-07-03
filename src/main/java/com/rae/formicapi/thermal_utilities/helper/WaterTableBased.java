package com.rae.formicapi.thermal_utilities.helper;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.thermal_utilities.SpecificRealGazState;
import com.rae.formicapi.thermal_utilities.eos.PengRobinsonEOS;
import net.createmod.catnip.data.Couple;
import org.jetbrains.annotations.NotNull;

import static com.rae.formicapi.thermal_utilities.eos.EquationOfState.R;

/**
 * this is a mix bwn the EOS and tabulated results
 */
public class WaterTableBased {
    private static final PengRobinsonEOS EOS = new PengRobinsonEOS(647.1, 22.064e6,0.344,18.01528e-3f);

    public static final SpecificRealGazState DEFAULT_STATE = new SpecificRealGazState(300f, 101300f, (float) EOS.getEnthalpy(300f,101300f,0f),0f);

    public static SpecificRealGazState isobaricTransfer(SpecificRealGazState fluidState, float specific_heat) {
        if (specific_heat == 0) {
            return fluidState;
        }
        else {
            float newH = fluidState.specificEnthalpy() + specific_heat;
            float newPressure = fluidState.pressure();
            float newT = get_T(newPressure,newH);
            float newVaporQuality = get_x(newH,newT,newPressure);
            return new SpecificRealGazState(newT, newPressure, newH, newVaporQuality);
        }
    }

    public static float get_T(float P, float H) {
        return FormicAPI.WATER_PH_T.getValue(P,H);
    }

    /**
     * Computes enthalpy by mass for a given vapor quality x, temperature T, and pressure P.
     * @param x vapor fraction (0 = liquid, 1 = vapor)
     * @param T temperature in K
     * @param P pressure in Pa
     * @return specific enthalpy in J/kg
     */
    public static float get_h(float x, float T, float P) {
        return (float) EOS.getEnthalpy(T, P, x);
    }
    /**
     * Estimate vapor quality `x` from enthalpy at given T and P.
     * Returns:
     *   - 0 for pure liquid
     *   - 1 for pure vapor
     *   - in (0, 1) for two-phase mixture
     */
    public static float get_x(float h, float T, float P) {
        if (T < 100) T = 100;
        if (P < 1) P = 1;
        double[] roots = EOS.getZFactors(T, P);
        if (roots.length == 0) return 0;
        Couple<Double> Vms = EOS.getSaturationVolumes(T);
        if ( Math.abs(Vms.getSecond() - Vms.getFirst()) < 1e-8 ) {
            double V = roots[0] * R * T / P;
            return (float) ((V > Vms.getSecond()) ? 1.0 : 0.0);
        }
        try {
            double saturationPressure = EOS.saturationPressure(T);
            if (P < saturationPressure *0.99f) return  1f;
            if (P > saturationPressure * 1.01f) return   0f;
        } catch (RuntimeException ignored){

        }
        /*if (roots.length < 2) {
            // Not in coexistence region → determine if it's clearly vapor or liquid

            if (roots.length == 1) {
                ; // crude guess
            }

            FormicAPI.LOGGER.debug("Not in coexistence region and cannot infer phase, this shouldn't be  possible, check " +
                    "that the EOS is cubic : "+ Arrays.toString(roots)+ " T = "+ T + " P = "+ P);//should be impossible
            return 0; // defaulting to liquid
        }*/

        double hl = EOS.totalEnthalpy(T, Vms.getFirst());
        double hv = EOS.totalEnthalpy(T, Vms.getSecond());


        double denominator = hv - hl;
        if (Math.abs(denominator) < 1e-6) {
            // Avoid divide-by-zero
            return (float)((h > hl) ? 1.0 : 0.0);
        }

        float x = (float)((h - hl) / denominator);

        // Clamp to [0,1] to account for numerical error
        return Math.min(Math.max(x,0f), 1f);
    }
    public static float get_x_from_entropy(double s, float T, float P) {//this is wrong.
        if (T < 100) T = 100;
        if (P < 1) P = 1;
        double[] roots = EOS.getZFactors(T, P);

        if (roots.length == 0){
            //System.out.println("out of validity limits with T = "+ T + " P = " + P);
            return 0;//fallback to 0 because we are at a specific volume lower than b
        }
        Couple<Double> Vms = EOS.getSaturationVolumes(T);
        if ( Math.abs(Vms.getSecond() - Vms.getFirst()) < 1e-8 ) {
            double V = roots[0] * R * T / P;
            return  (float) ((V > Vms.getSecond()) ? 1.0 : 0.0);
        }

        try {
            double saturationPressure = EOS.saturationPressure(T);
            if (P < saturationPressure *0.99f) return  1f;
            if (P > saturationPressure * 1.01f) return   0f;
        } catch (RuntimeException ignored){
        }


        double sl = EOS.totalEntropy(T, Vms.getFirst());   // saturated liquid
        double sv = EOS.totalEntropy(T, Vms.getSecond());  // saturated vapor

        final double epsilon = 1e-6;
        if (Math.abs(sl - sv) < epsilon) {
            // Avoid division by zero in pathological or near-critical cases
            return s < sl ? 0.0f : 1.0f;
        }

        float x = (float) ((s - sl) / (sv - sl));
        return Math.max(0.0f, Math.min(1.0f, x));
    }

    private static @NotNull SpecificRealGazState isentropicPressureChange(float T1, float P1, float x1, float finalPressure) {
        // Step 1: Compute constant entropy from initial state
        float sTarget = (float) EOS.getEntropy(T1, P1, x1);

        // Step 2: Set parameters for stepping pressure
        float Tcurrent = FormicAPI.WATER_PS_T.getValue(finalPressure, sTarget);
        float xcurrent = get_x_from_entropy(sTarget, Tcurrent, finalPressure);
        // Step 3: Compute final enthalpy
        float hFinal = get_h(xcurrent, Tcurrent, finalPressure);
        return new SpecificRealGazState(Tcurrent, finalPressure, hFinal, xcurrent);
    }
    /**
     * adiabatic reversible expansion
     * @param initial
     * @param expansionFactor the initial pressure over the pressure of the fluid at the end of the turbine
     * @return the new fluid state
     */
    public static SpecificRealGazState isentropicExpansion(SpecificRealGazState initial, float expansionFactor) {

        return isentropicPressureChange(initial.temperature(), initial.pressure(), initial.vaporQuality(), initial.pressure() / expansionFactor);
    }


    /**
     * adiabatic expansion
     * @param fluidState :
     * @param isentropicYield : how much the fluid lost enthalpy over what it should have if reversible (how much energy was taken from it)
     * @param expansionCoef : the initial pressure over the pressure of the fluid at the end of the turbine
     * @return the new fluid state
     */
    public static SpecificRealGazState realExpansion(SpecificRealGazState fluidState, float isentropicYield, float expansionCoef){
        SpecificRealGazState revFluidState = isentropicExpansion(fluidState,expansionCoef);
        float reversibleDh = revFluidState.specificEnthalpy()- fluidState.specificEnthalpy();
        float losth = reversibleDh*(1-isentropicYield);
        return isobaricTransfer(revFluidState,-losth);
    }
    /**
     * adiabatic reversible compression
     * @param initial :
     * @param compressionFactor :the pressure of the fluid at the end of the turbine over the initial pressure
     * @return the new fluid state
     */
    public static SpecificRealGazState isentropicCompression(SpecificRealGazState initial, float compressionFactor) {

        return isentropicPressureChange(initial.temperature(), initial.pressure(),initial.vaporQuality(), initial.pressure() *compressionFactor);

    }

    /**
     * adiabatic compression
     * @param fluidState :
     * @param yield :how much the fluid gained enthalpy over what it should have if reversible (how much energy was put into it)
     * @param compressionCoef :the pressure of the fluid at the end of the turbine over the initial pressure
     * @return the new fluid state
     */
    public static SpecificRealGazState realCompression(SpecificRealGazState fluidState, float yield, float compressionCoef){
        SpecificRealGazState revFluidState = isentropicCompression(fluidState,compressionCoef);
        float reversibleDh = revFluidState.specificEnthalpy()- fluidState.specificEnthalpy();
        float losth = reversibleDh*(1-yield);
        return isobaricTransfer(revFluidState,-losth);
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
                get_T(P,h), P, h, get_x(h,get_T(P,h),P));
        //System.out.println(state);
        if (first.temperature().isNaN() || second.temperature().isNaN()){
            return DEFAULT_STATE;
        }
        if (first.temperature() > 20000 || second.temperature() > 20000){
            return DEFAULT_STATE;
        }
        return state;
    }


}
