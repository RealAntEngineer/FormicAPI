package com.rae.formicapi.thermal_utilities;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.math.Solvers;
import net.createmod.catnip.data.Couple;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.function.Function;

//TODO make an interface.
public class WaterCubicEOSTransformationHelper {

    private static final PengRobinsonEOS EOS = new PengRobinsonEOS(647.1, 22.064e6,0.344);
    private static final double M = 18.01528e-3f;
    private static final double Rs = PengRobinsonEOS.R / M;

    public static final SpecificRealGazState DEFAULT_STATE = new SpecificRealGazState(300f, 101300f, get_h(0,300,101300),0f);

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

    /**
     * Computes enthalpy by mass for a given vapor quality x, temperature T, and pressure P.
     * @param x vapor fraction (0 = liquid, 1 = vapor)
     * @param T temperature in K
     * @param P pressure in Pa
     * @return specific enthalpy in J/kg
     */
    public static float get_h(float x, float T, float P) {
        final double R = PengRobinsonEOS.R;

        double[] roots = EOS.getZFactors(T, P);

        if (x <= 0f) {
            // Pure liquid
            double Zl = Arrays.stream(roots).min().orElseGet(() -> EOS.minZ(T,P)); // best effort
            double Vl = Zl * R * T / P;
            double hl = EOS.totalEnthalpy(T, Vl);
            if (Double.isNaN(hl)) {
                System.out.println("found it");
            }
            return (float) (hl / M);
        }

        if (x >= 1f) {
            // Pure vapor
            double Zv = Arrays.stream(roots).max().orElseGet(() ->  EOS.minZ(T,P)); // best effort
            double Vv = Zv * R * T / P;
            double hv = EOS.totalEnthalpy(T, Vv);
            if (Double.isNaN(hv)) {
                System.out.println("found it");
            }
            return (float) (hv / M);
        }

        // Two-phase region expected
        if (roots.length < 2) {
            double Zv = roots[0]; // best effort
            double Vv = Zv * R * T / P;
            double hv = EOS.totalEnthalpy(T, Vv);
            FormicAPI.LOGGER.debug("state wrongly associated with x : "+ x + " when single phase");
            if (Double.isNaN(hv)) {
                System.out.println("found it");
            }
            return (float) (hv / M);
        }

        double Zl = Arrays.stream(roots).min().getAsDouble();
        double Zv = Arrays.stream(roots).max().getAsDouble();

        double Vl = Zl * R * T / P;
        double Vv = Zv * R * T / P;

        double hl = EOS.totalEnthalpy(T, Vl);
        double hv = EOS.totalEnthalpy(T, Vv);

        double hMix = (1.0 - x) * hl + x * hv;
        if (Double.isNaN(hMix)) {
            System.out.println("found it");
        }
        return (float) (hMix / M);
    }
    public static float get_T(float P, float h) {
        //change that to a gradient decent + ensure we are using a by mass enthalpy and not a molar enthalpy
        //this doesn't work -> it gives weird results
        // Define residual: f(T) = h(T) - h_target
        Function<Float, Float> residual = (Float TGuess) -> {
            if (TGuess.isNaN()) {
                System.out.println("TGuess : " + TGuess);
            }
            float computedX = get_x(h,TGuess,P);
            if (Float.isNaN(computedX)) {
                System.out.println("computedX : " + computedX);
            }
            float computedH = get_h(computedX,TGuess,P);
            if (Float.isNaN(computedH)) {
                System.out.println("computedH : " + computedH);
            }
            return Math.abs(h - computedH);
            /*try {
                double[] roots = EOS.getZFactors(Tguess, P);
                if (roots.length == 1) {
                    // Single phase
                    double Z = roots[0];
                    double Vm = Z * PengRobinsonEOS.R * Tguess / P;
                    double hComputed = EOS.totalEnthalpy(Tguess, Vm) * M;//this is a molar enthalpy we need a by mass
                    return (float) Math.abs(hComputed - h);
                } else if (roots.length >= 2) {
                    Couple<Double> V = EOS.getSaturationVolumes(Tguess);

                    double hl = EOS.totalEnthalpy(Tguess, V.get(true)) * M;
                    double hv = EOS.totalEnthalpy(Tguess, V.get(false)) * M;
                    double xGuess = get_x(h, Tguess, P);
                    // If h within [hl, hv], calculate quality and mixture enthalpy
                    if (h >= hl && h <= hv) {
                        return 0f; // found T: h is between saturated enthalpies
                    }

                    // Otherwise, pick closest pure-phase approximation
                    double hClosest = (Math.abs(h - hl) < Math.abs(h - hv)) ? hl : hv;
                    System.out.println("weird choice + error : "+ Math.abs(hClosest - h)+ " at T = "+ Tguess+ " with hv "+ hv + " hl "+ hl);
                    return (float)Math.abs(hClosest - h);
                }
            } catch (Exception e) {
                return Float.MAX_VALUE;
            }
            return Float.MAX_VALUE;*/
        };

        float T = Solvers.gradientDecent(residual, 273f, 0.1f, 0.01f, 0.1f);
        if (Float.isNaN(T)) throw new RuntimeException("No solution for T at given h and P");

        return T;
    }
    /**
     * Estimate vapor quality `x` from enthalpy at given T and P.
     * Returns:
     *   - 0 for pure liquid
     *   - 1 for pure vapor
     *   - in (0, 1) for two-phase mixture
     */
    public static float get_x(float h, float T, float P) {
        double[] roots = EOS.getZFactors(T, P);
        if (roots.length < 2) {
            // Not in coexistence region → determine if it's clearly vapor or liquid
            if (T < 0) T = 1e-3f;
            if (P < 0) P = 1e-3f;
            if (roots.length == 1) {
                double V = EOS.getSaturationVolumes(T).get(true);
                double hSingle = EOS.totalEnthalpy(T, V) / M;
                return (float) ((h > hSingle) ? 1.0 : 0.0); // crude guess
            }

            FormicAPI.LOGGER.debug("Not in coexistence region and cannot infer phase, this shouldn't be  possible, check " +
                    "that the EOS is cubic : "+ Arrays.toString(roots)+ " T = "+ T + " P = "+ P);//should be impossible
            return 0; // defaulting to liquid
        }
        Couple<Double> V = EOS.getSaturationVolumes(T);

        double hl = EOS.totalEnthalpy(T, V.get(true)) / M;
        double hv = EOS.totalEnthalpy(T, V.get(false)) / M;

        double denominator = hv - hl;
        if (Math.abs(denominator) < 1e-6) {
            // Avoid divide-by-zero
            return (float)((h > hl) ? 1.0 : 0.0);
        }

        float x = (float)((h - hl) / denominator);

        // Clamp to [0,1] to account for numerical error
        if (x < 0f) return 0f;
        if (x > 1f) return 1f;
        return x;
    }
    /**
     * Estimate vapor quality `x` from entropy at given T and P.
     * Returns:
     *   - 0 for pure liquid
     *   - 1 for pure vapor
     *   - in (0, 1) for two-phase mixture
     */
    public static float get_x_from_entropy(double s, float T) {
        Couple<Double> V = EOS.getSaturationVolumes(T); // doesn't work at high V.
        double sl = EOS.totalEntropy(T, V.get(true));   // saturated liquid
        double sv = EOS.totalEntropy(T, V.get(false));  // saturated vapor

        final double epsilon = 1e-6;
        if (Math.abs(sl - sv) < epsilon) {
            // Avoid division by zero in pathological or near-critical cases
            return s < sl ? 0.0f : 1.0f;
        }

        float x = (float) ((s - sl) / (sv - sl));

        return Math.max(0.0f, Math.min(1.0f, x));
    }
    private static @NotNull SpecificRealGazState isentropicPressureChange(float T1, float P1, float x1, float finalPressure) {
        // Step 1: Initial entropy
        //we should handle the intermediate gracefully
        double sTarget = getEntropy(T1, P1, x1);
        System.out.println("sTarget :"+sTarget);


        // Step 2: Find T2 such that s(T2, P2) = s1
        //maybe do
        Function<Float, Float> entropyError = (Float Tguess) -> {
            try {
                float x = get_x_from_entropy(sTarget, Tguess);//potential big error if 2 phase
                double sFound  = getEntropy(Tguess, finalPressure, x);
                return (float)Math.abs(sFound-sTarget);
            } catch (Exception e) {
                return Float.MAX_VALUE;
            }
        };

        // You could tighten bounds depending on the expected fluid range
        //maybe a dichotomie ? bwn 0 and 2000
        float T2 = Solvers.gradientDecent(entropyError, T1,0.01f, 0.001f, 1e-1f);
        System.out.println("remaining error : "+ entropyError.apply(T2));

        Couple<Double> V = EOS.getSaturationVolumes(T2);
        System.out.println("V : "+ V);
        double sl = EOS.totalEntropy(T2, V.get(true));   // saturated liquid
        double sv = EOS.totalEntropy(T2, V.get(false));  // saturated vapor
        System.out.println("sl : "+sl + " sv : "+sv);
        if (Float.isNaN(T2)) throw new RuntimeException("Failed to find T for isentropic expansion");

        // Step 3: Get phase info and build final state
        float x, h;
            x = get_x_from_entropy(sTarget, T2);
            h = get_h(x, T2, finalPressure);

        return new SpecificRealGazState(T2, finalPressure, h, x);
    }

    private static double getEntropy(float T1, float P1, float x1) {
        double [] roots1 =  EOS.getZFactors(T1, P1);
        double sTarget;
        if (x1 == 1){
            double Z1 = Arrays.stream(roots1).max().getAsDouble();
            double Vm1 = Z1 * PengRobinsonEOS.R * T1 / P1;
            sTarget = EOS.totalEntropy(T1, Vm1);
        }
        else if (x1 == 0){
            double Z1 = Arrays.stream(roots1).min().getAsDouble();
            double Vm1 = Z1 * PengRobinsonEOS.R * T1 / P1;
            sTarget = EOS.totalEntropy(T1, Vm1);
        }
        else {
            Couple<Double> Vms = EOS.getSaturationVolumes(T1);
            double Vm1 = Vms.getFirst() * (1- x1) + Vms.getSecond() * x1;
            sTarget = EOS.totalEntropy(T1, Vm1);
        }
        return sTarget;
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
        System.out.println("first "+first+ " second"+second);
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