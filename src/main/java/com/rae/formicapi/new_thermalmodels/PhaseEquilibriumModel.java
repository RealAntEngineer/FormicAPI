package com.rae.formicapi.new_thermalmodels;

/**
 * Phase equilibrium model (mass-based)
 */
public interface PhaseEquilibriumModel {

    /** Saturation pressure [Pa] at temperature T [K] */
    double saturationPressure(double T);

    /** Liquid and vapor volumes [m³/kg] */
    double liquidVolume(double T, double P);
    double vaporVolume(double T, double P);

    /** Two-phase mass-based molar volume */
    default double volumeMass(double T, double P, double x) {
        if (x <= 0) return liquidVolume(T,P);
        if (x >= 1) return vaporVolume(T,P);
        double Vl = liquidVolume(T,P);
        double Vv = vaporVolume(T,P);
        return (1 - x) * Vl + x * Vv;
    }
}