package com.rae.formicapi.new_thermalmodels;

import net.createmod.catnip.data.Couple;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract mass-based cubic EOS.
 * Converts all molar quantities to mass units internally.
 */
public abstract class CubicEOS implements EquationOfState, ResidualProperties , PhaseEquilibriumModel{

    protected final double M;   // [kg/mol]
    protected final double Tc;  // [K]
    protected final double Pc;  // [Pa]

    protected CubicEOS(double M, double Tc, double Pc) {
        this.M = M;
        this.Tc = Tc;
        this.Pc = Pc;
    }

    // ====== Spinodal points in mass units ======

    /**
     * Returns 2 spinodal volumes [m³/kg] at temperature T.
     * Searches between vMin and vMax geometrically.
     */
    List<Double> findSpinodalPoints(double T, double vMin, double vMax) {
        List<Double> spinodals = new ArrayList<>();
        double prevV = vMin;
        double prevP = pressure(T, prevV);
        double prevSlope = Double.NaN;

        for (double Vm = vMin; Vm <= vMax; Vm *= 1.05) {
            double P = pressure(T, Vm);
            double slope = (P - prevP) / (Vm - prevV);

            if (!Double.isNaN(prevSlope) && prevSlope * slope <= 0) {
                spinodals.add(Vm);
            }

            prevV = Vm;
            prevP = P;
            prevSlope = slope;

            if (spinodals.size() == 2) break;
        }

        return spinodals;
    }
    public abstract List<Double> findSpinodalPoints(double T);

    /** Approximate critical volume [m³/kg] */
    public double criticalVolumeMass() {
        // Vc = R*Tc / Pc per mol → divide by molar mass
        return EquationOfState.R * Tc / Pc / M;
    }

    /**
     * @param T Temperature K
     * @return specific entropy for an ideal gaz [J/Kg*K]
     */
    double idealGasEntropy(double T, double P) {
        if (T < 0) T = 0.1f;
        if (P < 0) P = 0.1f;

        double Cp = 3.5 * R;

        return (Cp * Math.log(T) - R * Math.log(P))/M;
    }

    public abstract double fugacityCoefficient(double T, double P, double Vm_mass);

    // ---------- Saturation pressure ----------
    @Override
    public double liquidVolume(double T, double P) {
        Couple<Double> sat = saturationVolumes(T);
        return sat.getFirst();
    }

    @Override
    public double vaporVolume(double T, double P) {
        Couple<Double> sat = saturationVolumes(T);
        return sat.getSecond();
    }

    public Couple<Double> saturationVolumes(double T) {
        if (T >= Tc) {
            double Vm = criticalVolumeMass();
            return Couple.create(Vm, Vm);
        }

        // Compute spinodals
        double[] spinodals = findSpinodalPoints(T).stream().mapToDouble(Double::doubleValue).toArray();
        double Pmin = Math.max(1e-6, pressure(T, spinodals[0]));
        double Pmax = pressure(T, spinodals[1]);

        // Solve for equality of fugacities
        java.util.function.DoubleFunction<Couple<Double>> solver = (P) -> {
            double Vl = spinodals[0];
            double Vv = spinodals[1];
            double phil = fugacityCoefficient(T, P, Vl);
            double phiv = fugacityCoefficient(T, P, Vv);
            if (phil <= 0 || phiv <= 0) return Couple.create(Double.NaN, Double.NaN);
            return Couple.create(Vl, Vv);
        };

        return solver.apply(saturationPressure(T));
    }
    /**
     * @param T temperature [K]
     * @param Vm_mass molar volume [m^3/Kg] -> TODO go toward a by mass volume
     * @return specific entropy [J/Kg·K]
     */
    public final double totalEntropy(double T, double Vm_mass) {
        double SRef = idealGasEntropy(T_REF, P_REF) + residualEntropy(T_REF,  liquidVolume(T_REF, P_REF));

        double P,Vl, Vv, x;

        // Single-phase or supercritical
        if (T >= Tc) {
            P = pressure(T, Vm_mass);
            return residualEntropy(T, Vm_mass) + idealGasEntropy(T, P) - SRef;
        }

        // Two-phase region
        P = saturationPressure(T);
        Vl = liquidVolume(T, P);
        Vv = vaporVolume(T, P);

        if (Vm_mass <= Vl) {
            // compressed liquid
            return residualEntropy(T, Vm_mass) + idealGasEntropy(T, pressure(T, Vm_mass)) - SRef;
        }

        if (Vm_mass >= Vv) {
            // superheated vapor
            return residualEntropy(T, Vm_mass) + idealGasEntropy(T, pressure(T, Vm_mass)) - SRef;
        }

        // Two-phase interpolation by vapor mass fraction
        x = (Vm_mass - Vl) / (Vv - Vl);
        x = Math.max(0.0, Math.min(1.0, x));

        double S_l = residualEntropy(T, Vl);
        double S_v = residualEntropy(T, Vv);

        return (1 - x) * S_l + x * S_v - SRef + idealGasEntropy(T, P);
    }

    /**
     * Total specific enthalpy [J/kg] at given T [K] and mass volume [m³/kg].
     * Handles single-phase, supercritical, and two-phase regions.
     */
    public final double totalEnthalpy(double T, double Vm_mass) {
        // Reference at T_REF, P_REF
        double HRef = idealGasEnthalpy(T_REF) + residualEnthalpy(T_REF, liquidVolume(T_REF, P_REF));

        double P, Vl, Vv, x;

        // --- Single-phase or supercritical ---
        if (T >= Tc) {
            P = pressure(T, Vm_mass);
            return residualEnthalpy(T, Vm_mass) + idealGasEnthalpy(T) - HRef;
        }

        // --- Two-phase region ---
        P = saturationPressure(T);
        Vl = liquidVolume(T, P);
        Vv = vaporVolume(T, P);

        if (Vm_mass <= Vl) {
            // Compressed liquid
            P = pressure(T, Vm_mass);
            return residualEnthalpy(T, Vm_mass) + idealGasEnthalpy(T) - HRef;
        }

        if (Vm_mass >= Vv) {
            // Superheated vapor
            P = pressure(T, Vm_mass);
            return residualEnthalpy(T, Vm_mass) + idealGasEnthalpy(T) - HRef;
        }

        // --- Two-phase interpolation ---
        x = (Vm_mass - Vl) / (Vv - Vl);
        x = Math.max(0.0, Math.min(1.0, x));

        double H_l = residualEnthalpy(T, Vl) + idealGasEnthalpy(T);
        double H_v = residualEnthalpy(T, Vv) + idealGasEnthalpy(T);

        return (1 - x) * H_l + x * H_v - HRef;
    }

    /**
     *
     * @param T Temperature K
     * @return specific enthalpy for an ideal gaz [J/Kg]
     */
    protected final double idealGasEnthalpy(double T) {
        double Cp = 75.0; // J/mol·K
        return Cp * (T)/M;
    }

    abstract double[] getZFactors(double T, double P);

}

